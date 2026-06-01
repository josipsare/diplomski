"""Build the quarterly + annual panels from raw SEC and stock data.

Usage:
    python scripts/build_panel.py            # full run, all 44 quarters + all tickers
    python scripts/build_panel.py --quick    # one quarter + 5 tickers, sanity check
    python scripts/build_panel.py --quarters 2024q3 2024q4

Outputs:
    data/output/panels/long_sec.parquet         (raw long-format SEC records)
    data/output/panels/panel_quarterly.parquet  (wide quarterly panel)
    data/output/panels/panel_annual.parquet     (annual panel + joined stock features)
    data/output/panels/stock_annual.parquet     (annual stock features alone)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running the script directly: add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.panel_builder import (
    build_annual_panel,
    build_quarterly_panel,
    join_annual_with_stock,
    save_panel,
)
from src.data.sec_loader import (
    cik_to_sector_mapping,
    load_long_panel,
    load_universe_from_companies_csv,
)
from src.data.stock_loader import annualize_all
from src.features.financial_ratios import RATIO_COLUMNS, compute_all_ratios
from src.features.normalization import (
    fill_missing,
    make_missingness_mask,
    standardize_cross_sectionally,
)
from src.utils.paths import (
    INPUT_DIR,
    PANEL_DIR,
    SEC_DATA_DIR,
    STOCK_DATA_DIR,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Tiny run: 1 quarter + 5 tickers for sanity-check")
    p.add_argument("--quarters", nargs="*", default=None, help="Subset of quarters (e.g. 2024q3 2024q4)")
    p.add_argument("--tickers", nargs="*", default=None, help="Subset of stock tickers")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--universe", choices=["sp500", "russell3000"], default="sp500",
                   help="Which universe to use (sp500=companies.csv, russell3000=companies_russell3000.csv). "
                        "Default sp500 matches the bachelor's-thesis universe.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    quarters = args.quarters
    tickers = args.tickers
    if args.quick:
        quarters = quarters or ["2024q4"]
        tickers = tickers or ["AAPL", "MSFT", "ORCL", "MPW", "PM"]
        log.info("Running in --quick mode: quarters=%s tickers=%s", quarters, tickers)

    companies_csv = (
        INPUT_DIR / "companies_russell3000.csv" if args.universe == "russell3000"
        else INPUT_DIR / "companies.csv"
    )
    log.info("Universe: %s (file: %s)", args.universe, companies_csv.name)
    universe = load_universe_from_companies_csv(companies_csv)
    log.info("Universe size: %d CIKs", len(universe))

    show_progress = not args.no_progress

    log.info("Loading SEC long-format records …")
    long_df = load_long_panel(
        SEC_DATA_DIR,
        universe_ciks=universe,
        quarters=quarters,
        show_progress=show_progress,
    )
    log.info("Long-format records: %d rows", len(long_df))
    save_panel(long_df, PANEL_DIR / "long_sec.parquet")

    log.info("Deriving SIC sector mapping per CIK …")
    sector_map = cik_to_sector_mapping(long_df)
    log.info("Sector map: %d CIKs across %d SIC-2digit sectors",
             len(sector_map), sector_map["sector_2digit"].nunique() if not sector_map.empty else 0)
    save_panel(sector_map, PANEL_DIR / "sector_map.parquet")

    log.info("Building quarterly panel …")
    panel_q = build_quarterly_panel(long_df, sector_map=sector_map)
    log.info("Quarterly panel: %d rows × %d cols", *panel_q.shape)
    save_panel(panel_q, PANEL_DIR / "panel_quarterly.parquet")

    log.info("Building annual panel …")
    panel_a = build_annual_panel(long_df, sector_map=sector_map)
    log.info("Annual panel: %d rows × %d cols", *panel_a.shape)

    log.info("Loading stock data …")
    stock_a = annualize_all(STOCK_DATA_DIR, tickers=tickers, show_progress=show_progress)
    log.info("Stock annual features: %d rows", len(stock_a))
    save_panel(stock_a, PANEL_DIR / "stock_annual.parquet")

    log.info("Joining annual financial panel with stock features …")
    bridge = pd.read_csv(companies_csv, dtype=str).rename(columns=str.lower)
    if "symbol" not in bridge.columns:
        # Fall back to a column whose values look like tickers
        cand = [c for c in bridge.columns if c not in ("cik", "company_name")]
        if cand:
            bridge = bridge.rename(columns={cand[0]: "symbol"})
    bridge["cik"] = pd.to_numeric(bridge["cik"], errors="coerce").astype("Int64")

    annual_joined = join_annual_with_stock(panel_a, stock_a, bridge)
    log.info("Annual panel (joined): %d rows × %d cols", *annual_joined.shape)
    save_panel(annual_joined, PANEL_DIR / "panel_annual.parquet")

    # ----- Feature engineering layer: ratios + per-period z-score -----
    log.info("Computing financial ratios on quarterly panel …")
    panel_q_sorted = panel_q.sort_values(["cik", "period_end"]).reset_index(drop=True)
    ratios_q = compute_all_ratios(panel_q_sorted, group_key="cik", time_col="period_end")
    panel_q_with_ratios = pd.concat(
        [panel_q_sorted[["cik", "period_end", "fiscal_year", "fiscal_quarter"]
                        + (["sic", "sector_2digit", "sector_label"] if "sic" in panel_q_sorted.columns else [])],
         ratios_q],
        axis=1,
    )

    panel_q_with_ratios["_period_bucket"] = (
        panel_q_with_ratios["fiscal_year"].astype(str) + "Q" +
        panel_q_with_ratios["fiscal_quarter"].astype(str)
    )

    # DEFAULT: sector-aware standardization (per period × SIC-1 division).
    # SIC-1 (division) is broader than SIC-2 because per-SIC-2 buckets are too
    # thin (median ~7 firms over 442 universe). Sector-aware is the default per
    # decision 2026-05-09: the period-only variant bakes sector confounding
    # into every detector (banks vs tech vs REITs have radically different
    # ratio means) and was flagged as a "hidden bug" risk by the holistic reviewer.
    if "sector_2digit" in panel_q_with_ratios.columns:
        log.info("Standardizing per (fiscal_year, fiscal_quarter, SIC-1) — DEFAULT …")
        sector_panel = panel_q_with_ratios.copy()
        sector_panel["sector_1digit"] = sector_panel["sector_2digit"].str[:1]
        standardized = standardize_cross_sectionally(
            sector_panel,
            feature_columns=RATIO_COLUMNS,
            period_col="_period_bucket",
            sector_col="sector_1digit",
        ).drop(columns=["_period_bucket", "sector_1digit"])
    else:
        log.warning(
            "sector_2digit missing from panel; falling back to period-only standardization."
        )
        standardized = standardize_cross_sectionally(
            panel_q_with_ratios,
            feature_columns=RATIO_COLUMNS,
            period_col="_period_bucket",
        ).drop(columns="_period_bucket")
    save_panel(standardized, PANEL_DIR / "panel_quarterly_ratios_standardized.parquet")

    # ABLATION: period-only standardization. Persisted for Phase 6 ablation
    # comparing sector-aware vs period-only z-scores.
    log.info("Standardizing per (fiscal_year, fiscal_quarter) only — ablation sibling …")
    standardized_period_only = standardize_cross_sectionally(
        panel_q_with_ratios,
        feature_columns=RATIO_COLUMNS,
        period_col="_period_bucket",
    ).drop(columns="_period_bucket")
    save_panel(
        standardized_period_only,
        PANEL_DIR / "panel_quarterly_ratios_standardized_period_only.parquet",
    )

    # Persist a mask + fill-zero version of the DEFAULT (sector-aware) panel
    # for direct DL consumption. Phase 4-5 detector training reads these.
    mask = make_missingness_mask(standardized, RATIO_COLUMNS)
    filled = fill_missing(standardized, RATIO_COLUMNS, fill_value=0.0)
    save_panel(filled, PANEL_DIR / "panel_quarterly_ratios_filled.parquet")
    save_panel(
        pd.concat([standardized[["cik", "period_end"]], mask.add_suffix("_missing")], axis=1),
        PANEL_DIR / "panel_quarterly_missingness_mask.parquet",
    )

    log.info("Done. Outputs in %s", PANEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

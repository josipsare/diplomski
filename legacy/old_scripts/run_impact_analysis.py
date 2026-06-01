"""Run Phase 7 impact analysis: event-study + forward-outcome regression
across all detectors + ensembles. This answers the second half of the
thesis topic: do detector-flagged anomalies have measurable downstream
effects on company business outcomes?

Outputs (all in `data/output/scores/`):
    impact_event_study.parquet         per-detector mean CAR/BHAR around
                                       filed_date for top-K anomalous firms
    impact_event_study_random.parquet  random-firm baseline for comparison
    impact_forward_outcome.parquet     OLS regression of next-year outcome
                                       on this-year anomaly score
    impact_per_event_cars.parquet      raw per-event CARs (long format) for
                                       diagnostic / case-study work
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impact.event_study import (
    compare_detectors_event_study,
    event_study_baseline,
)
from src.impact.forward_outcome import compare_detectors_forward_outcome
from src.models.ensemble import (
    ensemble_rank_average,
    ensemble_score_zscore,
    ensemble_top_k_union,
)
from src.utils.paths import OUTPUT_DIR, PANEL_DIR, SCORES_DIR

log = logging.getLogger(__name__)
LABELS_DIR = OUTPUT_DIR / "labels"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["sp500", "russell3000"], default="russell3000")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--window-lo", type=int, default=-30)
    p.add_argument("--window-hi", type=int, default=60)
    p.add_argument("--spec", choices=["raw", "market", "capm"], default="capm")
    return p.parse_args()


def annualize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out["fiscal_year"] = pd.to_datetime(out["period_end"]).dt.year
    annual = out.groupby(["cik", "fiscal_year"], as_index=False)["score"].max()
    annual["model_name"] = name
    return annual


def load_scores(suffix: str = "") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    bf_path = SCORES_DIR / f"benford_first_digit{suffix}.parquet"
    if bf_path.exists():
        bf = pd.read_parquet(bf_path)
        out["benford_naive_+MAD"] = bf
        bf_inv = bf.copy()
        if "score_inverted" in bf_inv.columns:
            bf_inv["score"] = bf_inv["score_inverted"]
        else:
            bf_inv["score"] = -bf_inv["score"]
        bf_inv["model_name"] = "benford_inverted_-MAD"
        out["benford_inverted_-MAD"] = bf_inv

    for name in ("isolation_forest", "autoencoder", "vae",
                 "lstm_autoencoder", "transformer_encoder"):
        path = SCORES_DIR / f"{name}{suffix}.parquet"
        if path.exists():
            out[name] = annualize(pd.read_parquet(path), name)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    suffix = f"_{args.universe}"

    log.info("Loading detector scores …")
    score_dfs = load_scores(suffix)
    log.info("Loaded %d detectors", len(score_dfs))

    # Build ensembles from smart detectors only
    smart_keys = [k for k in score_dfs if not k.startswith("benford")]
    smart_dfs = {k: score_dfs[k] for k in smart_keys}
    if len(smart_dfs) >= 2:
        log.info("Building ensembles …")
        score_dfs["ensemble_rank_avg"] = ensemble_rank_average(smart_dfs)
        score_dfs["ensemble_zscore_avg"] = ensemble_score_zscore(smart_dfs)
        score_dfs["ensemble_top100_union"] = ensemble_top_k_union(smart_dfs, k=100)

    log.info("Loading impact data layer …")
    daily = pd.read_parquet(PANEL_DIR / "daily_returns.parquet")
    market = pd.read_parquet(PANEL_DIR / "market_returns.parquet")
    long_sec = pd.read_parquet(PANEL_DIR / "long_sec.parquet")
    panel_annual = pd.read_parquet(PANEL_DIR / "panel_annual.parquet")
    cik_to_ticker = pd.read_csv(PROJECT_ROOT / "data" / "input" /
                                ("companies_russell3000.csv" if args.universe == "russell3000"
                                 else "companies.csv"), dtype=str)
    log.info("daily=%d, market=%d, long_sec=%d, panel_annual=%d, ciks=%d",
             len(daily), len(market), len(long_sec), len(panel_annual), len(cik_to_ticker))

    # ----- Event study: top-K of each detector -----
    log.info("Running event study (window=(%d, %d), spec=%s, k=%d) …",
             args.window_lo, args.window_hi, args.spec, args.top_k)
    summary, per_event = compare_detectors_event_study(
        score_dfs, daily, market, long_sec, cik_to_ticker,
        k=args.top_k, window=(args.window_lo, args.window_hi), spec=args.spec,
    )
    summary["universe"] = args.universe
    summary.to_parquet(SCORES_DIR / f"impact_event_study_{args.universe}.parquet", index=False)
    log.info("Event study summary written")

    # Random baseline
    log.info("Computing random-firm baseline …")
    rand = event_study_baseline(
        panel_annual, daily, market, long_sec, cik_to_ticker,
        n_random=args.top_k, window=(args.window_lo, args.window_hi), spec=args.spec,
    )
    rand_row = pd.DataFrame([{
        "detector": "random_baseline",
        "n_top_k": args.top_k,
        "n_events_with_filed_date": rand.n_events,
        "n_car_computed": rand.n_completed,
        "mean_car": rand.mean_car,
        "median_car": rand.median_car,
        "std_car": rand.std_car,
        "t_stat": rand.t_stat,
        "p_value": rand.p_value,
        "pct_negative_car": rand.pct_negative,
        "mean_bhar": rand.mean_bhar,
        "universe": args.universe,
    }])
    summary_with_baseline = pd.concat([summary, rand_row], ignore_index=True)

    print()
    print(f"=== EVENT STUDY: top-{args.top_k} {args.spec}-CAR ({args.window_lo}, {args.window_hi}) ===")
    print(summary_with_baseline.sort_values("mean_car").to_string(index=False))

    # Save per-event CARs (for case studies)
    if per_event:
        all_ev = pd.concat(per_event.values(), ignore_index=True)
        all_ev.to_parquet(SCORES_DIR / f"impact_per_event_cars_{args.universe}.parquet", index=False)
        log.info("Per-event CARs written (%d rows)", len(all_ev))

    # ----- Forward-outcome regression -----
    log.info("Running forward-outcome regressions (lag=1) …")
    fwd = compare_detectors_forward_outcome(
        score_dfs, panel_annual,
        outcomes=("annual_return_market", "volatility_market",
                  "max_drawdown_market", "volume_growth_market"),
        forward_lag=1,
    )
    fwd["universe"] = args.universe
    fwd.to_parquet(SCORES_DIR / f"impact_forward_outcome_{args.universe}.parquet", index=False)
    print()
    print(f"=== FORWARD-OUTCOME REGRESSION (1-year ahead, cluster-SE on cik) ===")
    print(fwd.to_string(index=False))

    log.info("Done. Phase 7 outputs in %s", SCORES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

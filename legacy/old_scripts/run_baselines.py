"""Run Phase 3 baseline detectors and persist scores.

Each detector reads from `data/output/panels/` and writes a per-detector
Parquet to `data/output/scores/`:

    benford_first_digit.parquet
        cik, fiscal_year, score (= MAD), mad, chi_square, p_value,
        ks_stat, n_samples, model_name

    isolation_forest.parquet
        cik, period_end, score, model_name

Usage:
    python scripts/run_baselines.py                # all baselines
    python scripts/run_baselines.py --only benford
    python scripts/run_baselines.py --only iforest
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

from src.features.financial_ratios import RATIO_COLUMNS
from src.models.benford_baseline import BenfordDetector
from src.models.isolation_forest import IsolationForestDetector
from src.utils.paths import PANEL_DIR, SCORES_DIR

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["benford", "iforest", "all"], default="all")
    return p.parse_args()


def run_benford() -> None:
    log.info("Loading long_sec.parquet for Benford baseline …")
    long_df = pd.read_parquet(PANEL_DIR / "long_sec.parquet")
    log.info("Long-format rows: %d", len(long_df))

    det = BenfordDetector()
    det.fit(None)
    scores = det.score_panel(long_df)
    log.info("Benford scored %d (cik, fiscal_year) pairs", len(scores))

    out_path = SCORES_DIR / f"{det.name}.parquet"
    scores.to_parquet(out_path, index=False)
    log.info("Wrote %s", out_path)

    # Summary stats — useful sanity check
    log.info("Benford MAD distribution: mean=%.2f median=%.2f P90=%.2f",
             scores.mad.mean(), scores.mad.median(), scores.mad.quantile(0.9))


def run_iforest() -> None:
    log.info("Loading panel_quarterly_ratios_filled.parquet for Isolation Forest …")
    panel = pd.read_parquet(PANEL_DIR / "panel_quarterly_ratios_filled.parquet")
    log.info("Panel rows: %d × %d cols", *panel.shape)

    feature_cols = [c for c in RATIO_COLUMNS if c in panel.columns]
    log.info("Using %d feature columns", len(feature_cols))

    det = IsolationForestDetector(feature_columns=feature_cols)
    det.fit(panel)

    scored = det.score_dataframe(panel, feature_cols=feature_cols)
    out_path = SCORES_DIR / f"{det.name}.parquet"
    scored.to_parquet(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(scored))
    log.info("IF score distribution: mean=%.4f std=%.4f min=%.4f max=%.4f",
             scored.score.mean(), scored.score.std(),
             scored.score.min(), scored.score.max())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.only in ("benford", "all"):
        run_benford()
    if args.only in ("iforest", "all"):
        run_iforest()

    log.info("Done. Scores in %s", SCORES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""5-seed replication of the NN forward-outcome regressor.

R2 holistic review (2026-05-11) recommended this as the highest-value next
move after growth partial-out: with delta_r2 values now in the +0.001–+0.009
range, single-seed noise could flip signs on borderline cells. The 5-seed
replication produces mean ± std for every (detector × outcome) cell,
bulletproofing the headline against a "what if you got lucky?" challenge.

Compute budget: 4 detectors × 4 outcomes × 5 seeds = 80 NN trainings (each
trains 2 nets — full + baseline — so 160 actual fits). ~10 min per single
seed run → ~50 min total.

Output:
    data/output/scores/nn_impact_5seed_{universe}.parquet
        — columns: detector, outcome, seed, n_train, n_test, test_r2,
          test_r2_baseline, delta_r2, perm_importance
    data/output/scores/nn_impact_5seed_summary_{universe}.parquet
        — pivoted: detector × outcome × {mean_delta_r2, std_delta_r2,
          mean_perm, std_perm, n_seeds_positive_delta}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impact.forward_outcome import forward_outcome_regression
from src.utils.paths import PANEL_DIR, SCORES_DIR
from src.utils.scores import annualize

log = logging.getLogger(__name__)

NN_DETECTORS = ("autoencoder", "vae", "lstm_autoencoder", "transformer_encoder")
OUTCOMES = ("annual_return_market", "volatility_market",
            "max_drawdown_market", "volume_growth_market")
SEEDS = (42, 7, 123, 2024, 9999)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["sp500", "russell3000"], default="russell3000")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--test-year", type=int, default=2022)
    return p.parse_args()


def load_nn_scores(suffix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in NN_DETECTORS:
        path = SCORES_DIR / f"{name}{suffix}.parquet"
        if path.exists():
            out[name] = annualize(pd.read_parquet(path), name)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    suffix = f"_{args.universe}"

    log.info("Loading NN detector scores …")
    score_dfs = load_nn_scores(suffix)
    log.info("Loaded %d NN detectors: %s", len(score_dfs), list(score_dfs.keys()))

    log.info("Loading panel_annual …")
    panel_annual = pd.read_parquet(PANEL_DIR / "panel_annual.parquet")

    n_cells = len(score_dfs) * len(OUTCOMES) * len(SEEDS)
    log.info("Total cells to compute: %d  (%d detectors × %d outcomes × %d seeds)",
             n_cells, len(score_dfs), len(OUTCOMES), len(SEEDS))

    rows = []
    cell_idx = 0
    for detector_name, scores in score_dfs.items():
        for outcome in OUTCOMES:
            if outcome not in panel_annual.columns:
                continue
            for seed in SEEDS:
                cell_idx += 1
                log.info("[%d/%d] %s × %s × seed=%d …",
                         cell_idx, n_cells, detector_name, outcome, seed)
                try:
                    res = forward_outcome_regression(
                        scores, panel_annual,
                        detector_name=detector_name,
                        outcome_col=outcome,
                        forward_lag=1,
                        test_year_threshold=args.test_year,
                        epochs=args.epochs,
                        seed=seed,
                    )
                    rows.append({
                        "detector": detector_name,
                        "outcome": outcome,
                        "seed": seed,
                        "n_train": res.n_train,
                        "n_test": res.n_test,
                        "test_r2": res.test_r2,
                        "test_r2_baseline": res.test_r2_without_score,
                        "delta_r2": res.delta_r2,
                        "perm_importance": res.perm_importance,
                    })
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed: %s × %s × seed=%d: %s",
                                detector_name, outcome, seed, exc)

    raw = pd.DataFrame(rows)
    raw["universe"] = args.universe
    raw_path = SCORES_DIR / f"nn_impact_5seed_{args.universe}.parquet"
    raw.to_parquet(raw_path, index=False)
    log.info("Raw 5-seed output: %s (%d rows)", raw_path, len(raw))

    # Summary: mean / std / n_seeds_positive across seeds per (detector, outcome)
    summary = (
        raw.groupby(["detector", "outcome"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_delta_r2=("delta_r2", "mean"),
            std_delta_r2=("delta_r2", "std"),
            min_delta_r2=("delta_r2", "min"),
            max_delta_r2=("delta_r2", "max"),
            n_seeds_positive=("delta_r2", lambda s: int((s > 0).sum())),
            mean_perm=("perm_importance", "mean"),
            std_perm=("perm_importance", "std"),
        )
    )
    summary["universe"] = args.universe
    summary_path = SCORES_DIR / f"nn_impact_5seed_summary_{args.universe}.parquet"
    summary.to_parquet(summary_path, index=False)
    log.info("Summary: %s", summary_path)

    print()
    print(f"=== 5-SEED NN FORWARD-OUTCOME SUMMARY ({args.universe}) ===")
    cols = ["detector", "outcome", "n_seeds", "mean_delta_r2", "std_delta_r2",
            "min_delta_r2", "max_delta_r2", "n_seeds_positive", "mean_perm"]
    print(summary.sort_values(["outcome", "mean_delta_r2"], ascending=[True, False])[cols]
          .to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict NN-only impact analysis: NN anomaly detectors → NN forward-outcome
regressor.

Pipeline (every component is a neural network):
    1. Load score files from 4 NN detectors (AE, VAE, LSTM-AE, Transformer)
    2. For each (detector × outcome × forward_lag):
       - Build (cik, year) panel of (anomaly_score_t, controls_t, outcome_{t+lag})
       - Temporal train/test split (train < 2022, test ≥ 2022)
       - Train MLP regressor on train set, evaluate on test set
       - Compute delta_r2 = R²(with score) - R²(controls only)
       - Compute permutation importance of anomaly_score input
    3. Tabulate results.

Headline number per (detector, outcome): delta_r2. If positive and large,
the unsupervised NN detector's score carries genuine predictive value for
the next-year business outcome beyond what sector + size already provide.
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

from src.impact.forward_outcome import compare_detectors_forward_outcome
from src.utils.paths import PANEL_DIR, SCORES_DIR
from src.utils.scores import annualize

log = logging.getLogger(__name__)

NN_DETECTORS = ("autoencoder", "vae", "lstm_autoencoder", "transformer_encoder")


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
        else:
            log.warning("Missing %s — skipping", path.name)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    suffix = f"_{args.universe}"

    log.info("Loading NN detector scores (strict NN-only) …")
    score_dfs = load_nn_scores(suffix)
    log.info("Loaded %d NN detectors: %s", len(score_dfs), list(score_dfs.keys()))

    log.info("Loading panel_annual …")
    panel_annual = pd.read_parquet(PANEL_DIR / "panel_annual.parquet")
    log.info("Panel: %d firm-years", len(panel_annual))

    log.info("Running NN forward-outcome regressor (epochs=%d, test_year≥%d) …",
             args.epochs, args.test_year)
    fwd = compare_detectors_forward_outcome(
        score_dfs, panel_annual,
        outcomes=("annual_return_market", "volatility_market",
                  "max_drawdown_market", "volume_growth_market"),
        forward_lag=1, test_year_threshold=args.test_year, epochs=args.epochs,
    )
    fwd["universe"] = args.universe
    fwd.to_parquet(SCORES_DIR / f"nn_impact_forward_outcome_{args.universe}.parquet",
                   index=False)

    print()
    print(f"=== NN FORWARD-OUTCOME REGRESSION (strict NN-only, test >= {args.test_year}) ===")
    if not fwd.empty:
        cols = ["detector", "outcome", "n_train", "n_test", "test_r2",
                "test_r2_baseline", "delta_r2", "perm_importance"]
        print(fwd[cols].to_string(index=False))
    else:
        print("No results.")

    log.info("Done. Output: %s", SCORES_DIR / f"nn_impact_forward_outcome_{args.universe}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

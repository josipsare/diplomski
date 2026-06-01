"""Strict NN-only supervised evaluation. Runs only the 4 NN detectors
(AE, VAE, LSTM-AE, Transformer) against restatement labels.

Outputs:
    data/output/scores/eval_nn_only_{universe}.parquet      — comparison table
    data/output/scores/agreement_nn_kendall_{universe}.parquet  — Kendall tau
    data/output/scores/agreement_nn_jaccard_at100_{universe}.parquet — Jaccard@100

Excluded by design (moved to legacy/): Benford, Isolation Forest, ensembles.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.supervised_eval import compare_detectors
from src.utils.paths import OUTPUT_DIR, SCORES_DIR
from src.utils.scores import annualize

log = logging.getLogger(__name__)
LABELS_DIR = OUTPUT_DIR / "labels"

NN_DETECTORS = ("autoencoder", "vae", "lstm_autoencoder", "transformer_encoder")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["sp500", "russell3000"], default="russell3000")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    return p.parse_args()


def load_nn_scores(suffix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in NN_DETECTORS:
        path = SCORES_DIR / f"{name}{suffix}.parquet"
        if path.exists():
            out[name] = annualize(pd.read_parquet(path), name)
    return out


def kendall_matrix(score_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = list(score_dfs.keys())
    mat = pd.DataFrame(np.eye(len(rows)), index=rows, columns=rows, dtype=float)
    for a, b in combinations(rows, 2):
        df_a = score_dfs[a][["cik", "fiscal_year", "score"]].rename(columns={"score": "score_a"})
        df_b = score_dfs[b][["cik", "fiscal_year", "score"]].rename(columns={"score": "score_b"})
        joined = df_a.merge(df_b, on=["cik", "fiscal_year"], how="inner").dropna()
        tau = float("nan") if len(joined) < 10 else float(kendalltau(joined["score_a"], joined["score_b"])[0])
        mat.loc[a, b] = mat.loc[b, a] = tau
    return mat


def jaccard_at_k_matrix(score_dfs: dict[str, pd.DataFrame], k: int = 100) -> pd.DataFrame:
    rows = list(score_dfs.keys())
    mat = pd.DataFrame(np.eye(len(rows)), index=rows, columns=rows, dtype=float)
    top_sets: dict[str, set[tuple[int, int]]] = {}
    for name, df in score_dfs.items():
        top_k = df.dropna(subset=["score"]).nlargest(k, "score")
        top_sets[name] = set(zip(top_k["cik"].astype(int), top_k["fiscal_year"].astype(int)))
    for a, b in combinations(rows, 2):
        sa, sb = top_sets[a], top_sets[b]
        jacc = float("nan") if not (sa and sb) else len(sa & sb) / len(sa | sb)
        mat.loc[a, b] = mat.loc[b, a] = float(jacc)
    return mat


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    suffix = f"_{args.universe}"

    log.info("Loading strict NN-only scores …")
    score_dfs = load_nn_scores(suffix)
    log.info("Loaded %d NN detectors: %s", len(score_dfs), list(score_dfs.keys()))

    labels = pd.read_parquet(LABELS_DIR / "restatement_labels.parquet")

    log.info("Running supervised eval against restatement labels …")
    table = compare_detectors(
        score_dfs, labels, label_col="is_restatement",
        k_values=(10, 50, 100, 500),
    )
    table["universe"] = args.universe
    table.to_parquet(SCORES_DIR / f"eval_nn_only_{args.universe}.parquet", index=False)

    print()
    print(f"=== STRICT NN-ONLY EVAL ({args.universe}, restatement labels) ===")
    print(table[["detector", "n", "n_pos", "base_rate", "roc_auc", "pr_auc",
                  "precision@10", "precision@100", "lift@100"]].to_string(index=False))

    log.info("Computing NN-only agreement matrices …")
    kt = kendall_matrix(score_dfs)
    kt.to_parquet(SCORES_DIR / f"agreement_nn_kendall_{args.universe}.parquet")
    print()
    print("=== Kendall tau (NN detectors only) ===")
    print(kt.round(3).to_string())

    jacc = jaccard_at_k_matrix(score_dfs, k=100)
    jacc.to_parquet(SCORES_DIR / f"agreement_nn_jaccard_at100_{args.universe}.parquet")
    print()
    print("=== Jaccard@100 (NN detectors only) ===")
    print(jacc.round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

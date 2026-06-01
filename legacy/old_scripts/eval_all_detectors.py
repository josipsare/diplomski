"""Final supervised eval across all detectors + top-K agreement matrix.

Loads all per-detector score files, joins on the restatement label table,
runs `compare_detectors` with bootstrap CIs, and computes a Kendall-tau /
Jaccard agreement matrix at top-K to characterize how detector rankings
relate to one another (Phase 6 ablation: do classical and DL detectors flag
the same firms, or different ones?).

Outputs:
    data/output/scores/eval_phase5_full.parquet     # comparison table
    data/output/scores/agreement_kendall.parquet    # Kendall tau matrix
    data/output/scores/agreement_jaccard_at100.parquet  # Jaccard@K=100 matrix
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
from src.models.ensemble import (
    ensemble_rank_average,
    ensemble_score_zscore,
    ensemble_top_k_union,
)
from src.utils.paths import OUTPUT_DIR, SCORES_DIR

log = logging.getLogger(__name__)
LABELS_DIR = OUTPUT_DIR / "labels"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["sp500", "russell3000"], default="russell3000")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    return p.parse_args()


def annualize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out["fiscal_year"] = pd.to_datetime(out["period_end"]).dt.year
    annual = out.groupby(["cik", "fiscal_year"], as_index=False)["score"].max()
    annual["model_name"] = name
    return annual


def load_scores(universe_suffix: str = "") -> dict[str, pd.DataFrame]:
    """Load per-detector scores. `universe_suffix` is "_sp500" or
    "_russell3000" if universe-specific saved scores exist; empty for the
    current `data/output/scores/{model}.parquet`."""
    out: dict[str, pd.DataFrame] = {}

    bf_path = SCORES_DIR / f"benford_first_digit{universe_suffix}.parquet"
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
        path = SCORES_DIR / f"{name}{universe_suffix}.parquet"
        if path.exists():
            out[name] = annualize(pd.read_parquet(path), name)
        else:
            log.warning("Missing %s — skipping", path.name)

    return out


def kendall_matrix(score_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise Kendall tau between detector ranks on the intersection of
    (cik, fiscal_year) keys."""
    rows = list(score_dfs.keys())
    mat = pd.DataFrame(np.eye(len(rows)), index=rows, columns=rows, dtype=float)
    for a, b in combinations(rows, 2):
        df_a = score_dfs[a][["cik", "fiscal_year", "score"]].rename(columns={"score": "score_a"})
        df_b = score_dfs[b][["cik", "fiscal_year", "score"]].rename(columns={"score": "score_b"})
        joined = df_a.merge(df_b, on=["cik", "fiscal_year"], how="inner").dropna()
        if len(joined) < 10:
            tau = float("nan")
        else:
            tau, _ = kendalltau(joined["score_a"], joined["score_b"])
        mat.loc[a, b] = mat.loc[b, a] = float(tau)
    return mat


def jaccard_at_k_matrix(
    score_dfs: dict[str, pd.DataFrame], k: int = 100,
) -> pd.DataFrame:
    """Jaccard similarity of top-K firm-years per detector pair."""
    rows = list(score_dfs.keys())
    mat = pd.DataFrame(np.eye(len(rows)), index=rows, columns=rows, dtype=float)
    top_sets: dict[str, set[tuple[int, int]]] = {}
    for name, df in score_dfs.items():
        top_k = df.dropna(subset=["score"]).nlargest(k, "score")
        top_sets[name] = set(zip(top_k["cik"].astype(int), top_k["fiscal_year"].astype(int)))
    for a, b in combinations(rows, 2):
        sa, sb = top_sets[a], top_sets[b]
        if not sa or not sb:
            jacc = float("nan")
        else:
            jacc = len(sa & sb) / len(sa | sb)
        mat.loc[a, b] = mat.loc[b, a] = float(jacc)
    return mat


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    suffix = f"_{args.universe}"
    log.info("Loading scores for universe=%s", args.universe)
    score_dfs = load_scores(suffix)
    log.info("Loaded %d detectors: %s", len(score_dfs), list(score_dfs.keys()))

    labels_path = LABELS_DIR / "restatement_labels.parquet"
    labels = pd.read_parquet(labels_path)

    # Build ensembles from the smart detectors only (exclude Benford which is
    # uninformative — including it would dilute every ensemble strategy).
    smart_keys = [k for k in score_dfs if not k.startswith("benford")]
    smart_dfs = {k: score_dfs[k] for k in smart_keys}
    if len(smart_dfs) >= 2:
        log.info("Building ensembles from %d smart detectors …", len(smart_dfs))
        score_dfs["ensemble_rank_avg"] = ensemble_rank_average(smart_dfs)
        score_dfs["ensemble_zscore_avg"] = ensemble_score_zscore(smart_dfs)
        score_dfs["ensemble_top100_union"] = ensemble_top_k_union(smart_dfs, k=100)

    log.info("Computing supervised eval (bootstrap_n=%d) …", args.bootstrap_n)
    table = compare_detectors(
        score_dfs, labels, label_col="is_restatement",
        k_values=(10, 50, 100, 500),
    )
    table["universe"] = args.universe
    table_path = SCORES_DIR / f"eval_phase5_full_{args.universe}.parquet"
    table.to_parquet(table_path, index=False)
    log.info("Wrote %s", table_path)

    print()
    print(f"=== PHASE 5 HEADLINE TABLE ({args.universe}, restatement labels) ===")
    print(table[["detector", "n", "n_pos", "base_rate", "roc_auc", "pr_auc",
                  "precision@10", "precision@100", "lift@100"]].to_string(index=False))

    log.info("Computing Kendall tau agreement matrix …")
    kt = kendall_matrix(score_dfs)
    kt.to_parquet(SCORES_DIR / f"agreement_kendall_{args.universe}.parquet")
    print()
    print("=== Kendall tau (rank correlation between detectors) ===")
    print(kt.round(3).to_string())

    log.info("Computing Jaccard@100 matrix …")
    jacc = jaccard_at_k_matrix(score_dfs, k=100)
    jacc.to_parquet(SCORES_DIR / f"agreement_jaccard_at100_{args.universe}.parquet")
    print()
    print("=== Jaccard@100 (top-100 set overlap between detectors) ===")
    print(jacc.round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

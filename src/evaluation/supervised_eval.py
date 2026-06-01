"""Supervised evaluation of detector scores against binary fraud/restatement labels.

Joins a per-detector score table with a label table on (cik, fiscal_year),
then computes the standard supervised-anomaly-detection metrics:

    precision@K     fraction of top-K-scored firm-years that are positive labels
                    K ∈ {10, 50, 100, 500, 1000} reported by default
    ROC-AUC         area under ROC; threshold-free overall ranking quality
    PR-AUC          area under precision-recall; better than ROC for rare-event tasks
                    (restatement base rate ~3-15% depending on universe + period)
    lift@K          precision@K / base_rate; how much better than random the top-K is

Plus per-detector summary stats and a comparison table that ranks detectors
by PR-AUC (the rare-event-appropriate primary metric).

Usage:
    from src.evaluation.supervised_eval import evaluate_detector, compare_detectors
    scores = pd.read_parquet("data/output/scores/benford_first_digit.parquet")
    labels = pd.read_parquet("data/output/labels/restatement_labels.parquet")
    metrics = evaluate_detector(scores, labels, label_col="is_restatement")

The functions are framework-agnostic: any DataFrame with (cik, fiscal_year, score)
works, so this same harness will evaluate AE, VAE, LSTM-AE, Transformer in Phase 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """One detector's supervised evaluation against one label set."""

    detector_name: str
    label_name: str
    n_observations: int
    n_positive: int
    base_rate: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    lift_at_k: Dict[int, float] = field(default_factory=dict)
    n_positive_at_k: Dict[int, int] = field(default_factory=dict)
    notes: str = ""


def _sklearn_metrics():
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return roc_auc_score, average_precision_score
    except ImportError as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e


def _join_scores_with_labels(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
    score_col: str = "score",
    label_col: str = "is_restatement",
) -> pd.DataFrame:
    """Inner-join scores with labels on (cik, period). Negatives = panel rows
    not present in `labels` are inferred as label=False. Positives = present
    in labels with label_col=True.

    The semantics: the label table is sparse — only positive cases listed.
    A firm-year *missing* from the label table is treated as a negative label,
    NOT as missing data. This is correct for restatement labels (every
    panel row is either restated or not).
    """
    sub_scores = scores[[cik_col, period_col, score_col]].copy()
    sub_scores[cik_col] = sub_scores[cik_col].astype("Int64")
    sub_scores[period_col] = sub_scores[period_col].astype("Int64")

    # Reduce labels to one row per (cik, period) with the binary label only
    sub_labels = labels[[cik_col, period_col, label_col]].copy()
    sub_labels[cik_col] = sub_labels[cik_col].astype("Int64")
    sub_labels[period_col] = sub_labels[period_col].astype("Int64")
    sub_labels = sub_labels.groupby([cik_col, period_col], as_index=False)[label_col].any()

    out = sub_scores.merge(sub_labels, on=[cik_col, period_col], how="left")
    out[label_col] = out[label_col].fillna(False).astype(bool)
    return out


def evaluate_detector(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    detector_name: Optional[str] = None,
    label_name: str = "restatement",
    label_col: str = "is_restatement",
    score_col: str = "score",
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
    k_values: Sequence[int] = (10, 50, 100, 500, 1000),
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
) -> EvalResult:
    """Compute precision@K, ROC-AUC, PR-AUC for one detector × one label set."""
    roc_auc_score, average_precision_score = _sklearn_metrics()

    # Resolve detector name from the scores DataFrame if not provided.
    # `not Series.empty` works but `not Series` raises in pandas 2.x ambiguity.
    # Use len() check to be unambiguous.
    if detector_name is None:
        if "model_name" in scores.columns and len(scores) > 0:
            detector_name = str(scores["model_name"].iloc[0])
        else:
            detector_name = "unknown"

    # If period_col is not in the scores table, derive from period_end
    if period_col not in scores.columns and "period_end" in scores.columns:
        scores = scores.copy()
        scores[period_col] = pd.to_datetime(scores["period_end"]).dt.year

    joined = _join_scores_with_labels(
        scores, labels,
        cik_col=cik_col, period_col=period_col,
        score_col=score_col, label_col=label_col,
    )
    joined = joined.dropna(subset=[score_col]).copy()

    n = len(joined)
    n_pos = int(joined[label_col].sum())
    base_rate = n_pos / n if n > 0 else 0.0

    if n == 0 or n_pos == 0:
        return EvalResult(
            detector_name=detector_name, label_name=label_name,
            n_observations=n, n_positive=n_pos, base_rate=base_rate,
            roc_auc=None, pr_auc=None,
            notes="No positive labels in joined set; metrics undefined.",
        )

    y = joined[label_col].astype(int).to_numpy()
    s = joined[score_col].to_numpy()

    try:
        roc = float(roc_auc_score(y, s))
    except ValueError as e:
        roc = None
        log.warning("ROC-AUC failed for %s: %s", detector_name, e)
    try:
        pr = float(average_precision_score(y, s))
    except ValueError as e:
        pr = None
        log.warning("PR-AUC failed for %s: %s", detector_name, e)

    # precision@K — sort by score descending, take top K
    order = np.argsort(-s)
    y_sorted = y[order]
    precision_at_k: Dict[int, float] = {}
    n_pos_at_k: Dict[int, int] = {}
    lift_at_k: Dict[int, float] = {}
    for k in k_values:
        if k > n:
            continue
        top_k_pos = int(y_sorted[:k].sum())
        precision_at_k[k] = top_k_pos / k
        n_pos_at_k[k] = top_k_pos
        lift_at_k[k] = (precision_at_k[k] / base_rate) if base_rate > 0 else float("nan")

    result = EvalResult(
        detector_name=detector_name, label_name=label_name,
        n_observations=n, n_positive=n_pos, base_rate=base_rate,
        roc_auc=roc, pr_auc=pr,
        precision_at_k=precision_at_k, lift_at_k=lift_at_k,
        n_positive_at_k=n_pos_at_k,
    )

    # Optional bootstrap CIs — sample (y, s) with replacement bootstrap_n times
    # and report 5th/95th percentile for ROC, PR, precision@K. Tests whether
    # detectors that look indistinguishable point-wise (e.g. AE vs IF at
    # ROC=0.65) are statistically distinguishable.
    if bootstrap_n > 0:
        rng = np.random.default_rng(bootstrap_seed)
        roc_samples, pr_samples = [], []
        precat: Dict[int, list] = {k: [] for k in precision_at_k}
        for _ in range(bootstrap_n):
            idx = rng.integers(0, n, size=n)
            y_b, s_b = y[idx], s[idx]
            if y_b.sum() == 0 or y_b.sum() == n:
                continue
            try:
                roc_samples.append(float(roc_auc_score(y_b, s_b)))
            except ValueError:
                pass
            try:
                pr_samples.append(float(average_precision_score(y_b, s_b)))
            except ValueError:
                pass
            order_b = np.argsort(-s_b)
            y_sorted_b = y_b[order_b]
            for k in precision_at_k:
                if k <= len(y_sorted_b):
                    precat[k].append(float(y_sorted_b[:k].sum()) / k)
        if roc_samples:
            result.notes += (
                f" ROC95CI=[{np.quantile(roc_samples,0.025):.3f},{np.quantile(roc_samples,0.975):.3f}]; "
            )
        if pr_samples:
            result.notes += (
                f"PR95CI=[{np.quantile(pr_samples,0.025):.3f},{np.quantile(pr_samples,0.975):.3f}]; "
            )
        for k, samples in precat.items():
            if samples:
                lo, hi = np.quantile(samples, 0.025), np.quantile(samples, 0.975)
                result.notes += f"prec@{k}_95CI=[{lo:.3f},{hi:.3f}]; "

    return result


def compare_detectors(
    score_files: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    label_name: str = "restatement",
    label_col: str = "is_restatement",
    k_values: Sequence[int] = (10, 50, 100, 500, 1000),
    period_col: str = "fiscal_year",
    cik_col: str = "cik",
    score_col: str = "score",
) -> pd.DataFrame:
    """Run `evaluate_detector` for every (name → scores-DataFrame) and return a
    comparison table sorted by PR-AUC desc.

    DL detectors that emit non-default column names (e.g. `anomaly_score` from
    a VAE) can be compared by passing the relevant `score_col` here.
    """
    results: List[EvalResult] = []
    for name, scores in score_files.items():
        try:
            res = evaluate_detector(
                scores, labels,
                detector_name=name, label_name=label_name,
                label_col=label_col, period_col=period_col,
                cik_col=cik_col, score_col=score_col,
                k_values=k_values,
            )
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            log.warning("evaluate_detector failed for %s: %s", name, exc)

    rows: List[Dict] = []
    for r in results:
        row = {
            "detector": r.detector_name,
            "label": r.label_name,
            "n": r.n_observations,
            "n_pos": r.n_positive,
            "base_rate": r.base_rate,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
        }
        for k in k_values:
            row[f"precision@{k}"] = r.precision_at_k.get(k)
            row[f"lift@{k}"] = r.lift_at_k.get(k)
            row[f"n_pos@{k}"] = r.n_positive_at_k.get(k)
        rows.append(row)

    out = pd.DataFrame(rows)
    if "pr_auc" in out.columns:
        out = out.sort_values("pr_auc", ascending=False, na_position="last")
    return out.reset_index(drop=True)

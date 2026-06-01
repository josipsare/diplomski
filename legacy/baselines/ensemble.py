"""Ensemble anomaly detector that combines multiple base detectors.

Motivation (Phase 6 finding): individual detector ROC-AUCs converge to ~0.65
but top-100 set overlap is only 11-37%. Different model classes find DIFFERENT
specific firms — even though their ranking quality is similar. This suggests
ensemble methods can recover precision the individual detectors miss.

Three ensemble strategies, each useful for a different question:

    rank_average    Average of per-detector rank-percentiles. The most
                    conservative ensemble — keeps a firm's score "high" only
                    if MULTIPLE detectors agree it's anomalous. Good for
                    low-false-positive forensic screening.

    score_zscore    z-score each detector's scores within (period, sector)
                    then average. Treats detectors as independent estimators
                    of a latent anomaly factor. Standard quantitative-finance
                    factor-combination approach.

    union_top_k     Take the union of each detector's top-K firms. Higher
                    recall at the cost of precision — useful when you want
                    "every firm flagged by at least one detector."

The unified `(cik, period_end, score, model_name)` schema makes ensemble
construction a join + aggregate operation; no model retraining required.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _to_long(score_dfs: Dict[str, pd.DataFrame],
             cik_col: str = "cik", period_col: str = "fiscal_year") -> pd.DataFrame:
    """Stack all detector score frames into one long DataFrame."""
    parts = []
    for name, df in score_dfs.items():
        sub = df[[cik_col, period_col, "score"]].copy()
        sub["model_name"] = name
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def ensemble_rank_average(
    score_dfs: Dict[str, pd.DataFrame],
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
) -> pd.DataFrame:
    """Average the per-detector rank-percentiles (higher = more anomalous).

    Returns: DataFrame with `(cik, period_col, score, model_name)` where
    `score` is the average rank-percentile across detectors. Firm-periods
    not scored by every detector get NaN (intersection semantics).
    """
    long = _to_long(score_dfs, cik_col, period_col)

    # Per-detector rank percentile (NaN-aware, higher score → higher rank)
    long["rank_pct"] = long.groupby("model_name")["score"].rank(pct=True, method="average")

    wide = long.pivot_table(
        index=[cik_col, period_col], columns="model_name",
        values="rank_pct", aggfunc="first",
    )
    score = wide.mean(axis=1, skipna=False)  # require all detectors → NaN otherwise
    out = score.reset_index().rename(columns={0: "score"})
    out["score"] = score.values
    out["model_name"] = "ensemble_rank_average"
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    return out


def ensemble_score_zscore(
    score_dfs: Dict[str, pd.DataFrame],
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
) -> pd.DataFrame:
    """Per-detector z-score the raw anomaly scores, then average."""
    long = _to_long(score_dfs, cik_col, period_col)
    grp = long.groupby("model_name")["score"]
    long["z"] = (long["score"] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
    wide = long.pivot_table(
        index=[cik_col, period_col], columns="model_name",
        values="z", aggfunc="first",
    )
    score = wide.mean(axis=1, skipna=False)
    out = score.reset_index().rename(columns={0: "score"})
    out["score"] = score.values
    out["model_name"] = "ensemble_score_zscore"
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    return out


def ensemble_top_k_union(
    score_dfs: Dict[str, pd.DataFrame],
    k: int = 100,
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
) -> pd.DataFrame:
    """Score = number of detectors that placed this firm in their top-K.

    Higher score = more detectors agree the firm is anomalous. This is the
    discrete-valued "voting" ensemble; with N detectors a max score of N
    means "every detector flagged this firm".

    Returns a DataFrame keyed by the FULL union of (cik, period) pairs across
    all detectors, with score = vote count (0 for firms in nobody's top-K).
    Returning the full universe matters for downstream eval: a sparse table
    would let `compare_detectors` compute a metric over an enriched-positive
    pool whose base rate has nothing to do with the original universe (R2
    flagged this as a denominator artifact in the prior eval — top-100 union
    PR-AUC 0.169 vs single-detector 0.052 was inflated by the smaller pool,
    not by genuinely better ranking).
    """
    counts: Dict[tuple, int] = {}
    for name, df in score_dfs.items():
        top = df.nlargest(k, "score")[[cik_col, period_col]]
        for _, row in top.iterrows():
            key = (int(row[cik_col]), int(row[period_col]))
            counts[key] = counts.get(key, 0) + 1

    # Build the full universe = union of (cik, period) keys across all input frames
    universe: set[tuple[int, int]] = set()
    for df in score_dfs.values():
        for _, row in df[[cik_col, period_col]].dropna().iterrows():
            universe.add((int(row[cik_col]), int(row[period_col])))

    rows = [
        {cik_col: c, period_col: p,
         "score": float(counts.get((c, p), 0)),
         "model_name": f"ensemble_top{k}_union"}
        for (c, p) in universe
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

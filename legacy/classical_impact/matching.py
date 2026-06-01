"""Propensity-score matching of treated firms to control firms.

Used by the impact-analysis phase to construct counterfactual comparisons:
"if firm X was flagged as anomalous in year Y, what did otherwise-similar
non-anomalous firms in the same period and sector experience?"

Two implementations:

    1. Logistic-regression propensity score
       Fit a logistic regression of treatment ∈ {0, 1} on covariates,
       predict P(treatment | covariates), then match each treated firm to its
       k nearest controls by predicted score (with optional caliper).

    2. Mahalanobis nearest-neighbor on covariates directly
       No score model; match each treated firm to its k nearest controls in
       the covariate space using Mahalanobis distance.

Default covariate set for this thesis:
    sector_2digit (exact match required)
    log_assets    (continuous, nearest-neighbor)
    fiscal_year   (exact match required — same reporting period)

These choices follow standard accounting-research practice (e.g. Dechow et al.
2011 use industry × size × period exact matching as the workhorse design).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """One matched pair (treated, control) with quality diagnostics."""
    treated_cik: int
    treated_period: int  # fiscal_year
    control_cik: int
    control_period: int
    propensity_treated: float
    propensity_control: float
    distance: float


def _safe_logit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a logistic regression; fall back to sklearn for stability."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for propensity matching; "
            "pip install scikit-learn"
        ) from e

    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def _mahalanobis_nn(
    X: np.ndarray, treated_mask: np.ndarray, k: int,
) -> tuple[List[List[int]], List[np.ndarray]]:
    """Mahalanobis nearest-neighbor for each treated row.

    Returns:
        neighbors:  List with one entry per treated row, each a list of up to
                    `k` global control row indices (positions in X), sorted by
                    ascending Mahalanobis distance.
        distances:  List with one entry per treated row, each a numpy array of
                    the same `k` distances (ascending). The same `pinv(cov)`
                    matrix is used for every treated row to avoid redundant
                    inversions and to keep the two outputs numerically aligned.
    """
    cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    inv = np.linalg.pinv(cov)
    t_idx = np.where(treated_mask)[0]
    c_idx = np.where(~treated_mask)[0]
    neighbors: List[List[int]] = []
    distances: List[np.ndarray] = []
    for i in t_idx:
        diffs = X[c_idx] - X[i]
        dist_sq = np.einsum("ij,jk,ik->i", diffs, inv, diffs)
        order = np.argsort(dist_sq)
        neighbors.append(c_idx[order[:k]].tolist())
        # Report sqrt for interpretability; kept aligned with neighbors order
        distances.append(np.sqrt(dist_sq[order[:k]]))
    return neighbors, distances


def propensity_match(
    panel: pd.DataFrame,
    treatment_col: str,
    covariates: Sequence[str],
    *,
    period_col: str = "fiscal_year",
    sector_col: Optional[str] = "sector_2digit",
    cik_col: str = "cik",
    k: int = 1,
    caliper: float | None = 0.05,
    with_replacement: bool = True,
    min_bucket_size: int = 6,
) -> pd.DataFrame:
    """Propensity-score-match treated firm-years to control firm-years.

    Args:
        panel:           Long DataFrame with one row per (cik, period).
        treatment_col:   Binary treatment indicator (1 = treated).
        covariates:      Covariate columns for the propensity model.
        period_col:      Match within (period, sector) buckets.
        sector_col:      Optional exact-match key (e.g. SIC-2 division).
        cik_col:         Firm identifier.
        k:               Number of nearest controls per treated firm.
        caliper:         Max |ps_treated − ps_control| for a valid match.
                          None disables the caliper.
        with_replacement: If True (default), the same control may match many
                          treated firms (Abadie & Imbens 2006). Note this
                          inflates standard errors of paired-sample tests
                          unless corrected via Abadie-Imbens variance.
                          If False, each control is used at most once globally.
        min_bucket_size:  Below this bucket count we fall back to Mahalanobis
                          nearest-neighbor on the standardized covariates
                          rather than fitting a propensity model on too few
                          observations.

    Returns:
        DataFrame of matched pairs with columns:
            treated_cik, treated_period, control_cik, control_period,
            propensity_treated, propensity_control, distance, match_method
        Plus an attached `.attrs["caliper_drop_rate"]` giving the fraction of
        treated firms that had no valid match within the caliper.
    """
    if treatment_col not in panel.columns:
        raise KeyError(f"treatment column {treatment_col!r} not in panel")
    missing_cov = [c for c in covariates if c not in panel.columns]
    if missing_cov:
        raise KeyError(f"missing covariates: {missing_cov}")

    df = panel.dropna(subset=[treatment_col] + list(covariates)).copy()
    df[treatment_col] = df[treatment_col].astype(int)

    bucket_keys = [period_col]
    if sector_col is not None and sector_col in df.columns:
        bucket_keys.append(sector_col)

    used_controls: set[int] = set()
    pairs: List[tuple[MatchResult, str]] = []
    n_treated_total = 0
    n_treated_unmatched = 0

    for bucket_vals, bucket_df in df.groupby(bucket_keys, sort=False, dropna=True):
        treated = bucket_df[bucket_df[treatment_col] == 1]
        controls = bucket_df[bucket_df[treatment_col] == 0]
        n_treated_total += len(treated)
        if treated.empty or controls.empty:
            n_treated_unmatched += len(treated)
            continue

        treated_idx = treated.index.to_numpy()
        control_idx = controls.index.to_numpy()

        match_method = "logit"
        if bucket_df[treatment_col].nunique() < 2 or len(bucket_df) < min_bucket_size:
            # Mahalanobis fallback on standardized covariates
            X_raw = bucket_df[list(covariates)].to_numpy(dtype=float)
            mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-9
            X = (X_raw - mu) / sd
            treated_mask = (bucket_df[treatment_col].to_numpy() == 1)
            nn_local, distances_per_treated = _mahalanobis_nn(X, treated_mask, k=k)
            bucket_idx = bucket_df.index.to_numpy()
            ps_t = np.zeros(len(treated_idx))  # no propensity meaning
            # Map X-row indices back to global DataFrame indices
            neighbor_lists = [[int(bucket_idx[j]) for j in lst] for lst in nn_local]
            match_method = "mahalanobis"
        else:
            X_raw = bucket_df[list(covariates)].to_numpy(dtype=float)
            mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-9
            X = (X_raw - mu) / sd
            y = bucket_df[treatment_col].to_numpy()
            try:
                ps = _safe_logit_fit(X, y)
            except Exception as exc:  # noqa: BLE001
                log.warning("Propensity fit failed for bucket %s: %s", bucket_vals, exc)
                continue

            bucket_idx = bucket_df.index.to_numpy()
            ps_series = pd.Series(ps, index=bucket_idx)
            ps_t = ps_series.loc[treated_idx].to_numpy()
            ps_c = ps_series.loc[control_idx].to_numpy()

            neighbor_lists = []
            distances_per_treated = []
            for i in range(len(treated_idx)):
                dists = np.abs(ps_c - ps_t[i])
                order = np.argsort(dists)
                neighbor_lists.append([control_idx[o] for o in order])
                distances_per_treated.append(dists[order])

        # Accept matches with caliper + with/without-replacement bookkeeping
        for i, t_idx in enumerate(treated_idx):
            n_taken = 0
            took_any = False
            for j, c_idx in enumerate(neighbor_lists[i]):
                d = float(distances_per_treated[i][j])
                if caliper is not None and d > caliper:
                    break
                if not with_replacement and c_idx in used_controls:
                    continue
                ps_c_val = float(0.0 if match_method == "mahalanobis" else ps_series.loc[c_idx])
                pairs.append((MatchResult(
                    treated_cik=int(df.loc[t_idx, cik_col]),
                    treated_period=int(df.loc[t_idx, period_col]),
                    control_cik=int(df.loc[c_idx, cik_col]),
                    control_period=int(df.loc[c_idx, period_col]),
                    propensity_treated=float(ps_t[i]),
                    propensity_control=ps_c_val,
                    distance=d,
                ), match_method))
                used_controls.add(int(c_idx))
                took_any = True
                n_taken += 1
                if n_taken >= k:
                    break
            if not took_any:
                n_treated_unmatched += 1

    if not pairs:
        out = pd.DataFrame(columns=[
            "treated_cik", "treated_period", "control_cik", "control_period",
            "propensity_treated", "propensity_control", "distance", "match_method",
        ])
    else:
        out = pd.DataFrame([
            {**p.__dict__, "match_method": m} for p, m in pairs
        ])

    drop_rate = n_treated_unmatched / n_treated_total if n_treated_total > 0 else 0.0
    out.attrs["caliper_drop_rate"] = drop_rate
    out.attrs["with_replacement"] = with_replacement
    out.attrs["n_treated"] = n_treated_total
    out.attrs["n_unmatched"] = n_treated_unmatched
    log.info(
        "propensity_match: %d treated, %d unmatched (drop rate %.1f%%); "
        "with_replacement=%s",
        n_treated_total, n_treated_unmatched, 100 * drop_rate, with_replacement,
    )
    return out


def matched_pair_outcome_diff(
    matches: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_col: str,
    *,
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
) -> pd.DataFrame:
    """For each matched pair, compute (treated_outcome − control_outcome).

    Returns a DataFrame ready for paired-t-test, sign-test, or descriptive
    summary in the impact-analysis chapter.
    """
    out = outcomes[[cik_col, period_col, outcome_col]].copy()

    treated = matches.merge(
        out.rename(columns={cik_col: "treated_cik", period_col: "treated_period",
                            outcome_col: f"{outcome_col}_treated"}),
        on=["treated_cik", "treated_period"], how="left",
    )
    paired = treated.merge(
        out.rename(columns={cik_col: "control_cik", period_col: "control_period",
                            outcome_col: f"{outcome_col}_control"}),
        on=["control_cik", "control_period"], how="left",
    )
    paired["diff"] = paired[f"{outcome_col}_treated"] - paired[f"{outcome_col}_control"]
    return paired

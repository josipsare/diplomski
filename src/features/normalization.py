"""Cross-sectional standardization of financial-ratio features.

Two reasons we standardize *cross-sectionally per period* rather than globally:

1. **Sector / regime drift.** Median ROA in 2014 differed from 2024.
   A global z-score would let one period's distribution dominate the others
   and swamp anomaly signal with regime shift.

2. **Anomaly is relative.** A firm with `accruals_to_assets = 0.15` is not
   intrinsically anomalous — it is anomalous *given what its peers reported
   that period*. Per-period z-scoring makes the model see exactly that.

The default standardizer also winsorizes at the 1st and 99th percentile per
period before z-scoring, so a single firm with insane ratios (e.g. negative
equity making leverage explode) does not poison the per-period scale.

For the sector-aware variant, pass a `sector` Series aligned with the panel
index; the standardizer then z-scores within (period, sector) buckets.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def winsorize_per_group(
    df: pd.DataFrame,
    group_key: str,
    columns: List[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Clip each column at the per-group lower/upper quantiles."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        lo = out.groupby(group_key)[col].transform(lambda s: s.quantile(lower_q))
        hi = out.groupby(group_key)[col].transform(lambda s: s.quantile(upper_q))
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def zscore_per_group(
    df: pd.DataFrame,
    group_key: str,
    columns: List[str],
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Standardize columns within each group: z = (x - μ) / (σ + eps)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        mu = out.groupby(group_key)[col].transform("mean")
        sd = out.groupby(group_key)[col].transform("std")
        out[col] = (out[col] - mu) / (sd + eps)
    return out


def standardize_cross_sectionally(
    df: pd.DataFrame,
    feature_columns: List[str],
    period_col: str = "fiscal_year",
    sector_col: Optional[str] = None,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Run the full cross-sectional standardization pipeline.

    Returns a DataFrame with the original columns replaced by their standardized
    values for the given `feature_columns`. Other columns pass through.

    Args:
        df:                input panel (rows = firm-period observations)
        feature_columns:   columns to standardize
        period_col:        name of the period column (e.g. fiscal_year, period_end)
        sector_col:        optional sector column; if provided we z-score within
                            (period, sector) buckets
        winsorize:         clip extremes before z-scoring (recommended)
        lower_q, upper_q:  winsorization quantiles
    """
    if sector_col is not None and sector_col in df.columns:
        df = df.copy()
        df["__group__"] = df[period_col].astype(str) + "::" + df[sector_col].astype(str)
        group_key = "__group__"
    else:
        group_key = period_col

    if winsorize:
        df = winsorize_per_group(df, group_key, feature_columns, lower_q, upper_q)
    df = zscore_per_group(df, group_key, feature_columns)

    if group_key == "__group__":
        df = df.drop(columns="__group__")
    return df


def fill_missing(df: pd.DataFrame, columns: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """Replace NaN with `fill_value` after standardization.

    After per-period z-scoring, the within-period mean is 0 and a missing-then-
    filled value is "as average as possible". We also output a companion mask
    DataFrame so models that benefit from explicit missingness flags can use it.
    """
    out = df.copy()
    out[columns] = out[columns].fillna(fill_value)
    return out


def make_missingness_mask(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Boolean mask: True where the original (pre-fill) value was NaN."""
    return df[columns].isna().astype(np.float32)

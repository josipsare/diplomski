"""Turn the wide quarterly panel into per-firm sequence tensors.

For sequence-based detectors (LSTM autoencoder, Transformer encoder) we need
per-firm tensors of shape `(T, F)` where:
    T = number of quarters in the time axis (uniform across firms via padding)
    F = number of features

Many firms are missing whole quarters — usually because they didn't file a 10-Q
or a tag was unreported. We carry through a mask of shape `(T, F)` so the model
can ignore those positions during reconstruction loss.

Outputs are ready for `torch.from_numpy(...).float()` once you load them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SequenceTensors:
    """A bundle holding the parallel arrays a sequence model needs."""

    ciks: np.ndarray            # shape (N,)         — firm identifier per row
    period_ends: np.ndarray     # shape (T,)         — datetime64 period grid
    features: np.ndarray        # shape (N, T, F)    — float32, NaN-filled with 0
    mask: np.ndarray            # shape (N, T, F)    — float32, 1 where observed
    feature_names: List[str]    # length F


def build_sequences(
    panel: pd.DataFrame,
    feature_columns: List[str],
    *,
    cik_col: str = "cik",
    period_col: str = "period_end",
    period_grid: Optional[pd.DatetimeIndex] = None,
    min_observations: int = 4,
    snap_to_quarter_end: bool = True,
) -> SequenceTensors:
    """Convert a long/wide panel into aligned per-firm sequence tensors.

    Args:
        panel:           input panel; must contain `cik_col`, `period_col`, and
                          all `feature_columns`
        feature_columns: list of feature column names to include
        cik_col:         column holding firm identifier
        period_col:      column holding the period date
        period_grid:     optional fixed DatetimeIndex of quarter ends; if None we
                          infer it from the data
        min_observations: skip firms with fewer than this many observed quarters
        snap_to_quarter_end: if True, round the period column down to the nearest
                          quarter end so firms with off-cycle fiscal years still
                          align onto the same axis (with the cost that two
                          near-quarter periods can collide; we drop the older one)
    """
    feature_columns = [c for c in feature_columns if c in panel.columns]
    n_features = len(feature_columns)

    df = panel.loc[:, [cik_col, period_col] + feature_columns].copy()
    df[period_col] = pd.to_datetime(df[period_col])
    if snap_to_quarter_end:
        df[period_col] = df[period_col].dt.to_period("Q").dt.end_time.dt.normalize()
    else:
        # Drop time-of-day so the period_to_idx lookup matches even if the
        # source had odd microsecond residue.
        df[period_col] = df[period_col].dt.normalize()

    # Drop rows with no observations at all
    df = df.dropna(subset=feature_columns, how="all")

    # If two rows for the same (firm, quarter) survive (e.g. a Q1 and a snap-to-Q1
    # interim), keep the most complete one.
    df["_n_obs"] = df[feature_columns].notna().sum(axis=1)
    df = (
        df.sort_values([cik_col, period_col, "_n_obs"], ascending=[True, True, False])
        .drop_duplicates([cik_col, period_col], keep="first")
        .drop(columns="_n_obs")
    )

    if period_grid is None:
        period_grid = pd.DatetimeIndex(sorted(df[period_col].dropna().unique()))
    else:
        period_grid = pd.DatetimeIndex(period_grid).normalize()
    n_periods = len(period_grid)

    period_to_idx = pd.Series(np.arange(n_periods), index=period_grid)

    # Drop rows whose period_end isn't on the grid (silent skip would hide bugs)
    df = df[df[period_col].isin(period_grid)]
    df["_t"] = df[period_col].map(period_to_idx).astype("Int64")

    # Filter firms that don't meet the minimum-observations threshold
    obs_per_cik = df.groupby(cik_col).size()
    keep_ciks = obs_per_cik[obs_per_cik >= min_observations].index
    df = df[df[cik_col].isin(keep_ciks)]

    if df.empty:
        return SequenceTensors(
            ciks=np.empty(0, dtype=np.int64),
            period_ends=np.array(period_grid, dtype="datetime64[ns]"),
            features=np.empty((0, n_periods, n_features), dtype=np.float32),
            mask=np.empty((0, n_periods, n_features), dtype=np.float32),
            feature_names=list(feature_columns),
        )

    ciks_sorted = np.sort(df[cik_col].unique())
    cik_to_idx = pd.Series(np.arange(len(ciks_sorted)), index=ciks_sorted)
    df["_n"] = df[cik_col].map(cik_to_idx).astype("Int64")

    features = np.zeros((len(ciks_sorted), n_periods, n_features), dtype=np.float32)
    mask = np.zeros((len(ciks_sorted), n_periods, n_features), dtype=np.float32)

    n_idx = df["_n"].to_numpy()
    t_idx = df["_t"].to_numpy()
    raw = df[feature_columns].to_numpy(dtype=np.float64)
    obs = ~np.isnan(raw)
    features[n_idx, t_idx, :] = np.where(obs, raw, 0.0).astype(np.float32)
    mask[n_idx, t_idx, :] = obs.astype(np.float32)

    return SequenceTensors(
        ciks=ciks_sorted.astype(np.int64),
        period_ends=np.array(period_grid, dtype="datetime64[ns]"),
        features=features,
        mask=mask,
        feature_names=list(feature_columns),
    )


def _normalize_npz_path(path) -> "Path":
    """Always return a path ending in `.npz`. Replaces any existing suffix
    so the JSON sidecar (`.feature_names.json`) sits next to the .npz cleanly.
    """
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".npz":
        return p
    return p.with_suffix(".npz")


def save_sequences(seq: SequenceTensors, path) -> None:
    """Persist sequence tensors as a single .npz file plus a sidecar JSON
    for the feature-name list (str arrays force allow_pickle on load)."""
    import json

    p = _normalize_npz_path(path)
    np.savez_compressed(
        p,
        ciks=seq.ciks,
        period_ends=seq.period_ends,
        features=seq.features,
        mask=seq.mask,
    )
    p.with_suffix(".feature_names.json").write_text(
        json.dumps(list(seq.feature_names))
    )


def load_sequences(path) -> SequenceTensors:
    import json

    p = _normalize_npz_path(path)
    npz = np.load(p, allow_pickle=False)
    feature_names = json.loads(p.with_suffix(".feature_names.json").read_text())
    return SequenceTensors(
        ciks=npz["ciks"],
        period_ends=npz["period_ends"],
        features=npz["features"],
        mask=npz["mask"],
        feature_names=feature_names,
    )

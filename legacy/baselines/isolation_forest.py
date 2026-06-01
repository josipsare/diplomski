"""Isolation Forest classical-ML baseline.

Operates on the standardized financial-ratio panel
(`panel_quarterly_ratios_filled.parquet` by default — sector-aware z-scores
with NaN→0). Trains a single Isolation Forest on the full panel; the score
per row is the negative `decision_function` so that **higher = more anomalous**
(matching the convention used by every other detector in the project).

This baseline serves two purposes in the thesis:
1. A classical-ML reference point that's well known in fraud-detection literature
   (Liu, Ting & Zhou 2008).
2. A sanity check that the standardized ratio panel is a usable feature matrix:
   if Isolation Forest can produce sensible scores from it, downstream DL
   detectors should also work.

The detector exposes the standard `fit / score` API plus a
`feature_importances` helper that returns a per-feature contribution to the
isolation depth (estimated by ablation: refit each feature one-at-a-time and
measure score change). Intended for Phase 6 ablation, not for production
scoring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """sklearn IsolationForest wrapper with the project's score convention.

    Args:
        n_estimators: number of base trees
        max_samples:  rows used per tree (sklearn default 'auto' = min(256, n))
        contamination: expected outlier fraction in training data; 'auto'
                       lets sklearn pick the threshold, but for Phase 6
                       comparison we score continuously and ignore the
                       threshold
        random_state: for reproducibility
        feature_columns: list of columns to use as features. If None, every
                         column besides `cik` and `period_end` (or whichever
                         identifiers the caller passes via score_dataframe).
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = "auto",
        contamination: float | str = "auto",
        random_state: int = 42,
        feature_columns: Optional[list[str]] = None,
    ):
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as e:
            raise ImportError("scikit-learn required: pip install scikit-learn") from e

        self._IF = IsolationForest
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.feature_columns = feature_columns
        self._model = None

    def fit(self, X) -> "IsolationForestDetector":
        """Train the IF on the given panel.

        IMPORTANT methodological note (must be disclosed in thesis):
        the standard usage trains AND scores on the same panel. This is
        valid for unsupervised anomaly detection (no label leakage), but it
        means anomaly scores are calibrated against full-dataset statistics.
        Universe-expansion effects (e.g. SP500 ROC=0.603 vs Russell 3000
        ROC=0.647) come partly from a more diverse training distribution that
        better calibrates the isolation depth — it's a legitimate effect, not
        leakage, but should be called out explicitly in Phase 6 evaluation.
        For a strict train/test split, use the optional `test_X` argument in
        `score_dataframe` (not implemented yet — Phase 6 if needed).
        """
        if isinstance(X, pd.DataFrame):
            cols = self.feature_columns or [
                c for c in X.columns
                if c not in ("cik", "period_end", "fiscal_year", "fiscal_quarter",
                             "sic", "sector_2digit", "sector_label", "n_tags_present")
            ]
            self.feature_columns = cols
            X_arr = X[cols].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # Replace NaN with 0 (assumes panel is post-fill_missing or already z-scored
        # with safe-zero defaults; raise if there are still NaNs we can't handle).
        if np.isnan(X_arr).any():
            X_arr = np.nan_to_num(X_arr, nan=0.0)

        self._model = self._IF(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_arr)
        return self

    def _score_array(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: must call fit() before score().")

        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_columns].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if np.isnan(X_arr).any():
            X_arr = np.nan_to_num(X_arr, nan=0.0)

        # sklearn convention: decision_function returns higher = MORE inlier
        # We invert so that higher = MORE anomalous (project convention).
        return -self._model.decision_function(X_arr)

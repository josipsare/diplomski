"""Uniform anomaly-detector API.

Every detector — classical baseline, statistical, or DL — exposes the same
two-method interface so downstream evaluation treats them identically:

    detector = SomeDetector(...)
    detector.fit(X_train)
    scores = detector.score(X_test)        # ndarray, higher = more anomalous

Plus a `name` property used in score-table column names and a
`score_dataframe(panel)` convenience that returns a long-format
`(cik, period_end, score, model_name)` DataFrame ready to write to
`data/output/scores/{model_name}.parquet`.

The base class is intentionally tiny — just the contract. Specific detectors
(Benford, IsolationForest, AE, VAE, LSTM-AE, Transformer) all subclass and
implement `fit` + `_score_array`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class AnomalyDetector(ABC):
    """Common contract for every anomaly-detector implementation."""

    name: str = "base"

    @abstractmethod
    def fit(self, X) -> "AnomalyDetector":
        """Train (or precompute) on a panel/feature matrix."""

    @abstractmethod
    def _score_array(self, X) -> np.ndarray:
        """Return per-row anomaly scores. Higher = more anomalous."""

    def score(self, X) -> np.ndarray:
        return self._score_array(X)

    def score_dataframe(
        self,
        panel: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        cik_col: str = "cik",
        period_col: str = "period_end",
        max_missing_frac: Optional[float] = 0.5,
    ) -> pd.DataFrame:
        """Score a panel and return a long-format `(cik, period, score, model)` DataFrame.

        Args:
            max_missing_frac: if set, rows whose original feature vector had
                MORE than this fraction of NaN values get their score replaced
                with NaN in the output (so downstream eval can drop them rather
                than ranking them as "anomalous" simply because they are
                data-sparse). Default 0.5 = drop rows with >50% missing.
                Pass None to keep all scores regardless of sparsity.
        """
        if feature_cols is None:
            X = panel.drop(columns=[c for c in (cik_col, period_col) if c in panel.columns])
        else:
            X = panel[feature_cols]
        scores = self._score_array(X)
        if len(scores) != len(panel):
            raise RuntimeError(
                f"{self.name}: score length {len(scores)} != panel length {len(panel)}"
            )
        out = panel[[cik_col, period_col]].copy()
        out["score"] = scores
        out["model_name"] = self.name

        # Compute originally-missing fraction so eval downstream can audit
        # data-sparsity-driven outliers.
        if feature_cols is None:
            feature_cols = [c for c in X.columns]
        missing_frac = panel[feature_cols].isna().mean(axis=1).to_numpy()
        out["missing_frac"] = missing_frac

        if max_missing_frac is not None:
            mask_too_sparse = missing_frac > max_missing_frac
            n_drop = int(mask_too_sparse.sum())
            if n_drop > 0:
                import logging as _log
                _log.getLogger(__name__).info(
                    "%s: marking %d/%d rows as NaN-score (missing_frac > %.2f)",
                    self.name, n_drop, len(out), max_missing_frac,
                )
            out.loc[mask_too_sparse, "score"] = float("nan")

        return out

    def save_scores(self, scores_df: pd.DataFrame, scores_dir: Path) -> Path:
        """Persist a scored DataFrame as `{scores_dir}/{name}.parquet`."""
        scores_dir = Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)
        path = scores_dir / f"{self.name}.parquet"
        scores_df.to_parquet(path, index=False)
        return path

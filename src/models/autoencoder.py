"""Fully-connected autoencoder anomaly detector.

Architecture: a symmetric MLP encoder/decoder with bottleneck. The model is
trained on the standardized financial-ratio panel; per-row reconstruction
error becomes the anomaly score (higher = more anomalous).

Default topology for a ~35-feature ratio panel:
    35 → 16 → 8 → 16 → 35
with ReLU activations, MSE loss, Adam optimizer, batch size 256.

Why this is a useful detector for the thesis:
- Captures NONLINEAR relationships between ratios (Benford and IF cannot)
- Captures interactions: a firm with high accruals AND low cf_to_assets AND
  high revenue growth scores anomalous because the COMBINATION is rare,
  not because any single ratio is extreme. This is the methodological
  contribution over Benford and Isolation Forest.
- Reconstruction error is well-defined per row → fits cleanly into the
  uniform `(cik, period_end, score)` schema.

Training is unsupervised — model never sees restatement labels. Phase 6
evaluation against `restatement_labels.parquet` is the supervised
benchmark to test "does AE beat Benford+IF baseline?".
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import AnomalyDetector

log = logging.getLogger(__name__)


class _AEModule:
    """Lazy-imported torch module wrapper. Defined inside a function so torch
    is only imported when the detector is actually instantiated — keeps the
    import cost off the critical path for non-DL evaluation runs."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        dropout: float = 0.0,
    ):
        import torch
        import torch.nn as nn
        self._torch = torch
        self._nn = nn

        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)] if dropout > 0 \
                       else [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, bottleneck_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        layers = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)] if dropout > 0 \
                       else [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_features)]
        self.decoder = nn.Sequential(*layers)

        # Wrap as one nn.Module so .parameters() / .train() work
        class _Net(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.net = _Net(self.encoder, self.decoder)


class AutoencoderDetector(AnomalyDetector):
    """Fully-connected AE anomaly detector with the project's score convention.

    Args:
        hidden_dims:    sizes of intermediate encoder layers (decoder mirrors)
        bottleneck_dim: latent-code dimensionality
        dropout:        optional dropout rate inside encoder/decoder
        epochs:         training epochs
        batch_size:     mini-batch size
        lr:             Adam learning rate
        device:         "cpu" or "cuda" (defaults to CPU on this machine)
        random_seed:    for reproducibility
        feature_columns: features to use (None → infer from non-meta columns)
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        bottleneck_dim: int = 8,
        dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
        random_seed: int = 42,
        feature_columns: Optional[List[str]] = None,
    ):
        self.hidden_dims = hidden_dims or [16]
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_seed = random_seed
        self.feature_columns = feature_columns
        self._module: Optional[_AEModule] = None
        self._train_loss_history: List[float] = []

    def _features(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = self.feature_columns or [
                c for c in X.columns
                if c not in ("cik", "period_end", "fiscal_year", "fiscal_quarter",
                             "sic", "sector_2digit", "sector_label", "n_tags_present")
            ]
            self.feature_columns = cols
            arr = X[cols].to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)

    def _missingness_fraction(self, X) -> np.ndarray:
        """Per-row fraction of features that were originally NaN.

        Used by `score_dataframe` to flag/exclude rows whose anomaly score is
        driven by data sparsity (zero-fill creates artificially high recon
        error) rather than by a real anomaly signal. R1 critical finding.
        """
        if isinstance(X, pd.DataFrame):
            cols = self.feature_columns
            return X[cols].isna().mean(axis=1).to_numpy(dtype=np.float32)
        return np.zeros(len(X), dtype=np.float32)

    def fit(self, X) -> "AutoencoderDetector":
        import torch
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X_arr = self._features(X)
        n, n_features = X_arr.shape

        self._module = _AEModule(
            n_features=n_features,
            hidden_dims=self.hidden_dims,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        )
        net = self._module.net.to(self.device)

        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.from_numpy(X_arr).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        net.train()
        self._train_loss_history = []
        for epoch in range(self.epochs):
            running = 0.0
            n_batches = 0
            for (batch,) in loader:
                opt.zero_grad()
                recon = net(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                running += loss.item()
                n_batches += 1
            avg = running / max(1, n_batches)
            self._train_loss_history.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info("AE epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, avg)

        net.eval()
        return self

    def _score_array(self, X) -> np.ndarray:
        if self._module is None:
            raise RuntimeError(f"{self.name}: must fit() before score().")
        import torch
        X_arr = self._features(X)
        X_t = torch.from_numpy(X_arr).to(self.device)
        with torch.no_grad():
            recon = self._module.net(X_t).cpu().numpy()
        # Per-row mean squared error
        return np.mean((recon - X_arr) ** 2, axis=1).astype(np.float64)

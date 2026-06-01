"""LSTM autoencoder for sequence-based anomaly detection.

Architecture: encoder LSTM compresses a per-firm sequence of T=44 quarterly
ratio vectors into a latent code; decoder LSTM unrolls the code back into a
reconstructed sequence. Per-position reconstruction error becomes the
quarter-level anomaly score; the per-firm-year score is the max across the
quarters in that fiscal year.

Why this matters for the thesis:
- Phase 4 found AE ≈ VAE ≈ IF (ROC ≈ 0.65) on flat ratio panels — three
  different architectures hit the same ceiling. R2's interpretation: ceiling
  is in the INPUT (cross-sectional snapshot), not the model.
- LSTM-AE tests whether **temporal dynamics** (a firm's quarter-to-quarter
  trajectory) provide signal that flat panels cannot capture. A firm whose
  ratios drift smoothly is "normal"; a firm whose ratios jump erratically
  is anomalous in a way the flat AE cannot see.
- If LSTM-AE also ties IF, the input ceiling is confirmed and the next
  unexplored direction is multimodal (text features from 10-K).

Per-firm sequences are produced by `src/features/sequence_builder.py` which
returns `SequenceTensors(ciks, period_ends, features (N, T, F), mask (N, T, F))`.
The mask flags which (firm, quarter, feature) cells were actually observed
versus zero-filled — we use it to ignore reconstruction loss on unobserved
cells (so missing-quarter padding doesn't dominate training).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import AnomalyDetector

log = logging.getLogger(__name__)


def _build_lstm_ae(n_features: int, hidden_dim: int, num_layers: int, latent_dim: int):
    """Lazy-build the torch LSTM-AE module."""
    import torch
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=n_features, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
            )
            self.to_latent = nn.Linear(hidden_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
            )
            self.output_proj = nn.Linear(hidden_dim, n_features)

        def forward(self, x):
            # x: (batch, T, F)
            _, (h_n, _) = self.encoder(x)
            # h_n: (num_layers, batch, hidden) — take final layer
            z = self.to_latent(h_n[-1])  # (batch, latent_dim)
            # Repeat z across T steps as decoder input
            T = x.shape[1]
            dec_in = self.from_latent(z).unsqueeze(1).repeat(1, T, 1)  # (batch, T, hidden)
            dec_out, _ = self.decoder(dec_in)
            return self.output_proj(dec_out)  # (batch, T, F)

    return LSTMAutoencoder()


class LSTMAutoencoderDetector(AnomalyDetector):
    """LSTM autoencoder anomaly detector with mask-aware reconstruction loss.

    Args:
        hidden_dim:   LSTM hidden state size
        num_layers:   stacked LSTM layers (encoder & decoder)
        latent_dim:   bottleneck dimension between encoder and decoder
        epochs:       training epochs
        batch_size:   mini-batch size (firms per batch)
        lr:           Adam learning rate
        device:       "cpu" or "cuda"
        random_seed:  reproducibility
    """

    name = "lstm_autoencoder"

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        latent_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_seed = random_seed
        self._net = None
        self._train_loss_history: List[float] = []
        self._n_features: Optional[int] = None
        self._period_ends: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, sequences) -> "LSTMAutoencoderDetector":
        """Train on a `SequenceTensors` bundle.

        Args:
            sequences: SequenceTensors with attributes:
                       features (N, T, F) float32
                       mask     (N, T, F) float32 (1 = observed, 0 = missing)
                       ciks     (N,) int
                       period_ends (T,) datetime64
                       feature_names list[str]
        """
        import torch

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X = np.asarray(sequences.features, dtype=np.float32)
        M = np.asarray(sequences.mask, dtype=np.float32)
        N, T, F = X.shape
        self._n_features = F
        self._period_ends = np.asarray(sequences.period_ends)
        self._feature_names = list(sequences.feature_names)
        log.info("LSTM-AE training data: N=%d firms × T=%d quarters × F=%d features", N, T, F)

        self._net = _build_lstm_ae(F, self.hidden_dim, self.num_layers, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X).to(self.device)
        M_t = torch.from_numpy(M).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, M_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        self._train_loss_history = []
        for epoch in range(self.epochs):
            running, n_batches = 0.0, 0
            for batch_x, batch_m in loader:
                opt.zero_grad()
                recon = self._net(batch_x)
                # Mask-aware MSE — only count error on observed positions
                err = ((recon - batch_x) ** 2) * batch_m
                denom = batch_m.sum().clamp_min(1.0)
                loss = err.sum() / denom
                loss.backward()
                opt.step()
                running += loss.item()
                n_batches += 1
            avg = running / max(1, n_batches)
            self._train_loss_history.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info("LSTM-AE epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, avg)

        self._net.eval()
        return self

    def _score_array(self, sequences) -> np.ndarray:
        """Per-firm-quarter reconstruction error as a flat (N*T,) array.

        For consumption via the standard `score_dataframe` API the caller
        should use `score_sequences_to_long(...)` instead, which returns
        a long-format `(cik, period_end, score)` DataFrame.
        """
        if self._net is None:
            raise RuntimeError(f"{self.name}: must fit() before score().")
        import torch
        X = np.asarray(sequences.features, dtype=np.float32)
        M = np.asarray(sequences.mask, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            recon = self._net(X_t).cpu().numpy()
        # Per-(firm, quarter) MSE over OBSERVED features
        sq_err = (recon - X) ** 2
        denom = np.maximum(M.sum(axis=2), 1.0)  # (N, T)
        per_qt_score = (sq_err * M).sum(axis=2) / denom  # (N, T)
        # Mask out fully-unobserved quarters (denom=1 fallback would otherwise
        # equal the observed err sum which is zero for unobserved → score 0,
        # which is misleading); replace with NaN.
        no_obs = M.sum(axis=2) == 0
        per_qt_score[no_obs] = np.nan
        return per_qt_score.flatten()

    def score_sequences_to_long(
        self,
        sequences,
    ) -> pd.DataFrame:
        """Score a SequenceTensors bundle and return a long-format DataFrame
        with one row per (cik, period_end, score) — directly consumable by
        `supervised_eval.compare_detectors`.
        """
        if self._net is None:
            raise RuntimeError(f"{self.name}: must fit() before score().")

        ciks = np.asarray(sequences.ciks)
        period_ends = np.asarray(sequences.period_ends)
        per_qt = self._score_array(sequences).reshape(len(ciks), len(period_ends))

        rows = []
        for i, cik in enumerate(ciks):
            for t, pe in enumerate(period_ends):
                rows.append({
                    "cik": int(cik),
                    "period_end": pd.Timestamp(pe),
                    "score": float(per_qt[i, t]) if not np.isnan(per_qt[i, t]) else float("nan"),
                    "model_name": self.name,
                })
        out = pd.DataFrame(rows)
        out = out.dropna(subset=["score"]).reset_index(drop=True)
        return out

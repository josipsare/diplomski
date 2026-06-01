"""Transformer-encoder anomaly detector for multivariate time series.

Architecture: PatchTST-flavored self-attention encoder. Per-firm sequence of
T quarterly ratio vectors (each dim F) gets a sinusoidal positional embedding,
then passes through `n_layers` of `nn.TransformerEncoderLayer`. The output is
projected back to F-dim per timestep; per-position MSE reconstruction loss is
the anomaly score.

Why this matters for the thesis:
- LSTM-AE captures temporal dynamics via recurrence; Transformer captures them
  via attention (each timestep can directly attend to any other).
- Transformers tend to win on long-range dependencies but need more data /
  more capacity. With T=44 quarters per firm and ~2,500 firms, this is a
  reasonable scale for a small-to-medium Transformer.
- If both LSTM-AE and Transformer tie classical IF (per Phase 4 convergence
  finding), the input ceiling is doubly confirmed and the thesis's strongest
  remaining lever is multimodal text features (FinBERT 10-K MD&A).

This implementation deliberately keeps the architecture small (1-2 encoder
layers, 4 attention heads) to be trainable on CPU in minutes. Phase 6
ablation can scale up if signal warrants.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import AnomalyDetector

log = logging.getLogger(__name__)


def _build_transformer(n_features: int, d_model: int, n_heads: int, n_layers: int, dim_ff: int):
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=dim_ff, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.output_proj = nn.Linear(d_model, n_features)

        def forward(self, x, src_key_padding_mask=None):
            # x: (batch, T, F)
            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
            return self.output_proj(h)

    return TransformerAE()


class TransformerEncoderDetector(AnomalyDetector):
    """Self-attention encoder anomaly detector.

    Args:
        d_model:    Transformer hidden dimension
        n_heads:    multi-head attention heads
        n_layers:   stacked encoder layers
        dim_ff:     feed-forward inner dimension
        epochs:     training epochs
        batch_size: mini-batch size
        lr:         Adam learning rate
        device:     "cpu" or "cuda"
        random_seed: reproducibility
    """

    name = "transformer_encoder"

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 64,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
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

    def fit(self, sequences) -> "TransformerEncoderDetector":
        import torch

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X = np.asarray(sequences.features, dtype=np.float32)
        M = np.asarray(sequences.mask, dtype=np.float32)
        N, T, F = X.shape
        self._n_features = F
        self._period_ends = np.asarray(sequences.period_ends)
        self._feature_names = list(sequences.feature_names)
        log.info("Transformer training data: N=%d × T=%d × F=%d", N, T, F)

        self._net = _build_transformer(
            n_features=F, d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, dim_ff=self.dim_ff,
        ).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X).to(self.device)
        M_t = torch.from_numpy(M).to(self.device)
        # Padding mask: True = ignore (where the entire row is unobserved)
        # Convention: src_key_padding_mask is (batch, T), True = ignore
        pad_mask_t = (M_t.sum(dim=2) == 0).to(self.device)

        ds = torch.utils.data.TensorDataset(X_t, M_t, pad_mask_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        self._train_loss_history = []
        for epoch in range(self.epochs):
            running, n_batches = 0.0, 0
            for batch_x, batch_m, batch_pad in loader:
                opt.zero_grad()
                recon = self._net(batch_x, src_key_padding_mask=batch_pad)
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
                log.info("Transformer epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, avg)

        self._net.eval()
        return self

    def _score_array(self, sequences) -> np.ndarray:
        if self._net is None:
            raise RuntimeError(f"{self.name}: must fit() before score().")
        import torch
        X = np.asarray(sequences.features, dtype=np.float32)
        M = np.asarray(sequences.mask, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)
        pad_mask_t = (torch.from_numpy(M).to(self.device).sum(dim=2) == 0)
        with torch.no_grad():
            recon = self._net(X_t, src_key_padding_mask=pad_mask_t).cpu().numpy()
        sq_err = (recon - X) ** 2
        denom = np.maximum(M.sum(axis=2), 1.0)
        per_qt = (sq_err * M).sum(axis=2) / denom
        no_obs = M.sum(axis=2) == 0
        per_qt[no_obs] = np.nan
        return per_qt.flatten()

    def score_sequences_to_long(self, sequences) -> pd.DataFrame:
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
        return pd.DataFrame(rows).dropna(subset=["score"]).reset_index(drop=True)

"""Variational autoencoder (VAE) anomaly detector with explicit uncertainty.

Architecture: encoder maps `x → (μ, log σ²)` of a Gaussian latent; decoder
maps a sample `z ~ N(μ, σ²)` back to reconstruction. Loss is the standard
ELBO: reconstruction MSE + β × KL[q(z|x) ‖ N(0, I)].

Anomaly score is the negative log-likelihood under the model:
    score(x) = recon_error(x) + β × KL(q(z|x) ‖ p(z))
Higher = more anomalous (the model both reconstructs poorly AND its inferred
latent is far from the prior mean).

Why this is useful beyond a vanilla AE for the thesis:
- Provides a principled probabilistic anomaly score, not just MSE
- The KL term captures "how far this firm's latent is from the typical firm,"
  which catches anomalies that the reconstruction error alone misses
  (e.g. a firm whose ratios are reconstructable but unusually positioned in
  the latent manifold)
- β-VAE (β > 1) trades reconstruction fidelity for latent disentanglement;
  exposed as a hyperparameter for Phase 6 ablation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import AnomalyDetector

log = logging.getLogger(__name__)


def _vae_module(n_features: int, hidden_dims: List[int], latent_dim: int):
    import torch
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            enc_layers = []
            prev = n_features
            for h in hidden_dims:
                enc_layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            self.encoder_trunk = nn.Sequential(*enc_layers)
            self.mu = nn.Linear(prev, latent_dim)
            self.logvar = nn.Linear(prev, latent_dim)

            dec_layers = []
            prev = latent_dim
            for h in reversed(hidden_dims):
                dec_layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            dec_layers += [nn.Linear(prev, n_features)]
            self.decoder = nn.Sequential(*dec_layers)

        def encode(self, x):
            h = self.encoder_trunk(x)
            return self.mu(h), self.logvar(h)

        def reparam(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

    return VAE()


class VAEDetector(AnomalyDetector):
    """β-VAE anomaly detector. Score = recon_error + β × KL.

    Args:
        hidden_dims:    list of intermediate encoder layer sizes (decoder mirrors)
        latent_dim:     latent code dimensionality
        beta:           KL weight. β = 1 is standard VAE; β > 1 is β-VAE
                        (more disentanglement, less reconstruction fidelity).
        epochs:         training epochs
        batch_size:     mini-batch size
        lr:             Adam learning rate
        device:         "cpu" or "cuda"
        random_seed:    for reproducibility
        feature_columns: features to use (None → infer from non-meta columns)
    """

    name = "vae"

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        beta: float = 1.0,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
        random_seed: int = 42,
        feature_columns: Optional[List[str]] = None,
    ):
        self.hidden_dims = hidden_dims or [16]
        self.latent_dim = latent_dim
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_seed = random_seed
        self.feature_columns = feature_columns
        self._net = None
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

    def fit(self, X) -> "VAEDetector":
        import torch
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X_arr = self._features(X)
        n, n_features = X_arr.shape

        self._net = _vae_module(n_features, self.hidden_dims, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        recon_fn = torch.nn.MSELoss(reduction="sum")

        X_t = torch.from_numpy(X_arr).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        self._train_loss_history = []
        for epoch in range(self.epochs):
            running, n_seen = 0.0, 0
            for (batch,) in loader:
                opt.zero_grad()
                recon, mu, logvar = self._net(batch)
                recon_loss = recon_fn(recon, batch)
                # KL divergence between N(mu, sigma^2) and N(0, I)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl
                loss.backward()
                opt.step()
                running += loss.item()
                n_seen += batch.shape[0]
            avg = running / max(1, n_seen)
            self._train_loss_history.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info("VAE epoch %d/%d  loss/sample=%.6f", epoch + 1, self.epochs, avg)

        self._net.eval()
        return self

    def _score_array(self, X) -> np.ndarray:
        if self._net is None:
            raise RuntimeError(f"{self.name}: must fit() before score().")
        import torch
        X_arr = self._features(X)
        X_t = torch.from_numpy(X_arr).to(self.device)
        with torch.no_grad():
            recon, mu, logvar = self._net(X_t)
            recon_err = ((recon - X_t) ** 2).mean(dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            score = recon_err + self.beta * kl / X_arr.shape[1]  # normalize KL per feature
        return score.cpu().numpy().astype(np.float64)

"""Train Phase 4 deep-learning anomaly detectors (AE, VAE) and persist scores.

Each detector:
- reads `panel_quarterly_ratios_filled.parquet` (sector-aware standardized + NaN-filled)
- trains end-to-end on the full panel (unsupervised)
- writes per-(cik, period_end) scores to `data/output/scores/{model_name}.parquet`

Usage:
    python scripts/train_dl.py                      # AE + VAE
    python scripts/train_dl.py --only ae
    python scripts/train_dl.py --only vae
    python scripts/train_dl.py --epochs 100         # longer training
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.financial_ratios import RATIO_COLUMNS
from src.models.autoencoder import AutoencoderDetector
from src.models.vae import VAEDetector
from src.utils.paths import PANEL_DIR, SCORES_DIR

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["ae", "vae", "all"], default="all")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0, help="VAE only: KL weight")
    return p.parse_args()


def _load_panel():
    log.info("Loading panel_quarterly_ratios_filled.parquet …")
    panel = pd.read_parquet(PANEL_DIR / "panel_quarterly_ratios_filled.parquet")
    log.info("Panel: %d rows × %d cols", *panel.shape)
    feature_cols = [c for c in RATIO_COLUMNS if c in panel.columns]
    log.info("Using %d feature columns", len(feature_cols))
    return panel, feature_cols


def run_ae(panel, feature_cols, args) -> None:
    log.info("Training AE: hidden=[16], bottleneck=%d, epochs=%d, batch=%d, lr=%g",
             args.latent_dim, args.epochs, args.batch_size, args.lr)
    det = AutoencoderDetector(
        hidden_dims=[16],
        bottleneck_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_columns=feature_cols,
    )
    det.fit(panel)
    scored = det.score_dataframe(panel, feature_cols=feature_cols)
    out_path = SCORES_DIR / f"{det.name}.parquet"
    scored.to_parquet(out_path, index=False)
    log.info("AE wrote %s (%d rows). Score: mean=%.4f std=%.4f min=%.4f max=%.4f",
             out_path, len(scored),
             scored.score.mean(), scored.score.std(),
             scored.score.min(), scored.score.max())


def run_vae(panel, feature_cols, args) -> None:
    log.info("Training VAE: hidden=[16], latent=%d, beta=%g, epochs=%d, batch=%d",
             args.latent_dim, args.beta, args.epochs, args.batch_size)
    det = VAEDetector(
        hidden_dims=[16],
        latent_dim=args.latent_dim,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_columns=feature_cols,
    )
    det.fit(panel)
    scored = det.score_dataframe(panel, feature_cols=feature_cols)
    out_path = SCORES_DIR / f"{det.name}.parquet"
    scored.to_parquet(out_path, index=False)
    log.info("VAE wrote %s (%d rows). Score: mean=%.4f std=%.4f min=%.4f max=%.4f",
             out_path, len(scored),
             scored.score.mean(), scored.score.std(),
             scored.score.min(), scored.score.max())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    panel, feature_cols = _load_panel()

    if args.only in ("ae", "all"):
        run_ae(panel, feature_cols, args)
    if args.only in ("vae", "all"):
        run_vae(panel, feature_cols, args)

    log.info("Done. Scores in %s", SCORES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build per-firm sequences from the panel + train Phase 5 sequence detectors.

LSTM autoencoder + Transformer encoder. Both consume the standardized + filled
quarterly ratio panel, build per-firm `(T quarters, F features)` tensors via
`src/features/sequence_builder.py`, and produce per-(cik, period_end) anomaly
scores written to `data/output/scores/{model_name}.parquet`.

Usage:
    python scripts/train_sequence_dl.py                      # both
    python scripts/train_sequence_dl.py --only lstm
    python scripts/train_sequence_dl.py --only transformer
    python scripts/train_sequence_dl.py --epochs 100         # longer
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
from src.features.sequence_builder import build_sequences
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.transformer_encoder import TransformerEncoderDetector
from src.utils.paths import PANEL_DIR, SCORES_DIR

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["lstm", "transformer", "all"], default="all")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-observations", type=int, default=4,
                   help="Minimum observed quarters per firm to include in training")
    return p.parse_args()


def _build_panel_sequences(min_obs: int):
    log.info("Loading panel_quarterly_ratios_filled.parquet …")
    panel = pd.read_parquet(PANEL_DIR / "panel_quarterly_ratios_filled.parquet")
    feature_cols = [c for c in RATIO_COLUMNS if c in panel.columns]
    log.info("Panel: %d rows × %d features", len(panel), len(feature_cols))

    log.info("Building per-firm sequences (min_observations=%d) …", min_obs)
    seq = build_sequences(panel, feature_columns=feature_cols, min_observations=min_obs)
    log.info("Sequences: %d firms × %d quarters × %d features (mask density %.4f)",
             len(seq.ciks), len(seq.period_ends), seq.features.shape[2],
             float(seq.mask.mean()))
    return seq, feature_cols


def run_lstm(seq, args) -> None:
    log.info("Training LSTM-AE: hidden=32, latent=8, layers=1, epochs=%d, batch=%d, lr=%g",
             args.epochs, args.batch_size, args.lr)
    det = LSTMAutoencoderDetector(
        hidden_dim=32, num_layers=1, latent_dim=8,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    det.fit(seq)
    scored = det.score_sequences_to_long(seq)
    out_path = SCORES_DIR / f"{det.name}.parquet"
    scored.to_parquet(out_path, index=False)
    log.info("LSTM-AE wrote %s (%d rows). Score: mean=%.4f std=%.4f min=%.4f max=%.4f",
             out_path, len(scored),
             scored.score.mean(), scored.score.std(),
             scored.score.min(), scored.score.max())


def run_transformer(seq, args) -> None:
    log.info("Training Transformer: d_model=32, heads=4, layers=2, dim_ff=64, epochs=%d, batch=%d",
             args.epochs, args.batch_size)
    det = TransformerEncoderDetector(
        d_model=32, n_heads=4, n_layers=2, dim_ff=64,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    det.fit(seq)
    scored = det.score_sequences_to_long(seq)
    out_path = SCORES_DIR / f"{det.name}.parquet"
    scored.to_parquet(out_path, index=False)
    log.info("Transformer wrote %s (%d rows). Score: mean=%.4f std=%.4f min=%.4f max=%.4f",
             out_path, len(scored),
             scored.score.mean(), scored.score.std(),
             scored.score.min(), scored.score.max())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    seq, _ = _build_panel_sequences(args.min_observations)

    if args.only in ("lstm", "all"):
        run_lstm(seq, args)
    if args.only in ("transformer", "all"):
        run_transformer(seq, args)

    log.info("Done. Scores in %s", SCORES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

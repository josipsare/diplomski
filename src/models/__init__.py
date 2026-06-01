"""Neural-network anomaly detectors (strict NN-only thesis).

Every detector exposes a uniform scoring API so downstream evaluation
treats them identically:

    fit(X_train) -> None
    score(X) -> ndarray of per-row anomaly scores  (higher = more anomalous)

Modules:
    base               — AnomalyDetector ABC
    autoencoder        — dense MLP autoencoder, mask-aware MSE reconstruction
    vae                — beta-VAE with KL regularizer; anomaly = -log p(x)
    lstm_autoencoder   — recurrent encoder/decoder on quarterly sequences
    transformer_encoder — PatchTST-flavored self-attention encoder

Classical baselines (Benford, Isolation Forest, ensembles) live in
`legacy/baselines/` for bachelor's-thesis reference; they are NOT part
of the primary master's-thesis pipeline.
"""

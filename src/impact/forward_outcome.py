"""Forward-outcome NN regressor: anomaly_score_t → outcome_{t+1}.

Strict NN-only implementation (replaces prior OLS version). Per the thesis
topic, *both* halves of the work must use deep neural networks:
    1. Anomaly detection (AE, VAE, LSTM-AE, Transformer)
    2. Impact quantification (this module — MLP regressor)

Architecture: small MLP (3 hidden layers, GELU activation) that takes the
anomaly score plus control features (log_assets, sector embedding) and
predicts a next-year outcome. Trained with MSE loss; evaluated by:
    - Out-of-sample R² (test set held out by year — temporal split, no leakage)
    - Permutation importance of the anomaly_score input (the headline number:
      "how much predictive value does the detector's score carry?")
    - Per-detector comparison: which NN detector's score yields the best
      out-of-sample forward outcome prediction?

This answers RQ2 with a strict NN methodology: anomaly score from
unsupervised NN → supervised NN regressor → out-of-sample predictive R².

Note: training the forward-outcome regressor is supervised (we have y),
but it operates ON TOP of strictly-unsupervised anomaly scores from the
NN detectors. The supervision is in *the outcome* (next-year return), not
in the anomaly labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ForwardOutcomeResult:
    detector_name: str
    outcome_column: str
    n_train: int
    n_test: int
    test_r2: float          # held-out R² of the full model
    test_r2_without_score: float  # held-out R² of a model trained WITHOUT the anomaly score (controls only)
    delta_r2: float         # test_r2 - test_r2_without_score → headline incremental value
    perm_importance: float  # permutation-importance of anomaly_score in test set
    test_mse: float
    train_loss_final: float
    epochs_trained: int


def _build_mlp(in_dim: int, hidden: int, n_layers: int, dropout: float):
    import torch
    import torch.nn as nn

    layers: List[nn.Module] = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden: int = 32,
    n_layers: int = 3,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
):
    """Train an MLP and return ``(test_r2, test_mse, final_train_loss, net)``.

    The trained ``net`` is returned so the caller can reuse the *same* model
    state for permutation-importance probing — avoiding the prior bug where
    a second independent training run produced a slightly different model
    (DataLoader shuffle order differs even with the same seed across calls).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_train.shape[1]
    net = _build_mlp(in_dim, hidden, n_layers, dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_te = torch.from_numpy(X_test.astype(np.float32)).to(device)
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    final_loss = float("nan")
    net.train()
    for _ in range(epochs):
        running, n_b = 0.0, 0
        for bx, by in loader:
            opt.zero_grad()
            pred = net(bx).squeeze(-1)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            running += loss.item()
            n_b += 1
        final_loss = running / max(1, n_b)

    net.eval()
    with torch.no_grad():
        y_pred = net(X_te).squeeze(-1).cpu().numpy()
    y_true = y_test
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mse = ss_res / max(1, len(y_true))
    return r2, mse, final_loss, net


def _permutation_importance(
    net,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_idx: int,
    n_repeats: int = 5,
    seed: int = 42,
) -> float:
    """Compute permutation importance of one column on a TRAINED net.

    Shuffles ``X_test[:, feature_idx]`` n_repeats times, asks the net to
    predict each time, averages the resulting R²-drop versus the un-shuffled
    baseline. Higher value = feature more important to the trained model.
    """
    import torch

    rng = np.random.default_rng(seed)
    net.eval()
    with torch.no_grad():
        y_pred_base = net(torch.from_numpy(X_test.astype(np.float32))).squeeze(-1).cpu().numpy()
    ss_res_base = float(np.sum((y_test - y_pred_base) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2)) + 1e-12
    r2_unshuffled = 1.0 - ss_res_base / ss_tot

    drops = []
    for _ in range(n_repeats):
        X_perm = X_test.copy().astype(np.float32)
        X_perm[:, feature_idx] = rng.permutation(X_perm[:, feature_idx])
        with torch.no_grad():
            y_pred_p = net(torch.from_numpy(X_perm)).squeeze(-1).cpu().numpy()
        ss_res_p = float(np.sum((y_test - y_pred_p) ** 2))
        r2_perm = 1.0 - ss_res_p / ss_tot
        drops.append(r2_unshuffled - r2_perm)
    return float(np.mean(drops))


def _add_growth_controls(panel: pd.DataFrame, cik_col: str, period_col: str) -> pd.DataFrame:
    """Compute revenue_growth and asset_growth from year-over-year changes.

    These are the growth controls R2 flagged as load-bearing: without them,
    a NN that picks up high-growth firms as "anomalous" would have its
    delta-R² on volatility partially attributable to growth itself
    (high-growth firms are mechanically more volatile). Partial-out growth
    to isolate the genuine anomaly signal.
    """
    out = panel.sort_values([cik_col, period_col]).copy()
    for src, dst in (("revenues", "revenue_growth"), ("assets", "asset_growth")):
        if src in out.columns and dst not in out.columns:
            prev = out.groupby(cik_col)[src].shift(1)
            # Pct-change. Denominator preserves sign of prev (no .abs()) so the
            # formula is correct in the general case — important if this helper
            # is later extended to income/cash-flow line items that go negative.
            out[dst] = (out[src] - prev) / prev.where(prev.abs() > 1e-6)
            # Winsorize at [-2, 2] to dampen extreme outliers (firms with tiny
            # revenue base can show absurd 50000% growth and dominate the
            # regression).
            out[dst] = out[dst].clip(-2.0, 2.0).astype(float)
    return out


def forward_outcome_regression(
    scores_annual: pd.DataFrame,
    panel_annual: pd.DataFrame,
    *,
    detector_name: str = "detector",
    outcome_col: str = "annual_return_market",
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
    score_col: str = "score",
    control_cols: Sequence[str] = ("log_assets", "revenue_growth", "asset_growth"),
    sector_col: Optional[str] = "sector_2digit",
    forward_lag: int = 1,
    test_year_threshold: int = 2022,
    epochs: int = 80,
    seed: int = 42,
) -> ForwardOutcomeResult:
    """NN-based forward-outcome regression.

    Temporal split: train on years < `test_year_threshold`, test on years
    ≥ `test_year_threshold` (default 2022). Out-of-sample R² is the headline
    metric — it cannot be inflated by overfitting because the test years
    were never seen during training.

    Headline number: `delta_r2` = test_r2 - test_r2_without_score, the
    incremental R² provided by the anomaly score on top of the controls.
    This isolates the detector's contribution from sector / size baselines.
    """
    s = scores_annual[[cik_col, period_col, score_col]].copy()
    s[period_col] = s[period_col].astype(int)
    s = s.rename(columns={score_col: "score_t"})

    # Make sure growth controls are available on the panel before slicing
    panel_with_growth = _add_growth_controls(panel_annual, cik_col, period_col)

    # Build extra_cols dynamically from the caller-supplied control_cols list
    # (plus an "assets" fallback when "log_assets" is missing and the panel
    # carries raw "assets" we can log-transform on the fly).
    extra_cols: List[str] = []
    for c in control_cols:
        if c in panel_with_growth.columns:
            extra_cols.append(c)
    if "log_assets" in control_cols and "log_assets" not in panel_with_growth.columns \
            and "assets" in panel_with_growth.columns:
        extra_cols.append("assets")  # log-transform happens just below
    if sector_col and sector_col in panel_with_growth.columns:
        extra_cols.append(sector_col)

    p = panel_with_growth[[cik_col, period_col, outcome_col] + extra_cols].copy()
    if "log_assets" not in p.columns and "assets" in p.columns:
        p["log_assets"] = np.log1p(p["assets"].clip(lower=0))
    p[period_col] = p[period_col].astype(int)

    p_fwd = p[[cik_col, period_col, outcome_col]].copy()
    p_fwd[period_col] = p_fwd[period_col] - forward_lag
    p_fwd = p_fwd.rename(columns={outcome_col: f"{outcome_col}_fwd"})

    df = s.merge(p, on=[cik_col, period_col], how="inner")
    df = df.merge(p_fwd, on=[cik_col, period_col], how="inner")
    df = df.dropna(subset=[f"{outcome_col}_fwd", "score_t"])

    if len(df) < 100:
        return ForwardOutcomeResult(detector_name, outcome_col, 0, 0, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, 0)

    # Build feature matrix: score_t first (index 0 for permutation importance),
    # then numeric controls, then sector dummies
    feature_cols = ["score_t"]
    for c in control_cols:
        if c in df.columns:
            feature_cols.append(c)
    if sector_col and sector_col in df.columns:
        sec_dummies = pd.get_dummies(df[sector_col], prefix="sec", drop_first=True, dtype=float)
        df = pd.concat([df, sec_dummies], axis=1)
        feature_cols.extend(sec_dummies.columns.tolist())

    # Drop rows with NaNs in the design matrix
    df = df.dropna(subset=feature_cols)
    if len(df) < 100:
        return ForwardOutcomeResult(detector_name, outcome_col, 0, 0, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, 0)

    # Temporal split
    train_mask = df[period_col] < test_year_threshold
    test_mask = ~train_mask
    if train_mask.sum() < 50 or test_mask.sum() < 50:
        return ForwardOutcomeResult(detector_name, outcome_col, int(train_mask.sum()),
                                    int(test_mask.sum()), np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, 0)

    X_full = df[feature_cols].to_numpy(dtype=np.float32)
    y_full = df[f"{outcome_col}_fwd"].to_numpy(dtype=np.float32)

    # Standardize features using training-set statistics (no leakage)
    mu = X_full[train_mask].mean(axis=0)
    sd = X_full[train_mask].std(axis=0) + 1e-9
    X_full = (X_full - mu) / sd
    # Standardize the outcome too (for stable optimization)
    y_mu = y_full[train_mask].mean()
    y_sd = y_full[train_mask].std() + 1e-9
    y_std = (y_full - y_mu) / y_sd

    X_tr, y_tr = X_full[train_mask], y_std[train_mask]
    X_te, y_te = X_full[test_mask], y_std[test_mask]

    # Full model: WITH anomaly score — keep the trained net for permutation
    r2_full, mse_full, loss_full, net_full = _fit_mlp(
        X_tr, y_tr, X_te, y_te, epochs=epochs, seed=seed,
    )

    # Baseline model: WITHOUT anomaly score (controls only)
    if X_tr.shape[1] > 1:
        r2_base, _, _, _ = _fit_mlp(
            X_tr[:, 1:], y_tr, X_te[:, 1:], y_te, epochs=epochs, seed=seed,
        )
    else:
        r2_base = 0.0
    delta_r2 = r2_full - r2_base

    # Permutation importance on the SAME net that produced r2_full.
    # Index 0 is the anomaly score (feature_cols[0] == "score_t").
    perm_importance = _permutation_importance(
        net_full, X_te, y_te, feature_idx=0, n_repeats=5, seed=seed,
    )

    return ForwardOutcomeResult(
        detector_name=detector_name,
        outcome_column=outcome_col,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        test_r2=r2_full,
        test_r2_without_score=r2_base,
        delta_r2=delta_r2,
        perm_importance=perm_importance,
        test_mse=mse_full,
        train_loss_final=loss_full,
        epochs_trained=epochs,
    )


def compare_detectors_forward_outcome(
    detector_score_files: Dict[str, pd.DataFrame],
    panel_annual: pd.DataFrame,
    *,
    outcomes: Sequence[str] = ("annual_return_market", "volatility_market",
                                "max_drawdown_market"),
    forward_lag: int = 1,
    test_year_threshold: int = 2022,
    epochs: int = 80,
) -> pd.DataFrame:
    """Run NN forward-outcome regressor for every (detector × outcome) and
    return a comparison table. Headline column is `delta_r2`: the test-set
    R² gained by adding the anomaly score on top of the controls."""
    rows: List[Dict] = []
    for name, scores in detector_score_files.items():
        for outcome in outcomes:
            if outcome not in panel_annual.columns:
                continue
            try:
                res = forward_outcome_regression(
                    scores, panel_annual, detector_name=name,
                    outcome_col=outcome, forward_lag=forward_lag,
                    test_year_threshold=test_year_threshold, epochs=epochs,
                )
                rows.append({
                    "detector": res.detector_name,
                    "outcome": res.outcome_column,
                    "n_train": res.n_train,
                    "n_test": res.n_test,
                    "test_r2": res.test_r2,
                    "test_r2_baseline": res.test_r2_without_score,
                    "delta_r2": res.delta_r2,
                    "perm_importance": res.perm_importance,
                })
            except Exception as exc:  # noqa: BLE001
                log.warning("forward_outcome failed for %s × %s: %s", name, outcome, exc)
    if not rows:
        return pd.DataFrame(columns=["detector", "outcome", "n_train", "n_test",
                                      "test_r2", "test_r2_baseline", "delta_r2",
                                      "perm_importance"])
    return pd.DataFrame(rows).sort_values(["outcome", "delta_r2"], ascending=[True, False]).reset_index(drop=True)

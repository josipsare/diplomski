# Master's Thesis Plan

**Topic:** Identifikacija financijskih anomalija primjenom dubokih neuronskih mreža te utjecaj identificiranih anomalija na poslovanje kompanije

**Author:** Josip Sare
**Started:** 2026-05-08

---

## Research questions (REVISED 2026-05-10 — strict NN-only scope)

**RQ1.** Mogu li **različite arhitekture dubokih neuronskih mreža** (statički AE, VAE, recurrent LSTM-AE, attention-based Transformer) pouzdano identificirati firme s anomalnim financijskim izvještavanjem u nenadziranom režimu (training bez labela)?

**RQ2.** Imaju li firme koje NN detektor označi kao anomalne sustavno različite tržišne ishode (prinos, volatilnost, max drawdown, volumen) u godinama nakon detekcije, mjereno **NN forward-outcome regresorom** (MLP koji predviđa outcome iz anomaly_score + controls)?

**RQ3.** Koja vrsta NN arhitekture (statički AE, VAE, sekvencijalni LSTM/Transformer) daje detekcijski signal s najjačim odnosom prema budućem poslovanju, mjereno out-of-sample delta-R² NN regresora?

**Note:** Classical baselines (Benford, Isolation Forest) i klasične impact metode (event study, propensity matching) su u `legacy/` kao reference, ali NISU u primary thesis narrative.

---

## High-level architecture

```
                                   ┌────────────────────────┐
                                   │ Raw SEC EDGAR (44q)    │
                                   │ Stock daily (438 sym)  │
                                   └───────────┬────────────┘
                                               │
                                ┌──────────────┴───────────────┐
                                │ src/data/                    │
                                │  - sec_loader.py             │
                                │  - stock_loader.py           │
                                │  - panel_builder.py          │
                                └──────────────┬───────────────┘
                                               ▼
                       ┌──────────────────────────────────────────────┐
                       │ panel_quarterly.parquet                       │
                       │   (cik, period_end, tag1..tagN, qtrs)         │
                       │ panel_annual.parquet                          │
                       │   (cik, year, agg features, stock features)   │
                       └──────────────────────┬───────────────────────┘
                                              │
                  ┌───────────────────────────┴───────────────────────────┐
                  │ src/features/                                          │
                  │  - financial_ratios.py    (compute ratios from tags)   │
                  │  - sequence_builder.py    (build per-firm tensors)     │
                  │  - normalization.py       (cross-sectional z-score)    │
                  └───────────────────────────┬───────────────────────────┘
                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ src/models/      (every model produces a uniform score: cik, t, score)    │
   │  - benford_baseline.py     (reproduce bachelor's MAD as baseline)         │
   │  - isolation_forest.py     (classical ML baseline)                        │
   │  - autoencoder.py          (vanilla AE on ratio vector)                   │
   │  - vae.py                  (probabilistic anomaly + uncertainty)          │
   │  - lstm_autoencoder.py     (sequence reconstruction)                      │
   │  - transformer.py          (PatchTST / TST-style sequence encoder)        │
   └──────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ src/impact/                                                                │
   │  - event_study.py          CAR around detected anomaly windows             │
   │  - matching.py             propensity-score matching anomalous vs control  │
   │  - forward_outcome.py      DL predictor: anomaly+controls -> N+1 outcome   │
   │  - granger.py              Granger causality and transfer entropy          │
   └──────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ src/evaluation/                                                            │
   │  - detector_compare.py     stability, sectoral profile, top-K overlap     │
   │  - case_studies.py         deep-dive on top-K firms per detector          │
   │  - figures.py              all thesis figures                             │
   └──────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                   thesis/main.tex
```

---

## Phase plan

### Phase 0 — Cleanup (1 sitting)
- [ ] Move bachelor-specific code to `legacy/` (parser stays in `src/`)
- [ ] Decide: PyTorch (preferred) or TensorFlow for DL stack
- [ ] Commit a clean baseline
- [ ] requirements.txt: add `torch`, `pyarrow`, `scikit-learn`, `statsmodels`

### Phase 1 — Data layer (2–3 sittings)
- [ ] `src/data/sec_loader.py` — iterate over `sec_data/*q*` quarters, load `num.txt` + `sub.txt`, filter to 442-firm universe
- [ ] Curate XBRL tag list: ~80 standard US-GAAP tags spanning balance sheet, income statement, cash flow
- [ ] Per (cik, period_end_date, tag): keep most recent reported value; build long-format DataFrame
- [ ] Pivot to wide: `panel_quarterly.parquet` indexed by (cik, period_end)
- [ ] `src/data/stock_loader.py` — load `data/stock_data/*.csv`, compute per-day returns
- [ ] `src/data/panel_builder.py` — combine into `panel_annual.parquet` with both financial-aggregate features and annual stock features
- [ ] Sanity-check: confirm panel matches existing per-company-year counts (~4,800 firm-years)

### Phase 2 — Feature engineering (1–2 sittings)
- [ ] `src/features/financial_ratios.py` — compute standard ratios (ROA, ROE, current ratio, debt/equity, asset turnover, accruals, etc.) from tag-level data
- [ ] `src/features/normalization.py` — cross-sectional standardization (per quarter, per sector) to prevent scale/sector leakage
- [ ] `src/features/sequence_builder.py` — produce per-firm tensors of shape `(n_quarters, n_features)` with masked padding for missing quarters
- [ ] Handle missingness: imputation strategy (median per sector? carry-forward? mask flag?)

### Phase 3 — Baselines (1 sitting)
- [ ] `src/models/benford_baseline.py` — port bachelor's MAD/chi-square per firm-year (simplified, single function)
- [ ] `src/models/isolation_forest.py` — Isolation Forest on per-firm-year ratio vector
- [ ] Unified score schema and dump to `data/output/scores/{model_name}.parquet`

### Phase 4 — Static DL detectors (DONE 2026-05-10)
- [x] `src/models/autoencoder.py` — fully-connected AE 35 → 16 → 8 → 16 → 35 (35 features = current RATIO_COLUMNS, was originally planned 80; reconciled after sticking with curated ratio set)
- [x] `src/models/vae.py` — β-VAE with KL regularizer, anomaly = recon_err + (β/n_features)·KL
- [x] Train per quarter (cross-sectional) — score is per-row reconstruction error
- [x] Score every (cik, period_end) pair → aggregated to (cik, fiscal_year) via max
- [x] Headline result: AE ≈ VAE ≈ IF, ROC-AUC 0.65 ± 0.02 (bootstrap 1000 resamples). DL does not surpass classical ML on flat ratio panels — information ceiling identified.
- [x] R2 confirmation experiment: annual-grain AE training also ties (granularity is not the bottleneck).
- [x] R1 fix: `nan_to_num` zero-fill now flagged via `missing_frac` column; rows with >50% missing get `score=NaN` to prevent data-sparsity-driven spurious anomaly scores.

### Phase 5 — Sequential DL detectors (DONE 2026-05-10)
- [x] `src/models/lstm_autoencoder.py` — encoder LSTM (32-dim) → 8-dim latent → decoder LSTM, mask-aware MSE
- [x] `src/models/transformer_encoder.py` — PatchTST-flavored 2-layer Transformer with positional encoding, mask-aware MSE
- [x] Score = per-(cik, period_end) reconstruction error, aggregated to firm-year via max
- [x] Headline result: LSTM-AE ROC=0.637, Transformer ROC=0.616. **Both tie or underperform IF (0.647)** — confirms input ceiling across 5 architectures.
- [x] **Ensemble extension** (`src/models/ensemble.py`): rank-average, z-score-average, top-K union strategies. Empirical: rank-avg PR-AUC=0.056 (+8% vs single), top-100 union PR-AUC=0.169 (3.3× lift). Diversity makes ensembles work despite individual ROC ties.

### Phase 6 — Detector comparison (1 sitting)
- [ ] `src/evaluation/detector_compare.py` — top-K agreement (Kendall tau, Jaccard), distribution of scores, sectoral profile
- [ ] Stability: train on 2014–2019, score 2020–2024; check that anomaly rankings are stable in time
- [ ] **Supervised benchmark on AAER labels** — join `aaer_labels.parquet (cik, fiscal_year, is_aaer)` with each detector's scores; compute precision@K (K ∈ {10, 50, 100, 500}), ROC-AUC, PR-AUC per detector. Headline table compares unsupervised detectors against the labeled subset.
- [ ] Ablation: period-only standardization vs sector-aware (per period × SIC-1); cash-flow accruals vs Beneish; cf_to_net_income |·| vs signed

### Phase 7 — Impact analysis (3–4 sittings)
- [ ] `src/impact/event_study.py` — CAR(-30, +60), CAR(-30, +250) around the worst anomaly quarter per firm
- [ ] `src/impact/matching.py` — for each high-anomaly firm-year, find matched non-anomalous control by sector + market cap; t-test on outcomes
- [ ] `src/impact/forward_outcome.py` — small DL model (MLP) predicting next-year return/volatility/volume from anomaly score + control features; report R^2 and feature importance vs. baseline (no-anomaly) model
- [ ] `src/impact/granger.py` — annual Granger test anomaly → return per firm; aggregate

### Phase 8 — Case studies (1–2 sittings)
- [ ] Replay bachelor's case studies (Medical Properties Trust, TANDEM, Oracle, Philip Morris) under each new detector
- [ ] Pick 5 fresh firms identified by DL but missed by Benford; manually investigate
- [ ] Pick 5 firms flagged by Benford but cleared by DL; manually investigate

### Phase 9 — Figures and tables (parallel, ongoing)
- [ ] Architecture diagrams (model schematics)
- [ ] Score distributions per detector
- [ ] Top-K overlap heatmap across detectors
- [ ] CAR plot around anomaly date
- [ ] Forward-outcome regression coefficient plot
- [ ] Updated risk-distribution pie + suspicious-companies tables

### Phase 10 — Writing (parallel, ongoing)
- [ ] Update `thesis/main.tex` with new chapters; reuse Benford intro chapter
- [ ] Fresh literature review on DL anomaly detection in financial reporting
- [ ] Methodology chapter mirroring `src/` modules
- [ ] Results, discussion, limitations, conclusion

---

## Tech stack

- Python 3.10+
- **PyTorch** for DL (proposing — finalize in Phase 0)
- pandas / pyarrow for the panel storage (Parquet)
- scikit-learn for Isolation Forest, propensity matching
- statsmodels for Granger / event-study
- matplotlib + seaborn for figures

---

## Open questions / risks

- **Missingness:** small/young firms have many gaps in quarterly tags. Need a principled handling.
- **Scale invariance:** raw values span many orders of magnitude. Needs ratios + log + cross-sectional standardization.
- **Sector confounding:** REITs, banks, and tech have very different reporting structures; either include sector as a feature, or train per-sector models.
- **Causal interpretation of impact:** unsupervised anomaly score is endogenous to firm fundamentals; matching design must be defensible, otherwise impact analysis is correlational at best (which is honest, but should be stated).

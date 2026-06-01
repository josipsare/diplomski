# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Master's thesis:** Identifikacija financijskih anomalija primjenom dubokih neuronskih mreža te utjecaj identificiranih anomalija na poslovanje kompanije.

**Author:** Josip Sare (started 2026-05-08)

Builds on a prior bachelor's thesis (Benford's Law analysis of 442 S&P 500 firms 2014–2024) that found ~97.5% statistically-deviating firms but only r=0.03 correlation between deviation and stock returns. The master's thesis replaces the linear statistical methods with deep neural networks and rigorously quantifies the impact of detected anomalies on company business outcomes.

- Full project plan and roadmap: [THESIS_PLAN.md](THESIS_PLAN.md)
- Prior bachelor's thesis code (kept for reference / baseline reproduction): `legacy/`
- Persistent project memory: `~/.claude/projects/c--Users-Korisnik-Desktop-diplomski/memory/`

**Python 3.10+ required.**

---

## CRITICAL RULE — Dual Review After Every Meaningful Progress

After completing any meaningful unit of progress, launch **two subagents IN PARALLEL** (single message, two `Agent` tool calls) to review the work BEFORE moving on.

### Reviewer 1 — Local check ("Does the new piece make sense?")
- **Subagent type:** `feature-dev:code-reviewer`
- **Question it answers:** Does this specific thing I just built make sense on its own?
- **Looks for:** correctness, logic errors, edge cases, missing handling, code quality, security, fit-for-purpose
- **Brief it on:** what was just built (file paths, what changed), what it should do, the immediate surrounding context

### Reviewer 2 — Holistic check ("Does it all still hang together?")
- **Subagent type:** `general-purpose`
- **Question it answers:** Does the new piece combined with everything we've built so far still make coherent sense as a whole?
- **Looks for:** architectural fit, conceptual coherence, research-question alignment, contradictions with prior decisions, scope drift, drift from `THESIS_PLAN.md` goals
- **Brief it on:** the thesis topic + research questions, prior design decisions (in `memory/thesis_design_decisions.md`), full project structure, what was just added, why

### After both reviewers respond
1. **Surface their findings verbatim to the user** — including any disagreement between them
2. **Do NOT silently fix issues.** Let the user decide what to address.
3. **Block on user response** for any non-trivial flag.

### When this rule applies
**Run reviews when:**
- Adding a new module/file with substantive logic (e.g. a model class, a new analysis method, a data pipeline component)
- A phase milestone in `THESIS_PLAN.md` becomes complete
- A new model implementation lands (any anomaly detector or impact-analysis component)
- A multi-file refactor
- A major design pivot (changing data schema, switching framework, abandoning an approach)
- A non-trivial bug fix that touches real logic

**Skip reviews when:**
- Trivial edits (typos, formatting, single-line tweaks)
- Todo list updates
- Memory writes
- Status messages
- Cleanup tasks (moving files, renaming directories)
- Documentation-only changes

If unsure, default to running them.

---

## Key paths

- `src/` — current master's thesis code
  - `data/` — SEC + stock loaders, panel builder
  - `features/` — financial ratios, normalization, sequence builder
  - `models/` — anomaly detectors (Phase 3+)
  - `impact/` — business-impact analysis (Phase 7)
  - `evaluation/` — detector comparison + figures (Phase 6)
  - `utils/paths.py` — canonical filesystem paths
- `legacy/` — bachelor's thesis code, kept as reference baseline
- `scripts/` — orchestration scripts (e.g. `build_panel.py`)
- `sec_data/` — raw SEC EDGAR Financial Statement Data Sets, 44 quarters (2014Q1–2024Q4)
- `data/stock_data/` — 438 ticker daily price CSVs from yfinance
- `data/input/companies.csv` — CIK → ticker mapping for the 442-firm universe
- `data/output/panels/` — derived Parquet panels (built by `scripts/build_panel.py`)
- `data/output/scores/` — per-detector anomaly scores (created by Phase 3+)
- `data/output/figures/` — thesis figures
- `data/output/trained_models/` — saved model checkpoints
- `thesis/main.tex` — LaTeX thesis source

## Common commands

```bash
# Install
pip install -r requirements.txt

# Build full panels (44 quarters × 442 firms, ~5 min)
python scripts/build_panel.py

# Quick sanity-check (1 quarter + 5 tickers, ~30 sec)
python scripts/build_panel.py --quick
```

## Tech stack

- Python 3.10+
- PyTorch for all DL models
- pandas + pyarrow (Parquet) for the panel storage
- scikit-learn for classical baselines (Isolation Forest, propensity matching, One-Class SVM, LOF)
- statsmodels for Granger causality, event-study regressions
- matplotlib + seaborn for figures

## Conventions

- Code identifiers, comments, and committed text are in **English** (matches existing codebase).
- Conversational replies to the user are in **Croatian** unless the user switches.
- All derived data lives in `data/output/`; never write to `sec_data/` or `data/stock_data/` (raw inputs).
- Each anomaly detector exposes a uniform API: `fit(X) -> None` and `score(X) -> ndarray` (higher = more anomalous). Scores are written to `data/output/scores/{model_name}.parquet` with schema `(cik, period_end, score)`.

## STRICT NN-ONLY POLICY (decided 2026-05-10)

The thesis is **strictly neural-network + unsupervised** throughout the primary narrative.

**Primary thesis methods (in `src/`):**
- 4 NN anomaly detectors (`src/models/`): `autoencoder.py`, `vae.py`, `lstm_autoencoder.py`, `transformer_encoder.py` — all trained unsupervised (reconstruction loss)
- 1 NN forward-outcome regressor (`src/impact/forward_outcome.py`): MLP that takes (anomaly_score, controls) and predicts next-year outcome with temporal train/test split and out-of-sample delta-R² as headline

**Moved to `legacy/baselines/`:** Benford detector, Isolation Forest, ensemble strategies. Kept as code for reference and bachelor's-thesis baseline reproduction, NOT in the master's primary tables.

**Moved to `legacy/classical_impact/`:** event_study.py, abnormal_returns.py, matching.py, event_dates.py. Classical-statistics impact methods, NOT in the master's primary tables.

**Evaluation policy (dual-track, unchanged):** Detector training stays strictly unsupervised. Evaluation has two tracks: unsupervised diagnostics (top-K agreement between the 4 NN detectors, sectoral profile, case studies, stability) and supervised benchmarking against restatement labels at evaluation time only (no labels in training loop).

**Honest framing of NN forward-outcome regressor:** the MLP regressor is *supervised in y* (next-year return), but operates ON TOP of strictly-unsupervised NN anomaly scores. The supervision is in outcome prediction, not anomaly labeling.

## Bachelor's-thesis Benford reference (for baseline reproduction)

The master's thesis includes a Benford-Law baseline detector that reproduces the prior bachelor's findings on the same data. Reference values from that work:

- 442 S&P 500 firms × 11 years (2014–2024) = 4,862 firm-year observations
- 97.5% of firms statistically deviate (p < 0.05); average chi-square 59.83 (critical: 15.507)
- Linear correlation MAD ↔ annual return: r = 0.03 (negligible)
- Lagged correlation MAD(N) → return(N+1): r = 0.004, p = 0.827

Thresholds (Nigrini, 2012):
- First digit (8 df): MAD < 1.5 good, > 2.5 concerning; chi-square critical 15.507
- Second digit (9 df): MAD < 1.2 good; chi-square critical 16.919

Test CIKs: Apple 0000320193, Microsoft 0000789019, Oracle 0001341439, Tesla 0001318605, Amazon 0001018724.

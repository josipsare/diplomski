# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEC Financial Data Downloader and Benford's Law Analysis Tool - a Python project for downloading public company financial statements from SEC EDGAR and analyzing them for conformance to Benford's Law (a statistical method for detecting potential financial irregularities).

**Python 3.8+ required.**

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install as editable package (enables CLI entry points)
pip install -e .

# Download SEC data
python scripts/download_data.py --user-agent "YourName email@example.com" --latest

# Run batch analysis
python scripts/run_analysis.py --companies data/input/companies.csv

# Generate visualizations
python scripts/generate_graphs.py

# Analyze single company
python examples/analyze_company.py --cik 0000320193

# Run tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_benford.py -v

# Run specific test class
python -m pytest tests/test_benford.py::TestCalculateBenfordMetrics -v
```

### CLI Entry Points (after `pip install -e .`)

```bash
sec-download   # → scripts/download_data.py
sec-analyze    # → scripts/run_analysis.py
sec-visualize  # → scripts/generate_graphs.py
```

## Architecture

### Data Pipeline

```text
SEC EDGAR → src/downloader.py → src/parser.py → src/benford.py → src/visualization.py
   │              │                   │                │                  │
   │              ▼                   ▼                ▼                  ▼
   │        ./sec_data/         DataFrames      Chi-square, MAD,    data/output/graphs/
   │        └─ {YEAR}q{Q}/                       KS-test, p-value
   │           ├─ sub.txt (submissions)
   │           ├─ num.txt (numerical facts)
   │           ├─ tag.txt (tag definitions)
   │           └─ txt.txt (text facts)
```

### Core Modules (`src/`)

- **downloader.py**: `SECDataDownloader` - Downloads quarterly ZIP files from SEC, handles rate limiting
- **parser.py**: `SECDataParser` - Parses tab-delimited SEC files into DataFrames
- **benford.py**: Statistical analysis functions:
  - `calculate_benford_metrics(numbers_series)` - First-digit analysis (chi-square, MAD, KS, p-value)
  - `calculate_second_digit_benford_metrics(numbers_series)` - Second-digit analysis
  - `calculate_digit_zscores(numbers_series)` - Per-digit Z-scores showing which digits deviate most
  - `calculate_anomaly_score(numbers_series)` - Composite 0-100 anomaly score with risk levels
  - `get_digit_distribution(numbers_series)` - Observed vs expected distribution for visualization
  - `interpret_results(metrics)` / `interpret_second_digit_results(metrics)` - Human-readable interpretation
- **batch_analysis.py**: `run_batch_analysis()` - Process multiple companies
- **visualization.py**: `generate_all_visualizations()` - Create all graphs

### Key Data Files

- `data/input/companies.csv`: Input company list (columns: CIK, company_name)
- `data/output/results/benford_analysis.csv`: Output metrics per company/year
- `config.yaml`: All configurable settings (paths, thresholds, year range)

## SEC API Requirements

- User-Agent header required: `"CompanyName contact@email.com"`
- Rate limit: max 10 requests/second (tool uses 0.15s delay)
- Data URL: `https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YEAR}q{QUARTER}.zip`

## Benford's Law Metrics Thresholds

### First-Digit Analysis (digits 1-9, 8 df)

| Metric | Good Conformance | Concerning |
|--------|------------------|------------|
| MAD | < 1.5 | > 2.5 |
| Chi-square | < 15.507 | > 15.507 |
| p-value | > 0.05 | < 0.05 |

### Second-Digit Analysis (digits 0-9, 9 df)

| Metric | Good Conformance | Concerning |
|--------|------------------|------------|
| MAD | < 1.2 | > 1.2 |
| Chi-square | < 16.919 | > 16.919 |
| p-value | > 0.05 | < 0.05 |

## Test CIKs

- Apple: 0000320193
- Microsoft: 0000789019
- Tesla: 0001318605
- Amazon: 0001018724
- Google: 0001652044

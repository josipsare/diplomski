"""Canonical filesystem paths for the project.

Importing from here means no module hard-codes filesystem layout.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_DATA_DIR = PROJECT_ROOT / "sec_data"
STOCK_DATA_DIR = PROJECT_ROOT / "data" / "stock_data"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

PANEL_DIR = OUTPUT_DIR / "panels"
SCORES_DIR = OUTPUT_DIR / "scores"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "trained_models"

for _d in (PANEL_DIR, SCORES_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

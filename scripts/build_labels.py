"""Build supervised-evaluation labels (restatements + AAER) from existing sources.

Restatement labels come from SEC 10-K/A and 10-Q/A amendments already on disk
in `sec_data/`. AAER labels come from Bao et al. (2020)'s GitHub-published
firm-year file (currently 1971-2015 only; 2016+ is "future work" pending a
proper SEC AAER scrape).

Outputs in `data/output/labels/`:
    restatement_labels.parquet     # primary supervised label
    aaer_labels_bao2020.parquet    # secondary supervised label (limited coverage)

Usage:
    python scripts/build_labels.py
    python scripts/build_labels.py --no-progress
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

from src.data.restatement_loader import build_restatement_labels, load_amendments
from src.utils.paths import INPUT_DIR, OUTPUT_DIR, SEC_DATA_DIR

log = logging.getLogger(__name__)
LABELS_DIR = OUTPUT_DIR / "labels"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def build_restatements(show_progress: bool) -> None:
    log.info("Loading SEC amendments (10-K/A, 10-Q/A, 20-F/A, 10-KT/A, 10-QT/A) …")
    amendments = load_amendments(SEC_DATA_DIR, show_progress=show_progress)
    log.info("Amendment filings collected: %d", len(amendments))

    labels = build_restatement_labels(amendments)
    log.info("Unique restated (cik, period_end) pairs: %d across %d firms, %d fiscal years",
             len(labels), labels["cik"].nunique() if not labels.empty else 0,
             labels["fiscal_year"].nunique() if not labels.empty else 0)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / "restatement_labels.parquet"
    labels.to_parquet(out_path, index=False)
    log.info("Wrote %s", out_path)

    if not labels.empty:
        log.info("Form-type breakdown:")
        # Each row's amendment_form_types is comma-joined; count individual occurrences
        all_forms = labels["amendment_form_types"].str.split(",").explode().str.strip()
        for form, n in all_forms.value_counts().items():
            log.info("  %s: %d", form, n)


def build_aaer_bao2020() -> None:
    raw = INPUT_DIR / "aaer_firm_year_raw.csv"
    if not raw.exists():
        log.warning("Bao 2020 AAER source not found at %s; skipping", raw)
        return

    df = pd.read_csv(raw, dtype={"P_AAER": "Int64", "CIK": "Int64", "YEARA": "Int64"})
    df = df.rename(columns={"CIK": "cik", "YEARA": "fiscal_year",
                            "P_AAER": "aaer_release_number",
                            "UNDERSTATEMENT": "is_understatement"})
    df["is_aaer"] = True
    df["source"] = "bao_2020"
    cols = ["cik", "fiscal_year", "is_aaer", "is_understatement",
            "aaer_release_number", "source"]
    out = df[cols].dropna(subset=["cik", "fiscal_year"]).copy()
    out["cik"] = out["cik"].astype(int)
    out["fiscal_year"] = out["fiscal_year"].astype(int)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / "aaer_labels_bao2020.parquet"
    out.to_parquet(out_path, index=False)
    log.info("AAER (Bao 2020) labels: %d firm-years across %d firms, years %d–%d. Wrote %s",
             len(out), out["cik"].nunique(),
             int(out["fiscal_year"].min()), int(out["fiscal_year"].max()), out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    show_progress = not args.no_progress

    build_restatements(show_progress)
    build_aaer_bao2020()

    log.info("Done. Labels in %s", LABELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

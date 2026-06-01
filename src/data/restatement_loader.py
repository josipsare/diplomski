"""Build per-(cik, fiscal_year) restatement labels from SEC 10-K/A and 10-Q/A.

Forensic-accounting standard (Dechow, Ge, Larson, Sloan 2011; Hennes, Leone,
Miller 2008): a 10-K/A or 10-Q/A filing is a restatement of a previously-filed
periodic report. The restated period_end becomes a quasi-supervised "this
firm restated this fiscal year" label that's much more comprehensive than
AAER (every AAER → restatement, but many restatements have no AAER).

This loader iterates every quarterly SEC archive, extracts amendment filings
(10-K/A, 10-Q/A, plus optional 20-F/A and 10-KT/A for foreign private issuers
and transition-period filers), groups by (cik, period_end), and writes a
single Parquet with one row per unique restated firm-period.

Output schema:
    cik                    int
    period_end             datetime
    fiscal_year            int             period_end.dt.year
    n_amendments           int             how many amendment filings hit this period
    amendment_form_types   str             comma-joined unique form types
    earliest_amendment     datetime        first amendment filing date
    latest_amendment       datetime        most recent amendment filing date
    is_restatement         bool            always True (the row exists)

Plus a derived `is_restatement` flag joined back onto `panel_annual.parquet`
keyed by (cik, fiscal_year). Firms not in the restatement table get
`is_restatement = False`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .sec_parser import SECDataParser

log = logging.getLogger(__name__)


# Forms recognized as restatement amendments. 10-K/A is the heavy hitter
# (annual restatement). 10-Q/A is the quarterly equivalent. 20-F/A is for
# foreign private issuers. 10-KT/A and 10-QT/A are transition-period (e.g.
# a firm changing its fiscal year-end) — included for completeness.
DEFAULT_AMENDMENT_FORMS = ("10-K/A", "10-Q/A", "20-F/A", "10-KT/A", "10-QT/A")


def _quarter_dirs(sec_data_dir: Path) -> List[Path]:
    return sorted([p for p in sec_data_dir.iterdir() if p.is_dir() and "q" in p.name])


def load_amendments(
    sec_data_dir: Path,
    universe_ciks: Optional[Iterable[int]] = None,
    forms: tuple[str, ...] = DEFAULT_AMENDMENT_FORMS,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Iterate quarterly archives and collect every amendment filing.

    Args:
        sec_data_dir: directory containing 2014q1, 2014q2, ... subdirs
        universe_ciks: optional set/iterable of CIKs to restrict to. If None,
                       returns ALL filers (useful when the universe is being
                       expanded later — the restatement table can be reused).
        forms: tuple of form strings to count as amendments
        show_progress: tqdm bar across quarters

    Returns:
        DataFrame with columns:
            cik, adsh, form, period_end (= sub.period), filed, name
    """
    universe = set(universe_ciks) if universe_ciks is not None else None
    dirs = _quarter_dirs(sec_data_dir)
    if not dirs:
        raise FileNotFoundError(f"No quarter directories under {sec_data_dir}")

    iterator = tqdm(dirs, desc="quarters", unit="q") if show_progress else dirs
    frames: List[pd.DataFrame] = []
    for d in iterator:
        try:
            parser = SECDataParser(d)
            sub = parser.load_submissions()
            sub = sub[sub["form"].isin(forms)].copy()
            if universe is not None:
                sub = sub[sub["cik"].isin(universe)]
            if sub.empty:
                continue
            keep = sub[["cik", "adsh", "form", "period", "filed", "name"]].rename(
                columns={"period": "period_end"}
            )
            frames.append(keep)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", d.name, exc)

    if not frames:
        return pd.DataFrame(columns=["cik", "adsh", "form", "period_end", "filed", "name"])

    out = pd.concat(frames, ignore_index=True)
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
    return out


def build_restatement_labels(
    amendments: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate amendment filings to one row per (cik, period_end).

    Multiple amendment filings may hit the same restated period (e.g. an
    amendment to an amendment). We collapse them into a single row with
    diagnostic columns counting how many amendments and listing their forms.
    """
    if amendments.empty:
        return pd.DataFrame(columns=[
            "cik", "period_end", "fiscal_year", "n_amendments",
            "amendment_form_types", "earliest_amendment", "latest_amendment",
            "is_restatement",
        ])

    df = amendments.dropna(subset=["cik", "period_end"]).copy()
    grouped = df.groupby(["cik", "period_end"])

    out = grouped.agg(
        n_amendments=("adsh", "count"),
        amendment_form_types=("form", lambda s: ",".join(sorted(s.unique()))),
        earliest_amendment=("filed", "min"),
        latest_amendment=("filed", "max"),
    ).reset_index()

    out["fiscal_year"] = out["period_end"].dt.year
    out["is_restatement"] = True
    cols = [
        "cik", "period_end", "fiscal_year", "n_amendments",
        "amendment_form_types", "earliest_amendment", "latest_amendment",
        "is_restatement",
    ]
    return out[cols].sort_values(["cik", "period_end"]).reset_index(drop=True)


def attach_restatement_flag(
    panel: pd.DataFrame,
    restatement_labels: pd.DataFrame,
    *,
    cik_col: str = "cik",
    period_col: str = "fiscal_year",
) -> pd.DataFrame:
    """Left-join `is_restatement` onto a panel keyed by (cik, period).

    Rows in the panel without a matching restatement get `is_restatement = False`.
    """
    if restatement_labels.empty:
        out = panel.copy()
        out["is_restatement"] = False
        return out

    keep = restatement_labels[[cik_col, period_col, "is_restatement", "n_amendments"]]
    # Aggregate restatement_labels to one row per (cik, period_col) in case
    # the panel uses fiscal_year while the labels are per period_end (annual+
    # quarterly amendments can collide on the same year).
    keep = keep.groupby([cik_col, period_col], as_index=False).agg(
        is_restatement=("is_restatement", "any"),
        n_amendments=("n_amendments", "sum"),
    )

    out = panel.merge(keep, on=[cik_col, period_col], how="left")
    out["is_restatement"] = out["is_restatement"].fillna(False).astype(bool)
    out["n_amendments"] = out["n_amendments"].fillna(0).astype(int)
    return out

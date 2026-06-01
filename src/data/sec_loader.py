"""Load SEC EDGAR Financial Statement Data Sets into a long-format DataFrame.

Iterates over every quarterly archive in `sec_data/`, filters the filings to the
project's firm universe (442 CIKs), pulls the curated XBRL tag set, and produces
one row per (cik, period_end, alias, qtrs) — the canonical "long" format.

The loader does not yet pivot or pick canonical rows per (cik, period_end, alias);
that responsibility lives in `panel_builder`. This separation keeps the loader
purely about extraction and lets the builder experiment with different pivot
strategies (e.g. preferring qtrs=1 over qtrs=4 for flow tags) without re-reading
the slow underlying files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .sec_parser import SECDataParser
from .tags import alias_of, all_tags, canonical_tags_with_synonyms

log = logging.getLogger(__name__)


def _quarter_dirs(sec_data_dir: Path) -> List[Path]:
    """Return all quarterly subdirectories in sorted order (2014q1 first)."""
    return sorted([p for p in sec_data_dir.iterdir() if p.is_dir() and "q" in p.name])


def _load_one_quarter(
    quarter_dir: Path,
    universe_ciks: set[int],
    canonical_to_synonyms: dict[str, list[str]],
    forms: tuple[str, ...] = ("10-K", "10-Q"),
) -> pd.DataFrame:
    """Extract long-format records for one quarterly archive.

    Returns a DataFrame with columns:
        cik, period_end, alias, canonical_tag, qtrs, value, uom, adsh, filed
    """
    parser = SECDataParser(quarter_dir)
    sub = parser.load_submissions()
    num = parser.load_numbers()

    # Restrict to filings of interest
    sub = sub[sub["form"].isin(forms)].copy()
    sub = sub[sub["cik"].isin(universe_ciks)]
    if sub.empty:
        return pd.DataFrame()

    sub = sub[["adsh", "cik", "form", "period", "filed", "name", "sic"]].rename(
        columns={"period": "fiscal_period_end"}
    )

    # The full set of synonym tags we care about
    all_synonyms = {syn for syns in canonical_to_synonyms.values() for syn in syns}
    num_subset = num[num["adsh"].isin(sub["adsh"]) & num["tag"].isin(all_synonyms)].copy()

    if num_subset.empty:
        return pd.DataFrame()

    num_subset["qtrs"] = pd.to_numeric(num_subset["qtrs"], errors="coerce").astype("Int64")
    num_subset["value"] = pd.to_numeric(num_subset["value"], errors="coerce")
    num_subset = num_subset.dropna(subset=["value"])

    # Map every synonym to its canonical tag and alias
    syn_to_canonical: dict[str, str] = {}
    for canonical, syns in canonical_to_synonyms.items():
        for syn in syns:
            syn_to_canonical[syn] = canonical

    num_subset["canonical_tag"] = num_subset["tag"].map(syn_to_canonical)
    num_subset["alias"] = num_subset["canonical_tag"].map(alias_of)

    merged = num_subset.merge(sub, on="adsh", how="inner")

    out = merged[
        [
            "cik",
            "ddate",
            "alias",
            "canonical_tag",
            "tag",
            "qtrs",
            "value",
            "uom",
            "adsh",
            "form",
            "filed",
            "fiscal_period_end",
            "sic",
        ]
    ].rename(columns={"ddate": "period_end"})

    # Within a single filing, the same (alias, qtrs, period_end) often appears
    # several times — once for the consolidated top-level value and several more
    # for dimensional breakdowns (segments, geographies, products). The
    # consolidated total is always the largest in absolute value, so collapse
    # duplicates by taking the row with max |value|.
    out["_abs"] = out["value"].abs()
    out = (
        out.sort_values(["cik", "period_end", "alias", "qtrs", "adsh", "_abs"],
                        ascending=[True, True, True, True, True, False])
           .drop_duplicates(subset=["cik", "period_end", "alias", "qtrs", "adsh"], keep="first")
           .drop(columns=["_abs"])
    )

    return out


def load_universe_from_companies_csv(companies_csv: Path) -> set[int]:
    """Read the 442-firm universe and return a set of integer CIKs."""
    df = pd.read_csv(companies_csv, dtype=str)
    cik_col = "cik" if "cik" in df.columns else df.columns[0]
    return {int(c.lstrip("0") or "0") for c in df[cik_col].dropna()}


def cik_to_sector_mapping(long_df: pd.DataFrame) -> pd.DataFrame:
    """Derive a stable per-CIK sector code from the long-format records.

    A firm's SIC code can theoretically change over time (re-classification),
    but in practice it's stable for the universe we work with. We pick the
    most-recent reported SIC per CIK as the canonical sector. Returns a
    DataFrame with columns: cik, sic, sector_2digit, sector_label.
    """
    if long_df.empty or "sic" not in long_df.columns:
        return pd.DataFrame(columns=["cik", "sic", "sector_2digit", "sector_label"])

    sic_per_cik = (
        long_df[["cik", "sic", "filed"]]
        .dropna(subset=["sic"])
        .sort_values(["cik", "filed"], ascending=[True, False])
        .drop_duplicates("cik", keep="first")
        .drop(columns="filed")
    )
    sic_per_cik["sic"] = sic_per_cik["sic"].astype(str).str.zfill(4)
    sic_per_cik["sector_2digit"] = sic_per_cik["sic"].str[:2]
    sic_per_cik["sector_label"] = sic_per_cik["sector_2digit"].map(_SIC_2DIGIT_LABELS)
    return sic_per_cik.reset_index(drop=True)


# Standard SIC division headings collapsed to 2-digit prefixes. Coarser than
# GICS, but adequate for cross-sectional / sector-aware standardization.
_SIC_2DIGIT_LABELS: dict[str, str] = {
    "01": "Agriculture", "02": "Agriculture", "07": "Agriculture", "08": "Agriculture", "09": "Agriculture",
    "10": "Mining", "12": "Mining", "13": "Mining", "14": "Mining",
    "15": "Construction", "16": "Construction", "17": "Construction",
    "20": "Food & Beverage", "21": "Tobacco", "22": "Textiles", "23": "Apparel",
    "24": "Lumber & Wood", "25": "Furniture", "26": "Paper", "27": "Printing & Publishing",
    "28": "Chemicals & Pharma", "29": "Petroleum Refining",
    "30": "Rubber & Plastics", "31": "Leather",
    "32": "Stone, Clay & Glass", "33": "Primary Metals", "34": "Fabricated Metal",
    "35": "Industrial Machinery", "36": "Electronics", "37": "Transportation Equipment",
    "38": "Instruments", "39": "Misc. Manufacturing",
    "40": "Railroad", "41": "Local Transit", "42": "Trucking", "44": "Water Transport",
    "45": "Air Transport", "46": "Pipelines", "47": "Transportation Services",
    "48": "Communications", "49": "Utilities",
    "50": "Wholesale (Durables)", "51": "Wholesale (Non-Durables)",
    "52": "Retail (Building)", "53": "Retail (General)", "54": "Retail (Food)",
    "55": "Retail (Auto)", "56": "Retail (Apparel)", "57": "Retail (Home Furnish)",
    "58": "Retail (Restaurants)", "59": "Retail (Misc)",
    "60": "Banks", "61": "Non-Bank Credit", "62": "Securities & Brokers",
    "63": "Insurance Carriers", "64": "Insurance Agents", "65": "Real Estate",
    "67": "Holding & Investment Offices",
    "70": "Hotels", "72": "Personal Services", "73": "Business Services",
    "75": "Auto Repair", "76": "Misc. Repair", "78": "Motion Pictures",
    "79": "Amusement & Recreation",
    "80": "Health Services", "81": "Legal Services", "82": "Educational Services",
    "83": "Social Services", "84": "Museums", "86": "Membership Organizations",
    "87": "Engineering & Accounting", "88": "Private Households", "89": "Other Services",
    "91": "Government", "92": "Government", "93": "Government", "94": "Government",
    "95": "Government", "96": "Government", "97": "Government", "99": "Other",
}


def load_long_panel(
    sec_data_dir: Path,
    universe_ciks: Iterable[int],
    quarters: Optional[List[str]] = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Iterate quarterly archives and concatenate long-format records.

    Args:
        sec_data_dir: directory containing 2014q1, 2014q2, ... subdirs
        universe_ciks: iterable of integer CIKs to include
        quarters: optional subset of quarter folder names (e.g. ["2024q4"]);
                  None = all available
        forms: filing forms to include
        show_progress: tqdm progress bar across quarters

    Returns:
        DataFrame in long format with one row per (cik, period_end, tag, qtrs, filing).
        Multiple filings may report the same (cik, period_end, alias, qtrs) — keep all
        and let the panel_builder reconcile (latest filing wins).
    """
    universe = set(universe_ciks)
    canonical_to_synonyms = canonical_tags_with_synonyms()

    dirs = _quarter_dirs(sec_data_dir)
    if quarters is not None:
        wanted = set(quarters)
        dirs = [d for d in dirs if d.name in wanted]

    if not dirs:
        raise FileNotFoundError(f"No matching quarter directories under {sec_data_dir}")

    iterator = tqdm(dirs, desc="quarters", unit="q") if show_progress else dirs
    frames: List[pd.DataFrame] = []
    for d in iterator:
        try:
            frames.append(
                _load_one_quarter(d, universe, canonical_to_synonyms, forms=forms)
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the whole run
            log.warning("Skipping %s: %s", d.name, exc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Normalize types so downstream parquet round-trips cleanly
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
    out["fiscal_period_end"] = pd.to_datetime(out["fiscal_period_end"], errors="coerce")

    return out

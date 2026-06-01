"""Pivot SEC long-format records into wide quarterly + annual panels.

A *long* row from `sec_loader.load_long_panel` is keyed by
    (cik, period_end, alias, qtrs, adsh, filed)
because multiple filings often re-report the same period (restatements).

This builder:

1. Picks the canonical row per (cik, period_end, alias) by

       a. preferring the `qtrs` value appropriate to the tag kind
          - balance items: qtrs == 0 (instant)
          - flow items:    qtrs == 1 if available; else qtrs == 4
       b. among rows tied on (a), keeping the most recently filed (`filed` desc)

2. Pivots the result to a wide DataFrame with one column per alias.

3. Stamps each row with the calendar fiscal_year and fiscal_quarter derived
   from `period_end`.

4. Aggregates the quarterly panel to an annual panel by taking, per (cik, year):
   - balance items at the year-end snapshot (period_end == fiscal year-end)
   - flow items as the sum across the four quarters when single-quarter values
     are present, else the qtrs == 4 annual value

The annual panel can then be joined with `stock_loader.annualize_all` outputs
via a (cik, year) <-> (ticker, year) bridge mapped through `data/input/companies.csv`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .tags import BALANCE_TAGS, FLOW_TAGS, PER_SHARE_TAGS, alias_of, all_tags

log = logging.getLogger(__name__)


_BALANCE_ALIASES = {alias for _tag, alias in BALANCE_TAGS}
_FLOW_ALIASES = {alias for _tag, alias in FLOW_TAGS} | {alias for _tag, alias in PER_SHARE_TAGS}


def _qtrs_score_quarterly(qtrs: float, is_balance: bool) -> int:
    """Lower is preferred. Quarterly panel wants single-quarter (qtrs=1) flows."""
    if pd.isna(qtrs):
        return 99
    q = int(qtrs)
    if is_balance:
        return 0 if q == 0 else 10 + abs(q)
    # flow tag
    if q == 1:
        return 0
    if q == 4:
        return 1
    return 5 + abs(q)


# NOTE: there is intentionally no `_qtrs_score_annual` helper. The annual panel
# is built by direct qtrs filtering inside `build_annual_panel` because the
# canonical-row preference logic (used by quarterly panels) is per (cik,
# period_end, alias) — but annual aggregation is per (cik, fiscal_year, alias),
# so the two operations are different in kind. Keep them visibly separate.


def _pick_canonical_rows(long_df: pd.DataFrame, *, score_fn) -> pd.DataFrame:
    """Reduce duplicate (cik, period_end, alias) entries to one canonical row.

    `score_fn(qtrs, is_balance) -> int` controls qtrs preference; lower wins.
    Tiebreak: most recent `filed` first, then largest |value| first.
    """
    if long_df.empty:
        return long_df

    df = long_df.copy()
    df["is_balance"] = df["alias"].isin(_BALANCE_ALIASES)
    df["_qtrs_score"] = [score_fn(q, b) for q, b in zip(df["qtrs"], df["is_balance"])]
    df["_abs"] = df["value"].abs()

    df = df.sort_values(
        by=["cik", "period_end", "alias", "_qtrs_score", "filed", "_abs"],
        ascending=[True, True, True, True, False, False],
    )

    canonical = df.drop_duplicates(subset=["cik", "period_end", "alias"], keep="first")
    return canonical.drop(columns=["_qtrs_score", "_abs", "is_balance"])


def build_quarterly_panel(
    long_df: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pivot the long-format SEC data to a wide quarterly panel.

    Returns a DataFrame indexed by (cik, period_end) with one column per alias.
    Adds derived columns: fiscal_year, fiscal_quarter, n_tags_present, and (if
    `sector_map` is provided) sic, sector_2digit, sector_label.
    """
    if long_df.empty:
        return pd.DataFrame()

    canonical = _pick_canonical_rows(long_df, score_fn=_qtrs_score_quarterly)

    wide = canonical.pivot_table(
        index=["cik", "period_end"],
        columns="alias",
        values="value",
        aggfunc="first",
    ).reset_index()

    for _tag, alias in all_tags():
        if alias not in wide.columns:
            wide[alias] = np.nan

    wide["period_end"] = pd.to_datetime(wide["period_end"])
    wide["fiscal_year"] = wide["period_end"].dt.year
    wide["fiscal_quarter"] = wide["period_end"].dt.quarter
    wide["n_tags_present"] = wide[[a for _t, a in all_tags()]].notna().sum(axis=1)

    feature_cols = [a for _t, a in all_tags()]
    meta_cols = ["cik", "period_end", "fiscal_year", "fiscal_quarter", "n_tags_present"]

    if sector_map is not None and not sector_map.empty:
        wide = wide.merge(
            sector_map[["cik", "sic", "sector_2digit", "sector_label"]],
            on="cik", how="left",
        )
        meta_cols = meta_cols + ["sic", "sector_2digit", "sector_label"]

    return wide[meta_cols + feature_cols].sort_values(["cik", "period_end"]).reset_index(drop=True)


def _dedupe_per_period(flow_long: pd.DataFrame) -> pd.DataFrame:
    """Restatement-safe dedup: per (cik, period_end, alias, qtrs) keep latest filed.

    Different filings may re-report the same (cik, period_end, alias, qtrs)
    with restated values. We keep the most recent filing's value. Within a
    filing-tied tie we already deduped on max-abs in `sec_loader`, so any ties
    here are real restatements; latest wins.
    """
    if flow_long.empty:
        return flow_long
    out = flow_long.copy()
    out["_abs"] = out["value"].abs()
    out = (
        out.sort_values(
            ["cik", "period_end", "alias", "qtrs", "filed", "_abs"],
            ascending=[True, True, True, True, False, False],
        )
        .drop_duplicates(subset=["cik", "period_end", "alias", "qtrs"], keep="first")
        .drop(columns="_abs")
    )
    return out


def build_annual_panel(
    long_df: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate the long-format SEC data to one row per (cik, fiscal_year).

    Built directly from `long_df` rather than from the quarterly panel because
    qtrs preference for annual data is different from the quarterly case:
        - Balance items (qtrs=0): pick the value at the latest period_end in the
                                  year (fiscal year-end snapshot).
        - Flow items:             prefer qtrs=4 (annual) at the latest period_end
                                  in the year. Fall back to a sum of qtrs=1
                                  (quarterly) if no annual is reported. Fall back
                                  to a sum of qtrs=2 (semi-annual; foreign 20-F
                                  filers) if neither is available.

    Diagnostic columns:
        n_tags_present            number of distinct feature columns with a value
        qtrs0_tags_present        number of distinct balance aliases (qtrs=0)
        qtrs4_tags_present        number of distinct flow aliases sourced from qtrs=4 annual
        qtrs1_fallback_tags       number of distinct flow aliases that fell back to a qtrs=1 sum
        qtrs2_fallback_tags       number of distinct flow aliases that fell back to a qtrs=2 sum
    """
    if long_df.empty:
        return pd.DataFrame()

    feature_cols = [a for _t, a in all_tags()]

    df = long_df.copy()
    df["fiscal_year"] = df["period_end"].dt.year
    df["is_balance"] = df["alias"].isin(_BALANCE_ALIASES)

    # ------- Balance items: qtrs=0 at latest period_end within year -------
    bal = df[df["is_balance"] & (df["qtrs"] == 0)].copy()
    bal["_abs"] = bal["value"].abs()
    bal = (
        bal.sort_values(
            ["cik", "fiscal_year", "alias", "period_end", "filed", "_abs"],
            ascending=[True, True, True, False, False, False],
        )
        .drop_duplicates(subset=["cik", "fiscal_year", "alias"], keep="first")
    )
    bal_wide = bal.pivot_table(
        index=["cik", "fiscal_year"], columns="alias", values="value", aggfunc="first"
    )

    # ------- Flow items: qtrs=4 > qtrs=1 sum > qtrs=2 sum -------
    flow_long = _dedupe_per_period(df[~df["is_balance"]])

    # qtrs=4: at most one row per (cik, year, alias) — pick latest period_end
    annual_flow = (
        flow_long[flow_long["qtrs"] == 4]
        .sort_values(
            ["cik", "fiscal_year", "alias", "period_end"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["cik", "fiscal_year", "alias"], keep="first")
    )
    annual_wide = annual_flow.pivot_table(
        index=["cik", "fiscal_year"], columns="alias", values="value", aggfunc="first"
    )

    # qtrs=1 sum: each quarter contributes once (already deduped per period_end)
    qtrs1_flow = flow_long[flow_long["qtrs"] == 1]
    qtrs1_wide = qtrs1_flow.pivot_table(
        index=["cik", "fiscal_year"], columns="alias", values="value", aggfunc="sum"
    )

    # qtrs=2 sum: each half contributes once
    qtrs2_flow = flow_long[flow_long["qtrs"] == 2]
    qtrs2_wide = qtrs2_flow.pivot_table(
        index=["cik", "fiscal_year"], columns="alias", values="value", aggfunc="sum"
    )

    # Combine: annual > qtrs=1 sum > qtrs=2 sum
    flow_wide = annual_wide.combine_first(qtrs1_wide).combine_first(qtrs2_wide)

    # ------- Combine balance + flow into final wide panel -------
    annual_panel = bal_wide.combine_first(flow_wide).reset_index()

    for col in feature_cols:
        if col not in annual_panel.columns:
            annual_panel[col] = np.nan

    annual_panel["n_tags_present"] = annual_panel[feature_cols].notna().sum(axis=1)

    # ------- Diagnostic counts of distinct aliases per source kind -------
    def _alias_set(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=object)
        return frame.groupby(["cik", "fiscal_year"])["alias"].agg(set)

    bal_aliases = _alias_set(bal)
    annual_aliases = _alias_set(annual_flow)
    qtrs1_aliases = _alias_set(qtrs1_flow)
    qtrs2_aliases = _alias_set(qtrs2_flow)

    panel_idx = pd.MultiIndex.from_frame(annual_panel[["cik", "fiscal_year"]])

    def _aligned(s: pd.Series) -> pd.Series:
        """Reindex onto `panel_idx`, filling missing keys with empty set."""
        return s.reindex(panel_idx).apply(lambda v: v if isinstance(v, set) else set())

    bal_a = _aligned(bal_aliases)
    annual_a = _aligned(annual_aliases)
    qtrs1_a = _aligned(qtrs1_aliases)
    qtrs2_a = _aligned(qtrs2_aliases)

    # Union of all "higher-priority" sources for fallback computation.
    # Note: align EVERY series to `panel_idx` first so set operations work
    # element-wise without dropping firms that lack one source entirely.
    annual_or_qtrs1 = pd.Series(
        [a | b for a, b in zip(annual_a, qtrs1_a)], index=panel_idx
    )

    annual_panel["qtrs0_tags_present"] = bal_a.apply(len).to_numpy(dtype=int)
    annual_panel["qtrs4_tags_present"] = annual_a.apply(len).to_numpy(dtype=int)
    annual_panel["qtrs1_fallback_tags"] = np.array(
        [len(a - b) for a, b in zip(qtrs1_a, annual_a)], dtype=int
    )
    annual_panel["qtrs2_fallback_tags"] = np.array(
        [len(a - b) for a, b in zip(qtrs2_a, annual_or_qtrs1)], dtype=int
    )

    meta_cols = [
        "cik", "fiscal_year",
        "n_tags_present", "qtrs0_tags_present",
        "qtrs4_tags_present", "qtrs1_fallback_tags", "qtrs2_fallback_tags",
    ]

    if sector_map is not None and not sector_map.empty:
        annual_panel = annual_panel.merge(
            sector_map[["cik", "sic", "sector_2digit", "sector_label"]],
            on="cik", how="left",
        )
        meta_cols = meta_cols + ["sic", "sector_2digit", "sector_label"]

    return annual_panel[meta_cols + feature_cols].sort_values(["cik", "fiscal_year"]).reset_index(drop=True)


def join_annual_with_stock(
    annual_panel: pd.DataFrame,
    stock_annual: pd.DataFrame,
    cik_to_ticker: pd.DataFrame,
) -> pd.DataFrame:
    """Attach annual stock features to the annual financial panel.

    Args:
        annual_panel:  output of `build_annual_panel`
        stock_annual:  output of `stock_loader.annualize_all`
        cik_to_ticker: DataFrame with columns ['cik', 'symbol'] (CIK as int)

    Returns a DataFrame with the financial columns plus the stock columns,
    suffixed `_stock` where they would otherwise collide.
    """
    if annual_panel.empty or stock_annual.empty:
        return annual_panel.copy()

    bridge = cik_to_ticker.copy()
    bridge["cik"] = pd.to_numeric(bridge["cik"], errors="coerce").astype("Int64")
    bridge = bridge.dropna(subset=["cik", "symbol"])

    panel = annual_panel.merge(bridge[["cik", "symbol"]], on="cik", how="left")

    stock = stock_annual.rename(columns={"ticker": "symbol", "year": "fiscal_year"}).copy()

    stock_feature_cols = [c for c in stock.columns if c not in ("symbol", "fiscal_year")]
    stock = stock.rename(columns={c: f"{c}_market" for c in stock_feature_cols})

    out = panel.merge(stock, on=["symbol", "fiscal_year"], how="left")
    return out


def save_panel(df: pd.DataFrame, path: Path) -> None:
    """Save panel to parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Wrote %d rows × %d cols to %s", len(df), df.shape[1], path)

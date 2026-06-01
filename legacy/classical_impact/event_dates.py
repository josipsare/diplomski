"""Map (cik, period_end) → SEC filing date so impact analysis anchors on the
date markets actually saw the disclosure.

Critical because event studies anchored on `period_end` measure pre-disclosure
random walk in the [-30, +60] window: the period closes weeks-to-months before
the 10-K/10-Q is filed. Markets react to the filing, not to the period close.

Two helpers:

    period_to_filed(scores_or_panel, long_sec)
        Joins on (cik, period_end). For each unique (cik, period_end) pair in
        `scores_or_panel`, finds the earliest filing in `long_sec` whose
        `fiscal_period_end` matches `period_end`. Returns the input DataFrame
        with an added `filed_date` column.

    filing_lag_days(panel)
        Diagnostic: distribution of (filed - period_end) days. Used to
        characterize how stale `period_end` would be as an event anchor.

Edge cases:
    - 10-K filings include comparative data from prior periods; we want the
      earliest *original* filing for each period_end, not later restatements.
      Heuristic: pick min(filed) per (cik, period_end).
    - Restatements after the original filing become a SEPARATE event date
      worth studying in their own right; that's a Phase 7 extension, not handled
      by this basic adapter.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def period_to_filed(
    scores_or_panel: pd.DataFrame,
    long_sec: pd.DataFrame,
    *,
    cik_col: str = "cik",
    period_col: str = "period_end",
) -> pd.DataFrame:
    """Attach `filed_date` (earliest SEC filing date per period) to scores.

    Args:
        scores_or_panel: DataFrame with at least `cik_col` and `period_col`.
        long_sec:        DataFrame with columns cik, period_end, filed
                         (output of `sec_loader.load_long_panel`).
        cik_col:         Name of CIK column in scores_or_panel.
        period_col:      Name of period_end column in scores_or_panel.

    Returns:
        Copy of `scores_or_panel` with a new `filed_date` column. Rows with
        no matching filing get `NaT`.
    """
    if scores_or_panel.empty:
        return scores_or_panel.copy()

    if "filed" not in long_sec.columns:
        raise KeyError("long_sec must include a 'filed' column")
    if "period_end" not in long_sec.columns or "cik" not in long_sec.columns:
        raise KeyError("long_sec must include 'cik' and 'period_end' columns")

    # Earliest filing per (cik, period_end) — that's the original disclosure
    earliest = (
        long_sec[["cik", "period_end", "filed"]]
        .dropna(subset=["filed", "period_end"])
        .groupby(["cik", "period_end"])["filed"]
        .min()
        .reset_index()
        .rename(columns={"filed": "filed_date"})
    )

    out = scores_or_panel.copy()
    out[period_col] = pd.to_datetime(out[period_col])
    earliest["period_end"] = pd.to_datetime(earliest["period_end"])

    # Align column names for the merge
    bridge = earliest.rename(columns={"cik": cik_col, "period_end": period_col})
    out = out.merge(bridge, on=[cik_col, period_col], how="left")
    return out


def filing_lag_days(joined: pd.DataFrame, period_col: str = "period_end") -> pd.Series:
    """Days between period_end and filed_date — diagnostic.

    Returns a Series of integer day counts (filed_date − period_end) with NaT
    rows dropped. Use to confirm the typical lag (10-K usually 60-90 days,
    10-Q usually 30-45 days).
    """
    if "filed_date" not in joined.columns:
        raise KeyError("Run period_to_filed first; joined must have 'filed_date'.")
    lag = (
        pd.to_datetime(joined["filed_date"]) - pd.to_datetime(joined[period_col])
    ).dt.days
    return lag.dropna()

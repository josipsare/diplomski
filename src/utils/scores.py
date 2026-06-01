"""Shared helpers for working with detector score files.

`annualize` aggregates per-(cik, period_end) scores to (cik, fiscal_year)
granularity using the max within the year. Max is the right aggregator
for anomaly scores because a firm's "anomalousness" for the year is the
worst quarter, not the average — averaging dilutes the signal and would
hide brief but severe deviations.
"""

from __future__ import annotations

import pandas as pd


def annualize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse a per-(cik, period_end) score frame to per-(cik, fiscal_year).

    Args:
        df:   long-format scores with columns (cik, period_end, score, ...)
        name: detector label written into the returned `model_name` column

    Returns:
        DataFrame with columns (cik, fiscal_year, score, model_name).
    """
    out = df.copy()
    out["fiscal_year"] = pd.to_datetime(out["period_end"]).dt.year
    annual = out.groupby(["cik", "fiscal_year"], as_index=False)["score"].max()
    annual["model_name"] = name
    return annual

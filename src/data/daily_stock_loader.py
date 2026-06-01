"""Load daily stock-price CSVs and retain the daily series for event studies.

Sibling to `stock_loader.py`. The annual loader collapses each year to a single
row of features and is sufficient for the firm-year correlation analyses the
prior bachelor's thesis ran. The impact-analysis phase needs *daily* returns so
that we can compute cumulative abnormal returns (CAR) over event windows like
[-30, +60] or [-30, +250] aligned on detected anomaly dates.

Outputs (Parquet, written by `scripts/build_impact_panel.py`):

    daily_returns.parquet
        ticker      str         e.g. "AAPL"
        date        date        trading day (tz-naive)
        close       float       closing price
        volume      float       daily trading volume
        log_return  float       log(close_t / close_{t-1}); first row of each
                                ticker is NaN (no prior close)

    market_returns.parquet
        date           date     trading day
        eq_weighted    float    equal-weighted log return across the universe
                                that traded that day (universe market proxy;
                                a defensible substitute for SPY when we restrict
                                to S&P 500 constituents)
        n_tickers      int      number of tickers contributing to the average

A future extension could swap in real SPY data, Fama-French 5-factor returns,
or a sector-specific market series; the API on `compute_car` etc. is designed
to take ANY market_df keyed by date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .stock_loader import load_daily_prices

log = logging.getLogger(__name__)


def load_all_daily(
    stock_data_dir: Path,
    tickers: Optional[Iterable[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Concatenate every per-ticker daily CSV into a long-format DataFrame.

    Returns a DataFrame with columns: ticker, date, close, volume, log_return.
    """
    files = sorted(stock_data_dir.glob("*.csv"))
    if tickers is not None:
        keep = set(tickers)
        files = [f for f in files if f.stem in keep]

    iterator = tqdm(files, desc="daily-tickers", unit="t") if show_progress else files

    frames: List[pd.DataFrame] = []
    for f in iterator:
        try:
            df = load_daily_prices(f)
            if df.empty:
                continue
            df = df[["close", "volume", "log_return"]].copy()
            df["ticker"] = f.stem
            df = df.reset_index().rename(columns={"date": "date"})
            df["date"] = df["date"].dt.normalize()
            frames.append(df[["ticker", "date", "close", "volume", "log_return"]])
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume", "log_return"])

    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def build_market_proxy(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted log-return market proxy from the universe.

    A simple but defensible market proxy when the universe is itself S&P 500
    constituents: the cross-sectional mean log return per day. Drops NaNs
    (typically the first row of each ticker).

    Returns DataFrame with columns: date, eq_weighted, n_tickers.
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "eq_weighted", "n_tickers"])

    valid = daily_df.dropna(subset=["log_return"])
    grouped = valid.groupby("date")["log_return"]
    out = pd.DataFrame({
        "eq_weighted": grouped.mean(),
        "n_tickers": grouped.size().astype(int),
    }).reset_index().sort_values("date").reset_index(drop=True)
    return out


def daily_window(
    daily_df: pd.DataFrame,
    market_df: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    window: tuple[int, int] = (-30, 60),
) -> pd.DataFrame:
    """Slice the daily series for one ticker around an event date.

    The window is in *trading days*, not calendar days: we count the
    intersection of `daily_df[ticker]` with `market_df.date` and select the
    ones that fall in [event_date_idx + window[0], event_date_idx + window[1]].

    Returns a DataFrame with columns:
        date, close, volume, log_return, market_return, day_relative_to_event

    `day_relative_to_event` is 0 on the event day and ±N for surrounding days.
    """
    if daily_df.empty or market_df.empty:
        return pd.DataFrame()

    sub = daily_df[daily_df["ticker"] == ticker].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("date").reset_index(drop=True)
    sub = sub.merge(
        market_df[["date", "eq_weighted"]].rename(columns={"eq_weighted": "market_return"}),
        on="date", how="left",
    )

    event_date = pd.Timestamp(event_date).normalize()
    # Find the last trading day on or before `event_date` (handles weekend/holiday)
    on_or_before = sub.index[sub["date"] <= event_date]
    if len(on_or_before) == 0:
        return pd.DataFrame()
    event_idx = int(on_or_before.max())

    lo = max(0, event_idx + window[0])
    hi = min(len(sub) - 1, event_idx + window[1])
    if hi < lo or hi < 0 or lo > len(sub) - 1:
        return pd.DataFrame()
    snippet = sub.iloc[lo : hi + 1].copy()
    snippet["day_relative_to_event"] = np.arange(lo - event_idx, hi - event_idx + 1)
    return snippet.reset_index(drop=True)

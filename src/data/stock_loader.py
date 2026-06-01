"""Load daily stock-price CSVs and compute per-firm annual market features.

Inputs: `data/stock_data/{TICKER}.csv` with the schema produced by yfinance:
    Date, Open, High, Low, Close, Volume, Dividends, Stock Splits

Outputs (per ticker, per calendar year — these are the columns this module writes):
    annual_return        — last/first close − 1, in percent
    log_return_mean      — mean daily log return × 252  (annualized drift)
    volatility           — daily log-return std × sqrt(252), in percent
    max_drawdown         — worst peak-to-trough drawdown over the year, in percent
    sharpe_ratio         — log_return_mean / volatility (rf assumed 0; reasonable
                            for cross-sectional comparison since rf cancels)
    avg_volume           — mean daily Volume
    median_volume        — median daily Volume (more robust to spikes)
    volume_growth        — log(avg_volume) − log(prev_year_avg_volume)
    trading_days         — number of rows in the year

NOTE: The annual aggregation here intentionally discards the daily series. The
event-study and abnormal-return calculations needed by the impact-analysis phase
live in `src/data/daily_stock_loader.py` (raw daily close + volume, kept at full
granularity) and `src/impact/abnormal_returns.py` (CAR computation around event
dates). Do not extend this module to do double duty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def load_daily_prices(stock_csv: Path) -> pd.DataFrame:
    """Read a single daily-prices CSV produced by yfinance.

    Returns a DataFrame indexed by tz-naive date with columns
        open, high, low, close, volume, dividends, stock_splits, log_return
    """
    df = pd.read_csv(stock_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close"])

    df["log_return"] = np.log(df["close"]).diff()
    return df


def _annual_features_for_one_year(year_df: pd.DataFrame) -> Dict[str, float]:
    if year_df.empty or year_df["close"].dropna().empty:
        return {
            "annual_return": np.nan,
            "log_return_mean": np.nan,
            "volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe_ratio": np.nan,
            "avg_volume": np.nan,
            "median_volume": np.nan,
            "trading_days": 0,
        }

    closes = year_df["close"].dropna()
    first, last = closes.iloc[0], closes.iloc[-1]
    annual_return = (last / first - 1.0) * 100.0 if first > 0 else np.nan

    log_returns = year_df["log_return"].dropna()
    if len(log_returns) > 1:
        ann_drift = log_returns.mean() * 252
        ann_vol = log_returns.std(ddof=1) * np.sqrt(252) * 100.0  # percent
        sharpe = ann_drift / (log_returns.std(ddof=1) * np.sqrt(252)) if log_returns.std(ddof=1) > 0 else np.nan
    else:
        ann_drift = np.nan
        ann_vol = np.nan
        sharpe = np.nan

    running_peak = closes.cummax()
    drawdown = (closes / running_peak - 1.0) * 100.0
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    volumes = year_df["volume"].dropna()
    return {
        "annual_return": annual_return,
        "log_return_mean": ann_drift,
        "volatility": ann_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "avg_volume": volumes.mean() if not volumes.empty else np.nan,
        "median_volume": volumes.median() if not volumes.empty else np.nan,
        "trading_days": int(len(closes)),
    }


def annualize_one_ticker(
    ticker: str,
    stock_csv: Path,
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Compute annual stock features for one ticker across all available years.

    Returns a DataFrame keyed by (ticker, year) with the feature columns.
    """
    df = load_daily_prices(stock_csv)
    if df.empty:
        return pd.DataFrame()

    years_present = sorted(df.index.year.unique().tolist())
    if years is not None:
        wanted = set(int(y) for y in years)
        years_present = [y for y in years_present if y in wanted]

    rows: List[Dict] = []
    for y in years_present:
        slice_df = df[df.index.year == y]
        feats = _annual_features_for_one_year(slice_df)
        feats["ticker"] = ticker
        feats["year"] = y
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Volume growth requires prev-year reference, computed after we have all years
    out = out.sort_values("year").reset_index(drop=True)
    out["volume_growth"] = np.log(out["avg_volume"]).diff()

    return out


def annualize_all(
    stock_data_dir: Path,
    tickers: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute annual stock features for every ticker in `stock_data_dir`.

    Args:
        stock_data_dir: directory containing TICKER.csv files
        tickers: optional subset of tickers to load (filename stems)
        years: optional subset of years
        show_progress: tqdm progress

    Returns:
        DataFrame with columns
            ticker, year, annual_return, log_return_mean, volatility,
            max_drawdown, sharpe_ratio, avg_volume, median_volume,
            volume_growth, trading_days
    """
    files = sorted(stock_data_dir.glob("*.csv"))
    if tickers is not None:
        keep = set(tickers)
        files = [f for f in files if f.stem in keep]

    iterator = tqdm(files, desc="tickers", unit="t") if show_progress else files

    frames: List[pd.DataFrame] = []
    for f in iterator:
        try:
            frames.append(annualize_one_ticker(f.stem, f, years=years))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out[
        [
            "ticker",
            "year",
            "annual_return",
            "log_return_mean",
            "volatility",
            "max_drawdown",
            "sharpe_ratio",
            "avg_volume",
            "median_volume",
            "volume_growth",
            "trading_days",
        ]
    ]
    return out

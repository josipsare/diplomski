"""Cumulative abnormal return (CAR) and buy-and-hold abnormal return (BHAR)
computation around event dates.

Three specifications, each useful for a different sanity check:

    raw          The firm's own log-return cumulated over the window.
                 Has no benchmark; useful as a sanity check.

    market       firm_return − market_return per day, summed.
                 (a.k.a. market-adjusted abnormal return). Cheap and standard.

    capm         firm_return − (alpha + beta × market_return) per day, summed,
                 where alpha and beta are estimated on a pre-event estimation
                 window (default: trading days [-250, -30] relative to event).
                 The "real" event-study spec; controls for systematic risk.

CAR is the *sum* of daily abnormal returns over the window; BHAR is the
*compounded* abnormal-return product. CAR is more standard in short-window
event studies; BHAR is preferred for long-window studies (180+ days).

The functions are vectorized over multiple firms via `compute_event_study`,
which takes a DataFrame of (cik, ticker, event_date) rows and returns one
CAR/BHAR per (cik, event_date).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Spec = Literal["raw", "market", "capm"]


@dataclass
class CarResult:
    cik: int
    ticker: str
    event_date: pd.Timestamp
    spec: Spec
    window: tuple[int, int]
    car: float
    bhar: float
    n_days: int
    alpha: float | None = None  # only for spec="capm"
    beta: float | None = None
    estimation_n_days: int = 0


def _slice_window(
    daily_df: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    window: tuple[int, int],
) -> pd.DataFrame:
    """Return the per-day rows in the event window for one ticker.

    Output columns: date, log_return, market_return, day_relative_to_event.
    """
    sub = daily_df[daily_df["ticker"] == ticker]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("date").reset_index(drop=True)

    event_date = pd.Timestamp(event_date).normalize()
    on_or_before = sub.index[sub["date"] <= event_date]
    if len(on_or_before) == 0:
        return pd.DataFrame()
    event_idx = int(on_or_before.max())

    lo = max(0, event_idx + window[0])
    hi = min(len(sub) - 1, event_idx + window[1])
    # Guard: if the requested window ends before the series starts (or starts
    # after it ends), return empty rather than letting pandas iloc with a
    # negative endpoint produce a "slice from end" — that returned a giant
    # snippet whose length didn't match the (empty) day-counter array and
    # blew up the assignment.
    if hi < lo or hi < 0 or lo > len(sub) - 1:
        return pd.DataFrame()
    snippet = sub.iloc[lo : hi + 1].copy()
    snippet["day_relative_to_event"] = np.arange(lo - event_idx, hi - event_idx + 1)
    return snippet.reset_index(drop=True)


def _estimate_capm(
    daily_df: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    estimation_window: tuple[int, int],
) -> tuple[float, float, int]:
    """OLS estimate of (alpha, beta) on the pre-event estimation window.

    Returns (alpha, beta, n_days_used). NaN, NaN, 0 if too little data.
    """
    est = _slice_window(daily_df, ticker, event_date, estimation_window)
    if est.empty or "log_return" not in est.columns or "market_return" not in est.columns:
        return float("nan"), float("nan"), 0
    est = est.dropna(subset=["log_return", "market_return"])
    if len(est) < 30:
        return float("nan"), float("nan"), len(est)

    x = est["market_return"].to_numpy()
    y = est["log_return"].to_numpy()
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).sum() / (n - 1)
    var_x = ((x - x_mean) ** 2).sum() / (n - 1)
    if var_x <= 0:
        return float("nan"), float("nan"), n

    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    return float(alpha), float(beta), n


def compute_car(
    daily_df: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    window: tuple[int, int] = (-30, 60),
    spec: Spec = "market",
    estimation_window: tuple[int, int] = (-250, -31),
    min_window_days: int = 5,
) -> CarResult | None:
    """Compute CAR + BHAR for one (ticker, event_date) under a given spec.

    Window arithmetic is in trading-day units, anchored on the most recent
    trading day on or before `event_date` (handles weekend/holiday).

    For spec="capm" the estimation window must NOT overlap the event window;
    otherwise alpha/beta are estimated on data that includes the event itself,
    biasing the abnormal return toward zero.
    """
    if spec == "capm":
        est_lo, est_hi = estimation_window
        ev_lo, ev_hi = window
        if est_hi >= ev_lo:
            raise ValueError(
                f"Estimation window ({est_lo}, {est_hi}) overlaps event window "
                f"({ev_lo}, {ev_hi}); the estimation window must end strictly "
                f"before the event window starts."
            )
    snippet = _slice_window(daily_df, ticker, event_date, window)
    if snippet.empty or "log_return" not in snippet.columns:
        return None
    if len(snippet) < min_window_days:
        return None

    snippet = snippet.dropna(subset=["log_return"])
    if len(snippet) < min_window_days:
        return None

    # Compute the firm log-return series and a benchmark log-return series so
    # we can produce both CAR (sum of log abnormal returns) and a true BHAR
    # (compounded difference of *gross* returns). For log inputs, BHAR ≠ CAR.
    firm_log_ret = snippet["log_return"].to_numpy()

    if spec == "raw":
        bench_log_ret = np.zeros_like(firm_log_ret)
        alpha = beta = None
        est_n = 0
    elif spec == "market":
        if "market_return" not in snippet.columns:
            log.warning("market_return missing for %s @ %s; skipping market spec", ticker, event_date)
            return None
        bench_log_ret = snippet["market_return"].fillna(0).to_numpy()
        alpha = beta = None
        est_n = 0
    elif spec == "capm":
        alpha, beta, est_n = _estimate_capm(daily_df, ticker, event_date, estimation_window)
        if not np.isfinite(alpha) or not np.isfinite(beta):
            return None
        if "market_return" not in snippet.columns:
            return None
        bench_log_ret = (alpha + beta * snippet["market_return"].fillna(0)).to_numpy()
    else:
        raise ValueError(f"Unknown spec: {spec!r}")

    # CAR: sum of daily log abnormal returns.
    # For log returns this equals log(end_firm_price / start_firm_price)
    # − log(end_benchmark / start_benchmark) = log(BHAR_firm / BHAR_benchmark).
    ar_log = firm_log_ret - bench_log_ret
    car = float(ar_log.sum())

    # True BHAR: compound the *gross* arithmetic returns, then take the
    # difference. This is Barber & Lyon (1997) / Mitchell & Stafford (2000).
    #     BHAR = ∏(1 + r_firm) − ∏(1 + r_benchmark)
    # where r_x = exp(log_return_x) − 1
    firm_gross = np.expm1(firm_log_ret)
    bench_gross = np.expm1(bench_log_ret)
    bhar = float(np.prod(1.0 + firm_gross) - np.prod(1.0 + bench_gross))

    ar = ar_log  # kept name for backwards compatibility downstream

    cik_match = daily_df.loc[daily_df["ticker"] == ticker, "cik"].iloc[0] if "cik" in daily_df.columns else -1

    return CarResult(
        cik=int(cik_match) if pd.notna(cik_match) else -1,
        ticker=ticker,
        event_date=pd.Timestamp(event_date).normalize(),
        spec=spec,
        window=window,
        car=car,
        bhar=bhar,
        n_days=int(len(ar)),
        alpha=alpha,
        beta=beta,
        estimation_n_days=est_n,
    )


def compute_event_study(
    events: pd.DataFrame,
    daily_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    window: tuple[int, int] = (-30, 60),
    spec: Spec = "market",
    estimation_window: tuple[int, int] = (-250, -31),
    cik_to_ticker: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run an event study over many (cik, event_date) pairs.

    Args:
        events:        DataFrame with at least `cik` and `event_date` columns
        daily_df:      output of `daily_stock_loader.load_all_daily`
        market_df:     output of `daily_stock_loader.build_market_proxy`
        cik_to_ticker: bridge with `cik` and `symbol` columns. If `daily_df`
                       already has a `cik` column the bridge is ignored.

    Returns a DataFrame with one row per event:
        cik, ticker, event_date, spec, window_lo, window_hi, car, bhar,
        n_days, alpha, beta, estimation_n_days
    """
    if events.empty:
        return pd.DataFrame()

    daily = daily_df.copy()

    # Attach market_return once so it's available to every per-event slice
    daily = daily.merge(
        market_df[["date", "eq_weighted"]].rename(columns={"eq_weighted": "market_return"}),
        on="date", how="left",
    )

    # Attach cik via bridge if needed
    if "cik" not in daily.columns:
        if cik_to_ticker is None:
            raise ValueError("Need cik in daily_df or cik_to_ticker bridge")
        bridge = cik_to_ticker.rename(columns={"symbol": "ticker"})
        bridge["cik"] = pd.to_numeric(bridge["cik"], errors="coerce").astype("Int64")
        daily = daily.merge(bridge[["cik", "ticker"]], on="ticker", how="left")

    # Pre-build a lookup table of cik → ticker for the events
    ticker_for_cik = (
        daily[["cik", "ticker"]].drop_duplicates("cik").set_index("cik")["ticker"].to_dict()
    )

    results: List[CarResult] = []
    for _, row in events.iterrows():
        cik = int(row["cik"]) if pd.notna(row["cik"]) else None
        if cik is None or cik not in ticker_for_cik:
            continue
        ticker = ticker_for_cik[cik]
        res = compute_car(
            daily, ticker, row["event_date"],
            window=window, spec=spec, estimation_window=estimation_window,
        )
        if res is not None:
            res.cik = cik
            results.append(res)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "cik": r.cik, "ticker": r.ticker, "event_date": r.event_date,
            "spec": r.spec,
            "window_lo": r.window[0], "window_hi": r.window[1],
            "car": r.car, "bhar": r.bhar, "n_days": r.n_days,
            "alpha": r.alpha, "beta": r.beta,
            "estimation_n_days": r.estimation_n_days,
        }
        for r in results
    ])

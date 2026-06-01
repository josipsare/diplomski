"""Aggregate event-study across many detected-anomaly firms.

Pipeline:
    detector scores (cik, fiscal_year, score)
        + restatement labels (cik, fiscal_year, is_restatement)
        + filed_date adapter (cik, fiscal_year → filed_date)
        + daily returns + market proxy
    →
    Per-event CAR/BHAR (CAPM-adjusted), then aggregate across events:
        - Mean CAR by relative-day-from-event (event-study chart)
        - Distribution of CAR(-30, +60) and CAR(-30, +250) per detector's top-K
        - Statistical test: is mean CAR significantly different from 0?
        - Comparison: top-K anomalous firms vs all-firm baseline

This is the missing half of the thesis topic ("utjecaj identificiranih
anomalija na poslovanje kompanije"). Without this, RQ2 is unanswered.

The analysis answers: do firms our detectors flag as anomalous experience
worse stock performance after filing? If yes → DL detection has actionable
forensic value beyond just labeling. If no (the bachelor's r=0.03 trap),
then the master's positive contribution is the methodology itself
(restatement-label evaluation, ensemble lift) rather than the impact claim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .abnormal_returns import compute_event_study

log = logging.getLogger(__name__)


@dataclass
class EventStudySummary:
    """Aggregate statistics over many events."""
    detector_name: str
    n_events: int
    n_completed: int          # events with computable CAR (enough trading days)
    spec: str
    window: tuple[int, int]
    mean_car: float
    median_car: float
    std_car: float
    t_stat: float             # mean CAR / SE — H0: mean = 0
    p_value: float            # two-sided p-value
    pct_negative: float       # fraction of events with CAR < 0
    mean_bhar: float


def event_study_for_top_k(
    scores_annual: pd.DataFrame,
    daily_df: pd.DataFrame,
    market_df: pd.DataFrame,
    long_sec: pd.DataFrame,
    cik_to_ticker: pd.DataFrame,
    *,
    k: int = 100,
    detector_name: str = "detector",
    window: tuple[int, int] = (-30, 60),
    spec: str = "capm",
    estimation_window: tuple[int, int] = (-250, -31),
) -> tuple[EventStudySummary, pd.DataFrame]:
    """Run event study for the top-K most-anomalous firm-years from a detector.

    Args:
        scores_annual: DataFrame with (cik, fiscal_year, score) — already
                       aggregated to firm-year granularity
        daily_df:      output of `daily_stock_loader.load_all_daily`
        market_df:     output of `daily_stock_loader.build_market_proxy`
        long_sec:      output of `sec_loader.load_long_panel` — for filed_date
        cik_to_ticker: bridge with cik + symbol columns
        k:             top-K firm-years to include
        detector_name: label for the result
        window:        event-window in trading days
        spec:          "raw", "market", or "capm"

    Returns:
        (EventStudySummary, raw per-event DataFrame)
    """
    from .event_dates import period_to_filed

    top_k = scores_annual.dropna(subset=["score"]).nlargest(k, "score").copy()
    if top_k.empty:
        return EventStudySummary(detector_name, 0, 0, spec, window, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, np.nan), pd.DataFrame()

    # Attach filed_date (use period_end derived from fiscal_year)
    top_k["period_end"] = pd.to_datetime(top_k["fiscal_year"].astype(str) + "-12-31")
    with_filed = period_to_filed(top_k, long_sec, cik_col="cik", period_col="period_end")
    events = with_filed.dropna(subset=["filed_date"]).rename(columns={"filed_date": "event_date"})

    if events.empty:
        return EventStudySummary(detector_name, len(top_k), 0, spec, window, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), pd.DataFrame()

    car_df = compute_event_study(
        events[["cik", "event_date"]], daily_df, market_df,
        window=window, spec=spec, estimation_window=estimation_window,
        cik_to_ticker=cik_to_ticker,
    )

    if car_df.empty:
        return EventStudySummary(detector_name, len(events), 0, spec, window, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), pd.DataFrame()

    # Drop duplicates by (cik, event_date) — defensive
    car_df = car_df.drop_duplicates(subset=["cik", "event_date"])

    cars = car_df["car"].dropna().to_numpy()
    bhars = car_df["bhar"].dropna().to_numpy()
    n_done = len(cars)
    if n_done < 2:
        return EventStudySummary(detector_name, len(events), n_done, spec, window,
                                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                 np.nan), car_df

    t_stat, p_value = scipy_stats.ttest_1samp(cars, popmean=0.0)
    summary = EventStudySummary(
        detector_name=detector_name,
        n_events=len(events),
        n_completed=n_done,
        spec=spec,
        window=window,
        mean_car=float(np.mean(cars)),
        median_car=float(np.median(cars)),
        std_car=float(np.std(cars, ddof=1)),
        t_stat=float(t_stat),
        p_value=float(p_value),
        pct_negative=float((cars < 0).mean()),
        mean_bhar=float(np.mean(bhars)) if len(bhars) > 0 else np.nan,
    )
    return summary, car_df


def event_study_baseline(
    panel_annual: pd.DataFrame,
    daily_df: pd.DataFrame,
    market_df: pd.DataFrame,
    long_sec: pd.DataFrame,
    cik_to_ticker: pd.DataFrame,
    *,
    n_random: int = 100,
    seed: int = 42,
    window: tuple[int, int] = (-30, 60),
    spec: str = "capm",
) -> EventStudySummary:
    """Random-firm baseline event study to compare against detector's top-K."""
    rng = np.random.default_rng(seed)
    sample = panel_annual.dropna(subset=["fiscal_year"]).sample(
        n=min(n_random, len(panel_annual)), random_state=seed,
    )[["cik", "fiscal_year"]].copy()
    sample["score"] = rng.uniform(0, 1, size=len(sample))  # arbitrary, just to mimic the API
    summary, _ = event_study_for_top_k(
        sample, daily_df, market_df, long_sec, cik_to_ticker,
        k=n_random, detector_name="random_baseline", window=window, spec=spec,
    )
    return summary


def compare_detectors_event_study(
    detector_score_files: Dict[str, pd.DataFrame],
    daily_df: pd.DataFrame,
    market_df: pd.DataFrame,
    long_sec: pd.DataFrame,
    cik_to_ticker: pd.DataFrame,
    *,
    k: int = 100,
    window: tuple[int, int] = (-30, 60),
    spec: str = "capm",
) -> pd.DataFrame:
    """Run event study across many detectors and tabulate the results."""
    rows: List[Dict] = []
    per_detector_cars: Dict[str, pd.DataFrame] = {}
    for name, scores in detector_score_files.items():
        summary, cars = event_study_for_top_k(
            scores, daily_df, market_df, long_sec, cik_to_ticker,
            k=k, detector_name=name, window=window, spec=spec,
        )
        rows.append({
            "detector": summary.detector_name,
            "n_top_k": k,
            "n_events_with_filed_date": summary.n_events,
            "n_car_computed": summary.n_completed,
            "mean_car": summary.mean_car,
            "median_car": summary.median_car,
            "std_car": summary.std_car,
            "t_stat": summary.t_stat,
            "p_value": summary.p_value,
            "pct_negative_car": summary.pct_negative,
            "mean_bhar": summary.mean_bhar,
        })
        per_detector_cars[name] = cars
    return pd.DataFrame(rows).sort_values("mean_car"), per_detector_cars

"""Benford's Law baseline detector — port of the bachelor's-thesis approach.

Operates on the **raw** long-format SEC numerical values (not the standardized
ratio panel — running Benford on z-scored ratios produces garbage). For each
(cik, fiscal_year) we collect every numerical fact reported by the firm, take
the absolute value, drop zeros, extract the first significant digit, and
compare its empirical distribution against Benford's expectation.

Score = first-digit MAD (Mean Absolute Deviation, in percentage points). Higher
MAD means greater deviation from Benford → more anomalous. We use MAD rather
than chi-square because chi-square is sample-size dependent (firms with more
reported values trivially score higher), whereas MAD is a per-firm measure of
"how unlike Benford the digit distribution is regardless of N."

The scorer also exposes per-(cik, year) chi-square, KS-statistic, and sample
size as supplementary columns for ablation.

This baseline is **the same approach the bachelor's thesis used.** Reproducing
it here lets the master's thesis show: (a) we recover the bachelor's findings,
(b) DL detectors do or don't beat it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .base import AnomalyDetector

# Benford's Law expected first-digit distribution (percentages 1-9)
_BENFORD_FIRST_DIGIT_PCT = np.array([30.103, 17.609, 12.494, 9.691, 7.918,
                                      6.695, 5.799, 5.115, 4.576])


def _first_digit(value: float) -> int | None:
    """Return the first significant digit of |value|, or None for invalid inputs.

    Handles values of any magnitude including |value| < 1 (e.g. EPS in dollars,
    per-share ratios). Earlier implementation used `int(abs(value))` which
    truncated values < 1 to 0, silently dropping per-share / ratio observations.
    The bias was firm-size correlated and would have hit Russell 3000 small-caps
    hardest.
    """
    if pd.isna(value) or value == 0:
        return None
    import math
    av = abs(float(value))
    if not math.isfinite(av):
        return None
    # Scale to the range [1, 10) by dividing by 10^floor(log10).
    exponent = math.floor(math.log10(av))
    leading = av / (10 ** exponent)
    digit = int(leading)
    if digit < 1 or digit > 9:
        return None  # numerical edge case
    return digit


def _benford_metrics_for_series(values: pd.Series) -> dict[str, float]:
    """Compute MAD, chi-square, p-value, KS-stat, and sample size for one firm-year."""
    if values is None or len(values) == 0:
        return {"mad": np.nan, "chi_square": np.nan, "p_value": np.nan,
                "ks_stat": np.nan, "n_samples": 0}

    digits = values.apply(_first_digit).dropna().astype(int)
    n = len(digits)
    if n == 0:
        return {"mad": np.nan, "chi_square": np.nan, "p_value": np.nan,
                "ks_stat": np.nan, "n_samples": 0}

    counts = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
    observed_pct = (counts / n) * 100
    expected_pct = _BENFORD_FIRST_DIGIT_PCT

    mad = float(np.mean(np.abs(observed_pct - expected_pct)))

    expected_counts = n * expected_pct / 100.0
    chi_square = float(np.sum((counts - expected_counts) ** 2 / expected_counts))
    p_value = float(1 - chi2.cdf(chi_square, df=8))

    # KS statistic computed manually against the Benford CDF.
    # The Benford CDF at integer digit d is log10(d + 1), i.e. the cumulative
    # mass over digits 1..d. The previous implementation used log10(1 + 1/d)
    # which is the Benford PMF (probability of digit d alone), giving the
    # wrong KS value everywhere except by coincidence at d=1.
    benford_cdf_at_d = np.cumsum(_BENFORD_FIRST_DIGIT_PCT) / 100.0  # shape (9,)
    empirical_cdf_at_d = np.cumsum(counts) / n
    ks_stat = float(np.max(np.abs(empirical_cdf_at_d - benford_cdf_at_d)))

    return {"mad": mad, "chi_square": chi_square, "p_value": p_value,
            "ks_stat": float(ks_stat), "n_samples": n}


class BenfordDetector(AnomalyDetector):
    """First-digit Benford anomaly detector.

    Unlike the other detectors, Benford does NOT need a fit step — the
    expected distribution is fixed by Benford's Law. `fit()` is a no-op that
    returns self for API symmetry. `score()` computes per-(cik, fiscal_year)
    metrics from the raw long-format SEC values.

    Args:
        min_samples: minimum number of digits per (cik, fiscal_year) bucket
                     for the score to be reported. Below this threshold the
                     row is dropped from the output. Default 30 follows
                     Nigrini (2012) — fewer than ~30 observations gives a
                     statistically meaningless MAD (a single observation has
                     MAD = ~21, dominating the score table).

    Intended use:
        det = BenfordDetector()
        det.fit(None)  # no-op
        long_df = pd.read_parquet("data/output/panels/long_sec.parquet")
        scores = det.score_panel(long_df)
    """

    name = "benford_first_digit"

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples

    def fit(self, X=None) -> "BenfordDetector":
        return self

    def _score_array(self, X) -> np.ndarray:
        """Not the natural API for Benford — always raises.

        Benford works on a long-format `(cik, period, value)` table, not on a
        feature matrix where each row is one firm-period. Calling
        `score_dataframe(panel)` (which delegates here) would silently compute
        per-row MAD over the panel's *columns* — a meaningless number that
        would corrupt any Phase 6 evaluation join.

        Always use `BenfordDetector.score_panel(long_df)` instead.
        """
        raise NotImplementedError(
            "BenfordDetector requires score_panel(long_df); the base-class "
            "score()/score_dataframe() API is wrong for this detector and would "
            "return per-row MAD over feature columns, which is meaningless."
        )

    def score_panel(
        self,
        long_df: pd.DataFrame,
        cik_col: str = "cik",
        period_col: str = "fiscal_year",
        value_col: str = "value",
    ) -> pd.DataFrame:
        """Compute per-(cik, fiscal_year) Benford metrics from long-format data.

        Args:
            long_df: DataFrame with at least cik, period_end (or fiscal_year), value.
                     If `period_col == "fiscal_year"` and that column is absent,
                     it is derived from `period_end.dt.year`.
            cik_col, period_col, value_col: column names

        Returns:
            DataFrame with columns:
                cik, <period_col>, score (= mad), mad, chi_square, p_value,
                ks_stat, n_samples, model_name
        """
        df = long_df.copy()
        if period_col == "fiscal_year" and "fiscal_year" not in df.columns:
            if "period_end" not in df.columns:
                raise KeyError("Need either fiscal_year or period_end column")
            df["fiscal_year"] = pd.to_datetime(df["period_end"]).dt.year

        # Take absolute values, drop NaN/zeros — Benford ignores those
        df = df[df[value_col].notna() & (df[value_col] != 0)].copy()
        df["_abs"] = df[value_col].abs()

        groups = df.groupby([cik_col, period_col])
        rows: list[dict] = []
        for (cik, period), grp in groups:
            metrics = _benford_metrics_for_series(grp["_abs"])
            metrics[cik_col] = cik
            metrics[period_col] = period
            rows.append(metrics)

        if not rows:
            return pd.DataFrame(columns=[
                cik_col, period_col, "score", "mad", "chi_square", "p_value",
                "ks_stat", "n_samples", "model_name",
            ])

        out = pd.DataFrame(rows)
        # Drop firm-years with too few digits — MAD on n=1 is meaningless
        # and would dominate the score table with spurious "anomalies".
        n_before = len(out)
        out = out[out["n_samples"] >= self.min_samples].copy()
        n_dropped = n_before - len(out)
        if n_dropped > 0:
            import logging as _log
            _log.getLogger(__name__).info(
                "BenfordDetector: dropped %d/%d (%.1f%%) firm-years with n_samples < %d",
                n_dropped, n_before, 100 * n_dropped / n_before, self.min_samples,
            )

        # Canonical anomaly score = MAD (higher = more deviation from Benford,
        # the naive "more anomalous" direction). EMPIRICAL FINDING (verified
        # 2026-05-10 against restatement labels on SP500 + Russell 3000):
        #   ROC-AUC with +MAD ≈ 0.46  (sub-random)
        #   ROC-AUC with -MAD ≈ 0.54  (weak inverted signal)
        # Restating firms have systematically LOWER MAD (more Benford-conformant)
        # — consistent with Amiram, Bozanic & Rouen (2015): manipulators
        # artificially smooth digit distributions to appear conventional. This
        # is a more interesting finding than "Benford is noise" and is reported
        # in the thesis as the headline Benford result. The detector exposes
        # both `score` (= +MAD, naive) and `score_inverted` (= -MAD, empirical
        # direction) so downstream evaluation can use whichever interpretation.
        out["score"] = out["mad"]
        out["score_inverted"] = -out["mad"]
        out["model_name"] = self.name

        # Schema alignment with other detectors (Phase 6 evaluation joins on
        # period_end). Annual Benford uses fiscal_year as its natural key; we
        # synthesize a period_end at the calendar year-end so the universal
        # schema (cik, period_end, score, model_name) holds.
        if period_col == "fiscal_year":
            out["period_end"] = pd.to_datetime(out["fiscal_year"].astype(str) + "-12-31")

        cols = [cik_col, period_col, "score", "score_inverted", "mad",
                "chi_square", "p_value", "ks_stat", "n_samples", "model_name"]
        if "period_end" in out.columns and "period_end" not in cols:
            cols.append("period_end")
        return out[cols].sort_values([cik_col, period_col]).reset_index(drop=True)

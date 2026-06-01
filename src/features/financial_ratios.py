"""Derive standard financial ratios from the raw tag-level panel.

These ratios are scale-invariant and roughly comparable across firms and sectors,
so they make a much better feature set for an autoencoder than raw dollars do.

The ratio set covers six families used in the forensic-accounting literature
(Beneish 1999, Dechow et al. 2011, Nigrini 2012):

    Profitability  — return on assets, return on equity, gross margin,
                     operating margin, net margin, ebit margin
    Liquidity      — current ratio, quick ratio, cash ratio
    Leverage       — debt-to-assets, debt-to-equity, equity-to-assets
    Efficiency     — asset turnover, inventory turnover, receivable turnover,
                     payables turnover, days-receivable, days-inventory
    Cash flow      — operating cf / total assets, capex / assets,
                     accruals / assets (a Beneish-style fraud feature),
                     free cash flow / revenues, cf / net income
    Growth         — revenue growth, asset growth, equity growth (year-over-year)

Every ratio is left as NaN when its denominator is zero or missing; the
normalization layer handles missingness uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where the denominator is 0/NaN."""
    den = den.where(den.abs() > 0)
    return num / den


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    """Fetch a column as a Series. If the column is absent from the panel,
    return a Series of `default` aligned to the DataFrame's index.

    Using `df.get(name, 0)` returns the *scalar* `0` when the column is missing,
    which silently breaks subsequent Series arithmetic. This helper guarantees
    a Series of the right length so subtraction and division behave as expected.
    """
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype="float64", name=name)


def compute_profitability(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["roa"] = _safe_div(_col(df, "net_income"), _col(df, "assets"))
    out["roe"] = _safe_div(_col(df, "net_income"), _col(df, "equity"))
    out["gross_margin"] = _safe_div(_col(df, "gross_profit"), _col(df, "revenues"))
    out["operating_margin"] = _safe_div(_col(df, "operating_income"), _col(df, "revenues"))
    out["net_margin"] = _safe_div(_col(df, "net_income"), _col(df, "revenues"))
    return out


def compute_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["current_ratio"] = _safe_div(_col(df, "assets_current"), _col(df, "liabilities_current"))
    quick_assets = _col(df, "assets_current").fillna(0) - _col(df, "inventory").fillna(0)
    # If both source columns are missing entirely, treat quick_ratio as NaN
    # (not as 0 / liabilities_current = 0) by masking on observed inputs.
    quick_observed = _col(df, "assets_current").notna() | _col(df, "inventory").notna()
    out["quick_ratio"] = _safe_div(
        quick_assets.where(quick_observed), _col(df, "liabilities_current")
    )
    out["cash_ratio"] = _safe_div(_col(df, "cash"), _col(df, "liabilities_current"))
    return out


def compute_leverage(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    long_td = _col(df, "long_term_debt").fillna(0)
    short_td = _col(df, "short_term_debt").fillna(0)
    total_debt = long_td + short_td
    debt_observed = _col(df, "long_term_debt").notna() | _col(df, "short_term_debt").notna()
    total_debt = total_debt.where(debt_observed)

    out["debt_to_assets"] = _safe_div(total_debt, _col(df, "assets"))
    out["debt_to_equity"] = _safe_div(total_debt, _col(df, "equity"))
    out["liabilities_to_assets"] = _safe_div(_col(df, "liabilities"), _col(df, "assets"))
    out["equity_to_assets"] = _safe_div(_col(df, "equity"), _col(df, "assets"))
    out["interest_coverage"] = _safe_div(_col(df, "operating_income"), _col(df, "interest_expense"))
    return out


def compute_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["asset_turnover"] = _safe_div(_col(df, "revenues"), _col(df, "assets"))
    out["inventory_turnover"] = _safe_div(_col(df, "cost_of_revenue"), _col(df, "inventory"))
    out["receivables_turnover"] = _safe_div(_col(df, "revenues"), _col(df, "receivables"))
    out["payables_turnover"] = _safe_div(_col(df, "cost_of_revenue"), _col(df, "payables"))
    out["days_receivable"] = (365 / out["receivables_turnover"]).replace(
        [np.inf, -np.inf], np.nan
    )
    out["days_inventory"] = (365 / out["inventory_turnover"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def compute_cash_flow(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["cf_operating_to_assets"] = _safe_div(_col(df, "cf_operating"), _col(df, "assets"))
    out["capex_to_assets"] = _safe_div(_col(df, "capex"), _col(df, "assets"))

    cfo = _col(df, "cf_operating").fillna(0)
    capex = _col(df, "capex").fillna(0)
    fcf = cfo - capex
    fcf_observed = _col(df, "cf_operating").notna() | _col(df, "capex").notna()
    fcf = fcf.where(fcf_observed)
    out["fcf_to_revenues"] = _safe_div(fcf, _col(df, "revenues"))

    # Two complementary cf-to-NI specs (decision 2026-05-09):
    #   cf_to_net_income  uses |NI| → stable winsorize, sign-collapsed
    #   cf_to_signed_ni   uses signed NI → preserves the forensic-relevant
    #                     "loss-firm with positive cash flow" signal
    # The signed version explodes when NI is near zero; rely on the per-period
    # 1/99 winsorization in normalization.py to tame extreme tails.
    out["cf_to_net_income"] = _safe_div(_col(df, "cf_operating"), _col(df, "net_income").abs())
    out["cf_to_signed_ni"] = _safe_div(_col(df, "cf_operating"), _col(df, "net_income"))

    # Accruals = net income minus operating cash flow (cash-flow-statement approach;
    # Richardson et al. 2005). For a Beneish-style balance-sheet accruals measure,
    # see compute_balance_sheet_accruals (Phase 2.5+, when both observations exist).
    ni = _col(df, "net_income").fillna(0)
    accruals = ni - cfo
    accruals_observed = _col(df, "net_income").notna() | _col(df, "cf_operating").notna()
    accruals = accruals.where(accruals_observed)
    out["accruals_to_assets"] = _safe_div(accruals, _col(df, "assets"))
    return out


def compute_beneish_accruals(
    df: pd.DataFrame,
    group_key: str = "cik",
    time_col: str | None = None,
) -> pd.DataFrame:
    """Beneish (1999) balance-sheet accruals as a Beneish M-score component.

    Total accruals (TA) = (ΔCA − ΔCash) − (ΔCL − ΔSTD − ΔITP) − Depreciation
    where Δ is one-period lag (within-firm), and the result is divided by
    Total Assets to give a scale-invariant ratio.

        ΔCA   = ΔAssetsCurrent
        ΔCash = ΔCashAndCashEquivalentsAtCarryingValue
        ΔCL   = ΔLiabilitiesCurrent
        ΔSTD  = ΔLongTermDebtCurrent (current portion of LT debt;
                  some sources also subtract ΔShortTermBorrowings — we use the
                  GAAP "current maturities of LT debt" which is the standard
                  Beneish reference; a sensitivity check using the broader
                  short-term-borrowings tag would be a Phase-6 ablation)
        ΔITP  = ΔAccruedIncomeTaxesCurrent

    Caller must sort by `(group_key, time_col)` before calling. If `time_col` is
    given, this function asserts strictly increasing within each group.

    Returns:
        DataFrame with columns:
            beneish_accruals_to_assets   the TA / Total Assets ratio
            beneish_wc_accruals          (ΔCA − ΔCash − ΔCL + ΔSTD + ΔITP) / TA
                                         (working-capital piece alone, useful
                                          for ablation against the full TA)
    """
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time_col {time_col!r} not in df.columns")
        bad = (
            df.groupby(group_key)[time_col].apply(lambda s: not s.is_monotonic_increasing)
        )
        if bad.any():
            offenders = bad[bad].index.tolist()[:5]
            raise AssertionError(
                f"compute_beneish_accruals requires df sorted by ({group_key}, {time_col}); "
                f"out-of-order in groups (first 5): {offenders}"
            )

    def _delta(col: str) -> pd.Series:
        return _col(df, col).groupby(df[group_key], sort=False).diff()

    d_ca = _delta("assets_current")
    d_cash = _delta("cash")
    d_cl = _delta("liabilities_current")
    d_std = _delta("current_maturities_lt_debt")
    d_itp = _delta("income_taxes_payable")
    depr = _col(df, "depreciation_amortization")
    assets = _col(df, "assets")

    # Working-capital accruals: ΔCA − ΔCash − ΔCL + ΔSTD + ΔITP
    wc = (
        d_ca.fillna(0) - d_cash.fillna(0)
        - d_cl.fillna(0) + d_std.fillna(0) + d_itp.fillna(0)
    )
    # NaN-mask: WC is meaningful only if at least the two main pieces (ΔCA, ΔCL)
    # are observed
    wc_observed = d_ca.notna() & d_cl.notna()
    wc = wc.where(wc_observed)

    # Total accruals = WC − Depreciation. Relaxed: missing depreciation is
    # treated as 0 (some filers embed depreciation inside operating expenses
    # rather than reporting the explicit GAAP tag). The Beneish formula
    # technically requires depreciation, but discarding the row whenever it's
    # missing loses substantial coverage. Documented caveat: TA may slightly
    # understate accruals for firms that don't report depreciation explicitly.
    ta = wc - depr.fillna(0)
    ta = ta.where(wc_observed)  # only require WC pieces, not depreciation

    out = pd.DataFrame(index=df.index)
    out["beneish_wc_accruals"] = _safe_div(wc, assets)
    out["beneish_accruals_to_assets"] = _safe_div(ta, assets)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_growth(
    df: pd.DataFrame,
    group_key: str = "cik",
    time_col: str | None = None,
) -> pd.DataFrame:
    """Year-over-year growth ratios; require sequential rows per firm.

    The caller is responsible for sorting `df` by `(group_key, time_col)` before
    calling. If `time_col` is provided, this function asserts that ordering is
    strictly increasing within each group.
    """
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time_col {time_col!r} not in df.columns")
        # Assert strictly increasing within each group
        bad = (
            df.groupby(group_key)[time_col]
            .apply(lambda s: not s.is_monotonic_increasing)
        )
        if bad.any():
            offenders = bad[bad].index.tolist()[:5]
            raise AssertionError(
                f"compute_growth requires df sorted by ({group_key}, {time_col}); "
                f"out-of-order in groups (first 5): {offenders}"
            )

    out = pd.DataFrame(index=df.index)
    for col, name in [
        ("revenues", "revenue_growth"),
        ("assets", "asset_growth"),
        ("equity", "equity_growth"),
        ("net_income", "net_income_growth"),
        ("cf_operating", "cf_operating_growth"),
    ]:
        if col in df.columns:
            grp = df.groupby(group_key, sort=False)[col]
            growth = grp.pct_change(fill_method=None)
            growth = growth.replace([np.inf, -np.inf], np.nan)
            out[name] = growth
    return out


def compute_size_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-scale size features so model can use them without exploding."""
    out = pd.DataFrame(index=df.index)
    for col in ("assets", "revenues", "equity"):
        if col in df.columns:
            out[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return out


_RATIO_COLUMNS_VALIDATED = False


def compute_all_ratios(
    df: pd.DataFrame,
    group_key: str = "cik",
    time_col: str | None = None,
    skip_validation: bool = False,
) -> pd.DataFrame:
    """Run every ratio family and concatenate side-by-side.

    `df` must contain the raw tag aliases from `tags.py`. Missing columns are
    handled by `_col` (returns a NaN Series) and `_safe_div`.

    For growth and Beneish-accrual ratios, `df` must be sorted by
    `(group_key, time_col)`. Pass `time_col` to enable an assertion that the
    order is correct.

    `skip_validation=True` bypasses the one-time `_validate_ratio_columns_match`
    self-check (used by the helper itself to avoid recursion).
    """
    parts = [
        compute_profitability(df),
        compute_liquidity(df),
        compute_leverage(df),
        compute_efficiency(df),
        compute_cash_flow(df),
        compute_beneish_accruals(df, group_key=group_key, time_col=time_col),
        compute_growth(df, group_key=group_key, time_col=time_col),
        compute_size_features(df),
    ]
    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)

    global _RATIO_COLUMNS_VALIDATED
    if not skip_validation and not _RATIO_COLUMNS_VALIDATED:
        _validate_ratio_columns_match()
        _RATIO_COLUMNS_VALIDATED = True

    return out


RATIO_COLUMNS = [
    # Profitability
    "roa", "roe", "gross_margin", "operating_margin", "net_margin",
    # Liquidity
    "current_ratio", "quick_ratio", "cash_ratio",
    # Leverage
    "debt_to_assets", "debt_to_equity", "liabilities_to_assets",
    "equity_to_assets", "interest_coverage",
    # Efficiency
    "asset_turnover", "inventory_turnover", "receivables_turnover",
    "payables_turnover", "days_receivable", "days_inventory",
    # Cash flow / accruals (CF-statement approach)
    "cf_operating_to_assets", "capex_to_assets", "fcf_to_revenues",
    "cf_to_net_income", "cf_to_signed_ni", "accruals_to_assets",
    # Beneish balance-sheet accruals (forensic-accounting standard)
    "beneish_wc_accruals", "beneish_accruals_to_assets",
    # Growth
    "revenue_growth", "asset_growth", "equity_growth",
    "net_income_growth", "cf_operating_growth",
    # Log size
    "log_assets", "log_revenues", "log_equity",
]


def _validate_ratio_columns_match() -> None:
    """Sanity-check that `RATIO_COLUMNS` equals what `compute_all_ratios` produces.

    Drift would silently break downstream code that iterates over RATIO_COLUMNS.
    Auto-called once from `compute_all_ratios`; can also be invoked from tests.
    """
    dummy_cols = [
        "net_income", "assets", "equity", "gross_profit", "revenues",
        "operating_income", "assets_current", "liabilities_current", "inventory",
        "cash", "long_term_debt", "short_term_debt", "liabilities",
        "interest_expense", "cost_of_revenue", "receivables", "payables",
        "cf_operating", "capex",
        # Beneish balance-sheet inputs
        "current_maturities_lt_debt", "income_taxes_payable",
        "depreciation_amortization",
    ]
    dummy = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in dummy_cols})
    dummy["cik"] = [1, 1, 1]
    produced = compute_all_ratios(dummy, group_key="cik", skip_validation=True)
    expected = set(RATIO_COLUMNS)
    actual = set(produced.columns)
    if expected != actual:
        missing = expected - actual
        extra = actual - expected
        raise AssertionError(
            f"RATIO_COLUMNS ({len(expected)}) does not match compute_all_ratios "
            f"output ({len(actual)}). Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

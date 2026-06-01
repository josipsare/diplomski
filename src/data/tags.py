"""Curated US-GAAP XBRL tag list used as the feature vector for anomaly detection.

Tags are split into three groups by reporting convention:
    BALANCE_TAGS  — instant values at period end (qtrs == 0)
    FLOW_TAGS     — flow values reported per period (qtrs in {1, 4})
    PER_SHARE_TAGS — already normalized per share, treat like flows

For each tag we also keep a short alias used as the column name in the wide panel.
The alias is shorter and stable across small XBRL renames; the canonical tag may
have synonyms (e.g. Revenues vs RevenueFromContractWithCustomerExcludingAssessedTax)
which the loader resolves with FALLBACK chains.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (canonical_tag, alias) — primary tag we look for first.
# Multiple synonyms per alias are handled in FALLBACK_CHAINS.

BALANCE_TAGS: List[Tuple[str, str]] = [
    ("Assets", "assets"),
    ("AssetsCurrent", "assets_current"),
    ("AssetsNoncurrent", "assets_noncurrent"),
    ("CashAndCashEquivalentsAtCarryingValue", "cash"),
    ("AccountsReceivableNetCurrent", "receivables"),
    ("InventoryNet", "inventory"),
    ("PropertyPlantAndEquipmentNet", "ppe_net"),
    ("Goodwill", "goodwill"),
    ("IntangibleAssetsNetExcludingGoodwill", "intangibles"),
    ("Liabilities", "liabilities"),
    ("LiabilitiesCurrent", "liabilities_current"),
    ("LiabilitiesNoncurrent", "liabilities_noncurrent"),
    ("AccountsPayableCurrent", "payables"),
    ("LongTermDebtNoncurrent", "long_term_debt"),
    ("ShortTermBorrowings", "short_term_debt"),
    ("DeferredRevenueCurrent", "deferred_revenue"),
    ("StockholdersEquity", "equity"),
    ("RetainedEarningsAccumulatedDeficit", "retained_earnings"),
    ("CommonStockSharesOutstanding", "shares_outstanding"),
    ("CommonStockSharesIssued", "shares_issued"),
    ("TreasuryStockValue", "treasury_stock"),
    ("AdditionalPaidInCapital", "paid_in_capital"),
    ("AccruedIncomeTaxesCurrent", "income_taxes_payable"),
    ("LongTermDebtCurrent", "current_maturities_lt_debt"),
]

FLOW_TAGS: List[Tuple[str, str]] = [
    # Income statement
    ("Revenues", "revenues"),
    ("CostOfRevenue", "cost_of_revenue"),
    ("GrossProfit", "gross_profit"),
    ("OperatingExpenses", "operating_expenses"),
    ("ResearchAndDevelopmentExpense", "rnd"),
    ("SellingGeneralAndAdministrativeExpense", "sga"),
    ("DepreciationDepletionAndAmortization", "depreciation_amortization"),
    ("OperatingIncomeLoss", "operating_income"),
    ("InterestExpense", "interest_expense"),
    ("IncomeTaxExpenseBenefit", "income_tax"),
    ("NetIncomeLoss", "net_income"),
    ("ComprehensiveIncomeNetOfTax", "comprehensive_income"),
    # Cash flow statement
    ("NetCashProvidedByUsedInOperatingActivities", "cf_operating"),
    ("NetCashProvidedByUsedInInvestingActivities", "cf_investing"),
    ("NetCashProvidedByUsedInFinancingActivities", "cf_financing"),
    ("PaymentsToAcquirePropertyPlantAndEquipment", "capex"),
    ("PaymentsForRepurchaseOfCommonStock", "buybacks"),
    ("PaymentsOfDividendsCommonStock", "dividends"),
    ("ProceedsFromIssuanceOfLongTermDebt", "debt_issued"),
    ("RepaymentsOfLongTermDebt", "debt_repaid"),
    ("IncreaseDecreaseInAccountsReceivable", "delta_receivables"),
    ("IncreaseDecreaseInInventories", "delta_inventory"),
    ("IncreaseDecreaseInAccountsPayable", "delta_payables"),
    ("ShareBasedCompensation", "stock_comp"),
]

PER_SHARE_TAGS: List[Tuple[str, str]] = [
    ("EarningsPerShareBasic", "eps_basic"),
    ("EarningsPerShareDiluted", "eps_diluted"),
    ("WeightedAverageNumberOfSharesOutstandingBasic", "weighted_shares_basic"),
    ("WeightedAverageNumberOfDilutedSharesOutstanding", "weighted_shares_diluted"),
]

# When the primary tag is missing for a filing we try these synonyms in order.
# Many companies use the newer ASC-606 revenue tag instead of the legacy "Revenues".
FALLBACK_CHAINS: Dict[str, List[str]] = {
    "Revenues": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "CostOfRevenue": ["CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"],
    "DepreciationDepletionAndAmortization": [
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    # OperatingIncomeLoss has NO fallback by design (decision 2026-05-09).
    # The previous synonym (IncomeLossFromContinuingOperationsBeforeIncomeTaxes…)
    # is pre-tax income — it includes interest expense and non-operating items,
    # so substituting it as "operating income" was semantically wrong for any
    # leveraged firm (banks, utilities, REITs ≈ 10% of the universe). A NaN
    # propagating through downstream ratios is more honest than a contaminated
    # value. See memory/thesis_design_decisions.md item 7.
    "ShortTermBorrowings": ["ShortTermBankLoansAndNotesPayable", "CommercialPaper"],
    "LongTermDebtNoncurrent": ["LongTermDebt"],
}


def all_tags() -> List[Tuple[str, str]]:
    """Return the full list of (canonical_tag, alias) pairs."""
    return BALANCE_TAGS + FLOW_TAGS + PER_SHARE_TAGS


def is_balance(canonical_tag: str) -> bool:
    """True if the tag is a balance-sheet (instant) item."""
    return any(t == canonical_tag for t, _ in BALANCE_TAGS)


def is_flow(canonical_tag: str) -> bool:
    """True if the tag is a flow item (income / cash flow)."""
    return any(t == canonical_tag for t, _ in FLOW_TAGS) or any(
        t == canonical_tag for t, _ in PER_SHARE_TAGS
    )


def alias_of(canonical_tag: str) -> str:
    """Look up the alias for a canonical tag. Raises if unknown."""
    for tag, alias in all_tags():
        if tag == canonical_tag:
            return alias
    raise KeyError(canonical_tag)


def canonical_tags_with_synonyms() -> Dict[str, List[str]]:
    """Map every canonical tag to its full synonym list (canonical first).

    Used by the loader to resolve the "best available" tag per (filing, alias).
    """
    out: Dict[str, List[str]] = {}
    for tag, _alias in all_tags():
        out[tag] = [tag] + FALLBACK_CHAINS.get(tag, [])
    return out

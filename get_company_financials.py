"""
Simple script to download financial data for a specific company
"""

from sec_data_parser import SECDataParser
import pandas as pd
from pathlib import Path


def get_company_financials(company_name, data_dir="./sec_data/2024q4", output_file=None):
    """
    Get financial data for a company.

    Args:
        company_name: Company name to search for (e.g., "Apple", "Microsoft")
        data_dir: Directory with SEC data
        output_file: Optional CSV file to save results

    Returns:
        DataFrame with financial data
    """
    # Initialize parser
    parser = SECDataParser(data_dir)

    # Search for company
    print(f"Searching for '{company_name}'...")
    companies = parser.get_company_by_name(company_name)

    if companies.empty:
        print(f"No companies found matching '{company_name}'")
        return None

    # Show found companies
    print(f"\nFound {len(companies)} companies:")
    for idx, row in companies.iterrows():
        print(f"  {int(row['cik']):010d} - {row['name']}")

    # Get the first match (or you can choose)
    company = companies.iloc[0]
    cik = int(company['cik'])
    company_name = company['name']

    print(f"\nUsing: {company_name} (CIK: {cik:010d})")

    # Get all submissions for this company
    print("\nFetching submissions...")
    submissions = parser.get_company_submissions(cik)

    if submissions.empty:
        print("No submissions found for this company in this quarter")
        return None

    print(f"\nSubmissions:")
    print(submissions[['form', 'period', 'filed']].to_string())

    # Get all financial data
    print("\nFetching all financial data...")
    financials = parser.get_company_financials(cik)

    if financials.empty:
        print("No financial data found")
        return None

    print(f"\nFound {len(financials):,} financial data points")
    print(f"Unique tags: {financials['tag'].nunique()}")

    # Show sample
    print("\nSample of financial data:")
    sample = financials[['tag', 'value', 'uom', 'period', 'form']].head(20)
    print(sample.to_string(index=False))

    # Export to CSV if requested
    if output_file:
        output_path = Path(output_file)
        financials.to_csv(output_path, index=False)
        print(f"\nData exported to: {output_path.absolute()}")

    return financials


def get_key_metrics(company_name, data_dir="./sec_data/2024q4", output_file=None):
    """
    Get only key financial metrics for a company.

    Args:
        company_name: Company name to search for
        data_dir: Directory with SEC data
        output_file: Optional CSV file to save results

    Returns:
        DataFrame with key metrics
    """
    # Common key financial metrics
    key_tags = [
        'Assets',
        'AssetsCurrent',
        'Liabilities',
        'LiabilitiesCurrent',
        'StockholdersEquity',
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'CostOfRevenue',
        'GrossProfit',
        'OperatingIncomeLoss',
        'NetIncomeLoss',
        'EarningsPerShareBasic',
        'EarningsPerShareDiluted',
        'CashAndCashEquivalentsAtCarryingValue',
        'PropertyPlantAndEquipmentNet',
        'ComprehensiveIncomeNetOfTax'
    ]

    # Initialize parser
    parser = SECDataParser(data_dir)

    # Search for company
    print(f"Searching for '{company_name}'...")
    companies = parser.get_company_by_name(company_name)

    if companies.empty:
        print(f"No companies found matching '{company_name}'")
        return None

    # Get first match
    company = companies.iloc[0]
    cik = int(company['cik'])
    company_name = company['name']

    print(f"\nUsing: {company_name} (CIK: {cik:010d})")

    # Get key financials
    print("\nFetching key financial metrics...")
    financials = parser.get_company_financials(cik, tags=key_tags)

    if financials.empty:
        print("No key metrics found")
        return None

    print(f"\nFound {len(financials):,} key metric data points")

    # Pivot to show metrics over time
    print("\nKey Metrics by Period:")
    pivot = financials.pivot_table(
        index='tag',
        columns='period',
        values='value',
        aggfunc='first'
    )
    print(pivot.to_string())

    # Export to CSV if requested
    if output_file:
        output_path = Path(output_file)
        financials.to_csv(output_path, index=False)
        print(f"\nData exported to: {output_path.absolute()}")

    return financials


if __name__ == "__main__":
    import sys

    # Check if company name provided
    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:])
    else:
        # Default example
        company_name = "Apple"

    print("="*70)
    print(f"Getting financials for: {company_name}")
    print("="*70 + "\n")

    # Method 1: Get all financial data
    print("\n" + "="*70)
    print("METHOD 1: Get ALL Financial Data")
    print("="*70 + "\n")

    output_file = f"{company_name.replace(' ', '_')}_all_financials.csv"
    all_data = get_company_financials(company_name, output_file=output_file)

    # Method 2: Get only key metrics
    print("\n\n" + "="*70)
    print("METHOD 2: Get KEY Metrics Only")
    print("="*70 + "\n")

    output_file = f"{company_name.replace(' ', '_')}_key_metrics.csv"
    key_data = get_key_metrics(company_name, output_file=output_file)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nYou can now analyze the CSV files or use the DataFrames in Python")

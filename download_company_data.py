"""
Simple script to download financial data for ANY company.
Just run: python download_company_data.py
"""

from sec_data_parser import SECDataParser
import pandas as pd


def download_company_financials():
    """Interactive script to download company financials."""

    # Initialize parser
    parser = SECDataParser("./sec_data/2024q4")

    print("="*70)
    print("COMPANY FINANCIAL DATA DOWNLOADER")
    print("="*70)

    # Get company name from user
    company_name = input("\nEnter company name (e.g., Apple Inc, Microsoft, Tesla): ").strip()

    if not company_name:
        print("No company name provided. Exiting.")
        return

    # Search for company
    print(f"\nSearching for '{company_name}'...")
    companies = parser.get_company_by_name(company_name)

    if companies.empty:
        print(f"❌ No companies found matching '{company_name}'")
        return

    # Show all matches
    print(f"\n✓ Found {len(companies)} companies:\n")
    for idx, (_, row) in enumerate(companies.iterrows(), 1):
        print(f"  [{idx}] CIK {int(row['cik']):010d} - {row['name']}")

    # Let user choose
    if len(companies) > 1:
        choice = input(f"\nSelect company [1-{len(companies)}] (default: 1): ").strip()
        try:
            choice = int(choice) - 1 if choice else 0
            if choice < 0 or choice >= len(companies):
                choice = 0
        except:
            choice = 0
    else:
        choice = 0

    # Get selected company
    company = companies.iloc[choice]
    cik = int(company['cik'])
    name = company['name']

    print(f"\n{'='*70}")
    print(f"Selected: {name}")
    print(f"CIK: {cik:010d}")
    print(f"{'='*70}")

    # Get submissions
    print("\n📄 Fetching submissions...")
    submissions = parser.get_company_submissions(cik)

    if submissions.empty:
        print("❌ No submissions found for this company in Q4 2024")
        return

    print(f"\n✓ Found {len(submissions)} submissions:\n")
    print(submissions[['form', 'period', 'filed']].to_string())

    # Get all financial data
    print("\n💰 Fetching financial data...")
    financials = parser.get_company_financials(cik)

    if financials.empty:
        print("❌ No financial data found")
        return

    print(f"\n✓ Found {len(financials):,} financial data points")
    print(f"✓ {financials['tag'].nunique()} unique financial metrics")

    # Show summary by form type
    print("\n📊 Data by form type:")
    form_counts = financials['form'].value_counts()
    for form, count in form_counts.items():
        print(f"   {form}: {count:,} data points")

    # Show most common metrics
    print("\n🔝 Top 10 most common metrics:")
    top_tags = financials['tag'].value_counts().head(10)
    for tag, count in top_tags.items():
        print(f"   {tag}: {count}")

    # Export options
    print(f"\n{'='*70}")
    export = input("\nExport to CSV? (y/n, default: y): ").strip().lower()

    if export != 'n':
        # Clean filename
        filename = name.replace(' ', '_').replace(',', '').replace('.', '')
        filename = f"{filename}_financials_2024q4.csv"

        financials.to_csv(filename, index=False)
        print(f"\n✅ Data exported to: {filename}")
        print(f"   Size: {len(financials):,} rows x {len(financials.columns)} columns")

        # Also create a summary
        summary_file = filename.replace('.csv', '_summary.csv')

        # Get key metrics only
        key_tags = ['Assets', 'Revenues', 'NetIncomeLoss', 'StockholdersEquity',
                   'EarningsPerShareBasic', 'CashAndCashEquivalentsAtCarryingValue']

        key_data = financials[financials['tag'].isin(key_tags)]
        if not key_data.empty:
            key_data.to_csv(summary_file, index=False)
            print(f"✅ Summary exported to: {summary_file}")

    print(f"\n{'='*70}")
    print("DONE! 🎉")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        download_company_financials()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

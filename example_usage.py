"""
Example Usage: SEC Financial Data Download and Analysis

This script demonstrates how to:
1. Download SEC quarterly financial data
2. Parse and analyze the data
3. Extract company-specific information
4. Perform basic financial analysis
"""

from sec_data_downloader import SECDataDownloader
from sec_data_parser import SECDataParser
import pandas as pd
from pathlib import Path


def example_download():
    """Example: Download SEC data"""
    print("="*70)
    print("EXAMPLE 1: Downloading SEC Data")
    print("="*70 + "\n")

    # IMPORTANT: Replace with your information
    USER_AGENT = "Josip Sare josip.sare@gmail.com"

    # Initialize downloader
    downloader = SECDataDownloader(
        user_agent=USER_AGENT,
        output_dir="./sec_data"
    )

    # Download latest quarter
    print("Downloading latest quarter...")
    data_dir = downloader.download_latest()

    if data_dir:
        print(f"\nData downloaded to: {data_dir}")
        return data_dir
    else:
        print("\nFailed to download data")
        return None


def example_basic_parsing(data_dir: Path):
    """Example: Basic data parsing"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Basic Data Parsing")
    print("="*70 + "\n")

    # Initialize parser
    parser = SECDataParser(data_dir)

    # Get statistics
    print("Data Statistics:")
    stats = parser.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Load submissions
    print("\nLoading submissions...")
    submissions = parser.load_submissions()
    print(f"Total submissions: {len(submissions):,}")
    print(f"\nSample submissions:")
    print(submissions[['name', 'form', 'period', 'filed']].head(10))

    # Form type distribution
    print("\nForm type distribution:")
    form_counts = submissions['form'].value_counts().head(10)
    for form, count in form_counts.items():
        print(f"  {form}: {count:,}")


def example_company_search(data_dir: Path):
    """Example: Search for companies"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Company Search")
    print("="*70 + "\n")

    parser = SECDataParser(data_dir)

    # Search by name
    search_terms = ["Apple", "Microsoft", "Tesla", "Amazon"]

    for term in search_terms:
        print(f"\nSearching for '{term}'...")
        results = parser.get_company_by_name(term)

        if not results.empty:
            print(f"Found {len(results)} companies:")
            for _, row in results.head(3).iterrows():
                print(f"  CIK: {int(row['cik']):010d} - {row['name']}")
        else:
            print(f"  No companies found")


def example_company_financials(data_dir: Path):
    """Example: Get company financial data"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Company Financial Analysis")
    print("="*70 + "\n")

    parser = SECDataParser(data_dir)

    # Apple's CIK
    apple_cik = "0000320193"

    # Search for Apple first
    print("Searching for Apple...")
    companies = parser.get_company_by_name("Apple Inc", exact=True)
    if companies.empty:
        companies = parser.get_company_by_name("Apple")

    if not companies.empty:
        print(f"Found: {companies.iloc[0]['name']}")
        apple_cik = str(int(companies.iloc[0]['cik']))
        print(f"CIK: {apple_cik}")

        # Get submissions
        print(f"\nApple's recent submissions:")
        submissions = parser.get_company_submissions(apple_cik)
        if not submissions.empty:
            print(submissions[['form', 'period', 'filed']].head(10))

            # Get financial summary for 10-K (annual reports)
            print(f"\nApple's financial summary (10-K filings):")
            summary = parser.get_financial_summary(apple_cik, form='10-K')

            if not summary.empty:
                # Display key metrics
                key_metrics = ['Assets', 'Revenues', 'NetIncomeLoss']
                available_metrics = [m for m in key_metrics if m in summary.columns]

                if available_metrics:
                    print(summary[available_metrics].head())

                    # Export to CSV
                    output_file = Path("./apple_financials.csv")
                    parser.export_to_csv(summary, output_file)
                else:
                    print("No key metrics found in this dataset")
            else:
                print("No 10-K data found for Apple in this quarter")
        else:
            print("No submissions found")
    else:
        print("Apple not found in this dataset")


def example_tag_search(data_dir: Path):
    """Example: Search for XBRL tags"""
    print("\n" + "="*70)
    print("EXAMPLE 5: XBRL Tag Search")
    print("="*70 + "\n")

    parser = SECDataParser(data_dir)

    # Search for tags
    keywords = ["revenue", "asset", "liability", "equity"]

    for keyword in keywords:
        print(f"\nSearching for '{keyword}' tags...")
        tags = parser.search_tags(keyword)

        if not tags.empty:
            print(f"Found {len(tags)} tags. Top 5:")
            for _, row in tags.head(5).iterrows():
                print(f"  {row['tag']}: {row['tlabel']}")
        else:
            print(f"  No tags found")


def example_specific_metric_analysis(data_dir: Path):
    """Example: Analyze specific financial metrics"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Specific Metric Analysis")
    print("="*70 + "\n")

    parser = SECDataParser(data_dir)

    # Get all companies with revenue data
    print("Finding companies with revenue data...")

    # Revenue-related tags
    revenue_tags = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']

    # Load numerical data
    num_df = parser.load_numbers()

    # Filter for revenue tags
    revenue_data = num_df[num_df['tag'].isin(revenue_tags)].copy()

    if not revenue_data.empty:
        print(f"Found {len(revenue_data):,} revenue data points")

        # Get top 10 by value
        top_revenues = revenue_data.nlargest(10, 'value')

        # Merge with submission info to get company names
        sub_df = parser.load_submissions()
        top_revenues = top_revenues.merge(
            sub_df[['adsh', 'name', 'form', 'period']],
            on='adsh',
            how='left'
        )

        print("\nTop 10 revenue values in this quarter:")
        for idx, row in top_revenues.iterrows():
            company = row['name']
            value = row['value'] / 1_000_000  # Convert to millions
            period = row['period']
            print(f"  {company}: ${value:,.0f}M (Period: {period})")
    else:
        print("No revenue data found in this quarter")


def example_form_analysis(data_dir: Path):
    """Example: Analyze specific form types"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Form Type Analysis")
    print("="*70 + "\n")

    parser = SECDataParser(data_dir)
    sub_df = parser.load_submissions()

    # Analyze 10-K filings (annual reports)
    print("Analyzing 10-K filings (annual reports)...")
    filings_10k = sub_df[sub_df['form'] == '10-K']

    if not filings_10k.empty:
        print(f"\nTotal 10-K filings: {len(filings_10k):,}")

        # Industry distribution (by SIC code)
        print("\nTop 10 industries (by SIC code):")
        sic_counts = filings_10k['sic'].value_counts().head(10)
        for sic, count in sic_counts.items():
            print(f"  SIC {sic}: {count} filings")

        # Recent filings
        print("\nMost recent 10-K filings:")
        recent = filings_10k.nlargest(10, 'filed')
        for _, row in recent.iterrows():
            print(f"  {row['name']} - Filed: {row['filed'].date()}")
    else:
        print("No 10-K filings found in this quarter")


def main():
    """Main function to run all examples"""
    print("\n" + "="*70)
    print("SEC FINANCIAL DATA - COMPREHENSIVE EXAMPLES")
    print("="*70)

    # Example 1: Download data
    data_dir = example_download()

    if not data_dir:
        print("\nCannot proceed without data. Please check your internet connection")
        print("and ensure your User-Agent is set correctly.")
        return

    # Example 2: Basic parsing
    example_basic_parsing(data_dir)

    # Example 3: Company search
    example_company_search(data_dir)

    # Example 4: Company financials
    example_company_financials(data_dir)

    # Example 5: Tag search
    example_tag_search(data_dir)

    # Example 6: Specific metric analysis
    example_specific_metric_analysis(data_dir)

    # Example 7: Form analysis
    example_form_analysis(data_dir)

    print("\n" + "="*70)
    print("EXAMPLES COMPLETED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Modify the USER_AGENT with your email")
    print("  2. Explore the parser methods in sec_data_parser.py")
    print("  3. Read the documentation in README_SEC_DATA.md")
    print("  4. Start analyzing companies and financial data!")
    print("\n")


if __name__ == "__main__":
    # You can run individual examples or all of them
    main()

    # Or run specific examples:
    # data_dir = Path("./sec_data/2024q4")  # Use existing downloaded data
    # example_company_financials(data_dir)

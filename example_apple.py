"""
Example: Download Apple Inc financial data

This shows exactly how to get financial data for Apple Inc (the real one).
"""

from sec_data_parser import SECDataParser
import pandas as pd

# Apple Inc's official CIK number
APPLE_CIK = "320193"

# Initialize parser
parser = SECDataParser("./sec_data/2024q4")

print("="*70)
print("DOWNLOADING APPLE INC FINANCIAL DATA")
print("="*70)

# Method 1: Get by exact CIK (most reliable)
print(f"\n1. Getting Apple Inc data by CIK ({APPLE_CIK})...")
submissions = parser.get_company_submissions(APPLE_CIK)

if not submissions.empty:
    print(f"\n[OK] Company: {submissions.iloc[0]['name']}")
    print(f"[OK] CIK: {int(submissions.iloc[0]['cik']):010d}")
    print(f"\nSubmissions in Q4 2024:")
    print(submissions[['form', 'period', 'filed']].to_string())

    # Get all financial data
    print("\n2. Getting all financial data...")
    financials = parser.get_company_financials(APPLE_CIK)

    print(f"\n[OK] Found {len(financials):,} financial data points")
    print(f"[OK] Unique metrics: {financials['tag'].nunique()}")

    # Show sample
    print("\n3. Sample financial data:")
    sample = financials[['tag', 'value', 'uom', 'period']].head(15)
    print(sample.to_string(index=False))

    # Get key metrics
    print("\n4. Getting key financial metrics...")
    key_metrics = [
        'Assets',
        'AssetsCurrent',
        'Liabilities',
        'StockholdersEquity',
        'Revenues',
        'NetIncomeLoss',
        'EarningsPerShareBasic',
        'EarningsPerShareDiluted',
        'CashAndCashEquivalentsAtCarryingValue'
    ]

    key_data = financials[financials['tag'].isin(key_metrics)]

    if not key_data.empty:
        print(f"\n[OK] Found {len(key_data)} key metrics:\n")

        # Group by tag and show values
        for tag in key_metrics:
            tag_data = key_data[key_data['tag'] == tag]
            if not tag_data.empty:
                value = tag_data.iloc[0]['value']
                uom = tag_data.iloc[0]['uom']
                period = tag_data.iloc[0]['period']

                # Format large numbers
                if value > 1e9:
                    formatted = f"${value/1e9:.2f}B"
                elif value > 1e6:
                    formatted = f"${value/1e6:.2f}M"
                else:
                    formatted = f"${value:,.2f}"

                print(f"   {tag:40s} {formatted:>15s} ({period})")

    # Export to CSV
    print("\n5. Exporting data...")
    output_file = "Apple_Inc_financials_2024q4.csv"
    financials.to_csv(output_file, index=False)
    print(f"[SAVED] Exported to: {output_file}")

    # Export key metrics only
    key_file = "Apple_Inc_key_metrics_2024q4.csv"
    key_data.to_csv(key_file, index=False)
    print(f"[SAVED] Key metrics exported to: {key_file}")

    print(f"\n{'='*70}")
    print("SUCCESS!")
    print(f"{'='*70}")
    print("\nFiles created:")
    print(f"  - {output_file} - All financial data")
    print(f"  - {key_file} - Key metrics only")

else:
    print("[ERROR] No data found for Apple Inc in this quarter")

print("\n" + "="*70)
print("MORE EXAMPLES:")
print("="*70)
print("""
# Get other companies by CIK:
Microsoft:  '789019'
Tesla:      '1318605'
Amazon:     '1018724'
Google:     '1652044'
Meta:       '1326801'

# Usage:
parser.get_company_submissions('789019')  # Microsoft
parser.get_company_financials('1318605')  # Tesla
""")

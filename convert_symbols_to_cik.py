"""
Convert stock ticker symbols to CIK numbers using SEC data
"""

import pandas as pd
import requests
import json
from pathlib import Path

print("="*70)
print("CONVERTING TICKER SYMBOLS TO CIK NUMBERS")
print("="*70)

# Read the input file
input_file = "CIK oznake.csv"
output_file = "CIK_oznake_with_cik.csv"

print(f"\nReading: {input_file}")
df = pd.read_csv(input_file)
print(f"Found {len(df)} companies")

# Download SEC company tickers mapping
print("\nDownloading SEC company tickers mapping...")
url = "https://www.sec.gov/files/company_tickers.json"

headers = {
    'User-Agent': 'YourName your@email.com'  # SEC requires user agent
}

try:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    sec_data = response.json()
    print(f"[OK] Downloaded {len(sec_data)} company mappings from SEC")
except Exception as e:
    print(f"[ERROR] Failed to download SEC data: {e}")
    exit(1)

# Create a mapping dictionary: ticker -> CIK
ticker_to_cik = {}
ticker_to_name = {}

for key, company in sec_data.items():
    ticker = company['ticker'].upper()
    cik = str(company['cik_str']).zfill(10)  # Pad with zeros to 10 digits
    name = company['title']

    ticker_to_cik[ticker] = cik
    ticker_to_name[ticker] = name

print(f"[OK] Created mapping for {len(ticker_to_cik)} tickers")

# Convert symbols to CIK
print("\nConverting symbols to CIK numbers...")
results = []
not_found = []

for idx, row in df.iterrows():
    symbol = str(row['Symbol']).upper().strip()
    security_name = row['Security']

    # Handle multiple symbols (e.g., "CWEN, CWEN.A")
    symbols = [s.strip() for s in symbol.split(',')]
    primary_symbol = symbols[0]

    if primary_symbol in ticker_to_cik:
        cik = ticker_to_cik[primary_symbol]
        company_name = ticker_to_name[primary_symbol]

        results.append({
            'cik': cik,
            'company_name': company_name,
            'symbol': primary_symbol,
            'original_security': security_name
        })
        print(f"  {idx+1:3d}. {primary_symbol:6s} -> CIK {cik} ({company_name})")
    else:
        not_found.append({
            'rank': row['Rank'],
            'symbol': primary_symbol,
            'security': security_name
        })
        print(f"  {idx+1:3d}. {primary_symbol:6s} -> NOT FOUND")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
print(f"\n{'='*70}")
print("SAVING RESULTS")
print("="*70)

results_df.to_csv(output_file, index=False)
print(f"\n[OK] Saved {len(results_df)} companies with CIK to: {output_file}")

if not_found:
    print(f"\n[WARNING] {len(not_found)} symbols not found:")
    for item in not_found[:10]:  # Show first 10
        print(f"  - {item['symbol']} ({item['security']})")
    if len(not_found) > 10:
        print(f"  ... and {len(not_found) - 10} more")

    # Save not found to separate file
    not_found_df = pd.DataFrame(not_found)
    not_found_file = "symbols_not_found.csv"
    not_found_df.to_csv(not_found_file, index=False)
    print(f"\n[INFO] Not found symbols saved to: {not_found_file}")

print(f"\n{'='*70}")
print("CONVERSION COMPLETE")
print("="*70)
print(f"[OK] Success: {len(results_df)} companies")
print(f"[!] Not found: {len(not_found)} companies")
print(f"\nNext step: Run batch_benford_analysis.py with {output_file}")

"""
Debug script to check CIK format in SEC data
"""

from sec_data_parser import SECDataParser
from pathlib import Path

# Load one quarter as example
quarter_dir = Path("./sec_data/2024q4")

if quarter_dir.exists():
    print("Loading 2024q4 data...")
    parser = SECDataParser(quarter_dir)

    print("\nSubmissions DataFrame info:")
    print(parser.submissions.head())
    print("\nCIK column dtype:", parser.submissions['cik'].dtype)
    print("\nSample CIKs from submissions:")
    print(parser.submissions['cik'].head(20).tolist())

    # Check if Apple (320193) is in there
    apple_cik_int = 320193
    apple_cik_str = "320193"
    apple_cik_padded = "0000320193"

    print(f"\nLooking for Apple (CIK {apple_cik_int})...")
    print(f"  As int {apple_cik_int}: {apple_cik_int in parser.submissions['cik'].values}")
    print(f"  As str '{apple_cik_str}': {apple_cik_str in parser.submissions['cik'].astype(str).values}")
    print(f"  As padded '{apple_cik_padded}': {apple_cik_padded in parser.submissions['cik'].astype(str).str.zfill(10).values}")

    # Try getting Apple data
    print("\nTrying to get Apple financials with different CIK formats...")
    for cik_format in [apple_cik_int, apple_cik_str, apple_cik_padded]:
        try:
            data = parser.get_company_financials(cik_format)
            print(f"  CIK {cik_format}: Found {len(data)} records")
        except Exception as e:
            print(f"  CIK {cik_format}: Error - {e}")
else:
    print("2024q4 directory not found!")

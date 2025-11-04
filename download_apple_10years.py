"""
Download 10 years of Apple financial data and extract numbers for Benford's Law analysis
"""

from sec_data_downloader import SECDataDownloader
from sec_data_parser import SECDataParser
import pandas as pd
from pathlib import Path
import time

# Apple's CIK
APPLE_CIK = "320193"

# Years to download (2014-2024 = 10 years)
START_YEAR = 2014
END_YEAR = 2024

print("="*70)
print("DOWNLOADING 10 YEARS OF APPLE FINANCIAL DATA")
print("="*70)
print(f"\nYears: {START_YEAR} - {END_YEAR}")
print(f"Company: Apple Inc (CIK: {APPLE_CIK})")
print("\nThis will download ~40 quarterly files (~20-30 GB total)")

# Step 1: Download all quarters
print("\n" + "="*70)
print("STEP 1: Downloading SEC Data")
print("="*70)

downloader = SECDataDownloader(
    user_agent="Josip Sare josip.sare@gmail.com",
    output_dir="./sec_data"
)

# Download range
print(f"\nDownloading {END_YEAR - START_YEAR + 1} years of data...")
downloaded_dirs = downloader.download_range(START_YEAR, END_YEAR, extract=True, keep_zip=False)

print(f"\n[OK] Downloaded {len(downloaded_dirs)} quarters")

# Step 2: Extract Apple data from all quarters
print("\n" + "="*70)
print("STEP 2: Extracting Apple Data")
print("="*70)

all_apple_numbers = []
all_apple_data = []

for quarter_dir in downloaded_dirs:
    print(f"\nProcessing: {quarter_dir.name}...")

    try:
        parser = SECDataParser(quarter_dir)

        # Get Apple's data for this quarter
        financials = parser.get_company_financials(APPLE_CIK)

        if not financials.empty:
            print(f"  Found {len(financials):,} data points")

            # Extract numerical values only (non-null, non-zero)
            numbers = financials['value'].dropna()
            numbers = numbers[numbers != 0]

            all_apple_numbers.extend(numbers.tolist())

            # Also keep full data for reference
            financials['quarter'] = quarter_dir.name
            all_apple_data.append(financials)
        else:
            print(f"  No Apple data in this quarter")

    except Exception as e:
        print(f"  Error: {e}")
        continue

# Step 3: Create datasets
print("\n" + "="*70)
print("STEP 3: Creating Datasets")
print("="*70)

# All numbers for Benford's Law
print(f"\nTotal numbers collected: {len(all_apple_numbers):,}")

# Save numbers only (for Benford's Law)
numbers_df = pd.DataFrame({
    'value': all_apple_numbers
})

# Remove negative numbers for Benford's Law (use absolute values)
numbers_df['abs_value'] = numbers_df['value'].abs()
numbers_df = numbers_df[numbers_df['abs_value'] > 0]

print(f"Numbers after filtering: {len(numbers_df):,}")

# Save
output_file = "apple_numbers_2014_2024.csv"
numbers_df.to_csv(output_file, index=False)
print(f"\n[SAVED] Numbers dataset: {output_file}")

# Save full data
if all_apple_data:
    full_data = pd.concat(all_apple_data, ignore_index=True)
    full_output = "apple_full_data_2014_2024.csv"
    full_data.to_csv(full_output, index=False)
    print(f"[SAVED] Full dataset: {full_output}")

    print(f"\nFull dataset stats:")
    print(f"  Total rows: {len(full_data):,}")
    print(f"  Date range: {full_data['period'].min()} to {full_data['period'].max()}")
    print(f"  Unique tags: {full_data['tag'].nunique()}")

# Step 4: Benford's Law Analysis
print("\n" + "="*70)
print("STEP 4: Benford's Law Analysis")
print("="*70)

# Extract first digit
def get_first_digit(number):
    """Extract first significant digit from a number"""
    if pd.isna(number) or number == 0:
        return None

    # Use absolute value
    num_str = str(abs(number))

    # Remove decimal point and find first non-zero digit
    num_str = num_str.replace('.', '').replace('-', '')

    for digit in num_str:
        if digit != '0':
            return int(digit)

    return None

numbers_df['first_digit'] = numbers_df['abs_value'].apply(get_first_digit)
numbers_df = numbers_df.dropna(subset=['first_digit'])

# Count frequency of first digits
digit_counts = numbers_df['first_digit'].value_counts().sort_index()
digit_freq = digit_counts / len(numbers_df) * 100

# Benford's Law expected distribution
benford_expected = {
    1: 30.1,
    2: 17.6,
    3: 12.5,
    4: 9.7,
    5: 7.9,
    6: 6.7,
    7: 5.8,
    8: 5.1,
    9: 4.5
}

# Create comparison
print("\nFirst Digit Distribution vs Benford's Law:")
print("\nDigit  | Actual % | Benford % | Difference")
print("-"*50)

comparison_data = []
for digit in range(1, 10):
    actual = digit_freq.get(digit, 0)
    expected = benford_expected[digit]
    diff = actual - expected

    print(f"  {digit}    |  {actual:6.2f}% |   {expected:5.1f}%  | {diff:+6.2f}%")

    comparison_data.append({
        'digit': digit,
        'actual_percent': actual,
        'benford_percent': expected,
        'difference': diff,
        'count': digit_counts.get(digit, 0)
    })

# Save Benford analysis
benford_df = pd.DataFrame(comparison_data)
benford_output = "apple_benford_analysis_2014_2024.csv"
benford_df.to_csv(benford_output, index=False)
print(f"\n[SAVED] Benford analysis: {benford_output}")

# Save numbers with first digits
numbers_with_digits = "apple_numbers_with_first_digit.csv"
numbers_df.to_csv(numbers_with_digits, index=False)
print(f"[SAVED] Numbers with first digits: {numbers_with_digits}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nFiles created:")
print(f"  1. {output_file} - All numbers ({len(numbers_df):,} values)")
print(f"  2. {full_output} - Full financial data")
print(f"  3. {benford_output} - Benford's Law analysis")
print(f"  4. {numbers_with_digits} - Numbers with first digit extracted")
print("\nYou can now perform further Benford's Law analysis on these datasets!")

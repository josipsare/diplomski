"""
Process existing Apple data from already-downloaded SEC quarters
Use this if you want to analyze data without downloading everything
"""

from sec_data_parser import SECDataParser
import pandas as pd
from pathlib import Path
import numpy as np

# Apple's CIK
APPLE_CIK = "320193"

print("="*70)
print("PROCESSING EXISTING APPLE DATA FOR BENFORD'S LAW")
print("="*70)

# Find all downloaded quarters
sec_data_dir = Path("./sec_data")
quarter_dirs = sorted([d for d in sec_data_dir.glob("20*q*") if d.is_dir()])

print(f"\nFound {len(quarter_dirs)} downloaded quarters:")
for qdir in quarter_dirs:
    print(f"  - {qdir.name}")

if not quarter_dirs:
    print("\nNo data found! Run the downloader first.")
    exit(1)

# Extract Apple data from all quarters
print("\n" + "="*70)
print("EXTRACTING APPLE NUMBERS")
print("="*70)

all_numbers = []
all_data = []
quarters_processed = []

for quarter_dir in quarter_dirs:
    print(f"\nProcessing: {quarter_dir.name}...")

    try:
        parser = SECDataParser(quarter_dir)

        # Get Apple's data
        financials = parser.get_company_financials(APPLE_CIK)

        if not financials.empty:
            print(f"  Found {len(financials):,} data points")

            # Extract all numerical values
            numbers = financials['value'].dropna()
            numbers = numbers[numbers != 0]

            all_numbers.extend(numbers.tolist())
            quarters_processed.append(quarter_dir.name)

            # Keep full data
            financials['quarter'] = quarter_dir.name
            all_data.append(financials)
        else:
            print(f"  No Apple data")

    except Exception as e:
        print(f"  Error: {e}")

print(f"\n[OK] Processed {len(quarters_processed)} quarters")
print(f"[OK] Collected {len(all_numbers):,} numbers")

# Create dataset
print("\n" + "="*70)
print("CREATING BENFORD'S LAW DATASET")
print("="*70)

# Numbers dataframe
numbers_df = pd.DataFrame({'value': all_numbers})

# Use absolute values for Benford's Law
numbers_df['abs_value'] = numbers_df['value'].abs()
numbers_df = numbers_df[numbers_df['abs_value'] > 0]

print(f"\nNumbers after filtering: {len(numbers_df):,}")

# Extract first digit
def get_first_digit(number):
    """Extract first significant digit"""
    if pd.isna(number) or number == 0:
        return None

    # Convert to string, remove decimal and signs
    num_str = f"{abs(number):.10e}"  # Use scientific notation to handle very large/small numbers

    # Extract first digit
    for char in str(int(abs(number))):
        if char != '0':
            return int(char)

    return None

numbers_df['first_digit'] = numbers_df['abs_value'].apply(get_first_digit)
numbers_df = numbers_df.dropna(subset=['first_digit'])

print(f"Numbers with valid first digit: {len(numbers_df):,}")

# Benford's Law Analysis
print("\n" + "="*70)
print("BENFORD'S LAW ANALYSIS")
print("="*70)

# Count frequency
digit_counts = numbers_df['first_digit'].value_counts().sort_index()
digit_freq = (digit_counts / len(numbers_df) * 100).round(2)

# Expected distribution
benford_expected = {
    1: 30.1, 2: 17.6, 3: 12.5, 4: 9.7, 5: 7.9,
    6: 6.7, 7: 5.8, 8: 5.1, 9: 4.5
}

# Display results
print(f"\nTotal numbers analyzed: {len(numbers_df):,}")
print(f"Quarters: {', '.join(quarters_processed[:5])}{'...' if len(quarters_processed) > 5 else ''}")
print(f"\nFirst Digit Distribution:\n")
print("Digit | Count      | Actual % | Benford % | Diff")
print("-"*60)

comparison = []
for digit in range(1, 10):
    count = digit_counts.get(digit, 0)
    actual = digit_freq.get(digit, 0.0)
    expected = benford_expected[digit]
    diff = actual - expected

    print(f"  {digit}   | {count:10,} | {actual:7.2f}% | {expected:8.1f}% | {diff:+6.2f}%")

    comparison.append({
        'digit': digit,
        'count': count,
        'actual_percent': actual,
        'benford_percent': expected,
        'difference': diff
    })

# Chi-square test
print("\n" + "-"*60)
observed = [digit_counts.get(i, 0) for i in range(1, 10)]
expected_counts = [len(numbers_df) * benford_expected[i] / 100 for i in range(1, 10)]
chi_square = sum((o - e)**2 / e for o, e in zip(observed, expected_counts))
print(f"Chi-Square Statistic: {chi_square:.4f}")
print(f"Critical value (8 df, α=0.05): 15.507")
print(f"Result: {'CONFORMS' if chi_square < 15.507 else 'DOES NOT CONFORM'} to Benford's Law")

# Save files
print("\n" + "="*70)
print("SAVING FILES")
print("="*70)

# 1. All numbers
output_numbers = "apple_numbers_for_benford.csv"
numbers_df.to_csv(output_numbers, index=False)
print(f"\n[SAVED] {output_numbers}")
print(f"        {len(numbers_df):,} numbers with first digit")

# 2. Benford analysis
benford_df = pd.DataFrame(comparison)
output_benford = "apple_benford_analysis.csv"
benford_df.to_csv(output_benford, index=False)
print(f"[SAVED] {output_benford}")

# 3. Full data
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    output_full = "apple_full_financial_data.csv"
    full_df.to_csv(output_full, index=False)
    print(f"[SAVED] {output_full}")
    print(f"        {len(full_df):,} total data points")

# 4. Summary stats
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nQuarters analyzed: {len(quarters_processed)}")
print(f"Total numbers: {len(numbers_df):,}")
print(f"Value range: ${numbers_df['abs_value'].min():,.0f} to ${numbers_df['abs_value'].max():,.0f}")
print(f"Median value: ${numbers_df['abs_value'].median():,.0f}")
print(f"\nBenford's Law Result: {'CONFORMS ✓' if chi_square < 15.507 else 'DOES NOT CONFORM ✗'}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)

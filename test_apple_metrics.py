"""
Test benford_metrics with Apple data to verify it matches existing calculations
"""

from sec_data_parser import SECDataParser
import pandas as pd
from pathlib import Path
from benford_metrics import calculate_benford_metrics

# Apple's CIK
APPLE_CIK = "320193"

print("="*70)
print("TESTING BENFORD METRICS WITH APPLE DATA")
print("="*70)

# Find all downloaded quarters
sec_data_dir = Path("./sec_data")
quarter_dirs = sorted([d for d in sec_data_dir.glob("20*q*") if d.is_dir()])

print(f"\nFound {len(quarter_dirs)} downloaded quarters")

# Extract Apple data from all quarters
all_numbers = []

for quarter_dir in quarter_dirs:
    try:
        parser = SECDataParser(quarter_dir)
        financials = parser.get_company_financials(APPLE_CIK)

        if not financials.empty:
            numbers = financials['value'].dropna()
            numbers = numbers[numbers != 0]
            all_numbers.extend(numbers.tolist())
    except Exception as e:
        pass

print(f"Collected {len(all_numbers):,} numbers from Apple")

# Calculate metrics using new function
numbers_series = pd.Series(all_numbers)
metrics = calculate_benford_metrics(numbers_series)

print("\n" + "="*70)
print("BENFORD'S LAW METRICS FOR APPLE (ALL YEARS)")
print("="*70)
print(f"\nTotal numbers analyzed: {metrics['n_samples']:,}")
print(f"\nChi-Square Statistic: {metrics['chi_square']:.4f}")
print(f"  Critical value (8 df, α=0.05): 15.507")
print(f"  Result: {'CONFORMS' if metrics['chi_square'] < 15.507 else 'DOES NOT CONFORM'}")

print(f"\nP-Value: {metrics['p_value']:.4f}")
print(f"  Significant deviation: {'YES' if metrics['p_value'] < 0.05 else 'NO'}")

print(f"\nMAD (Mean Absolute Deviation): {metrics['MAD']:.4f}")
if metrics['MAD'] < 0.006:
    level = "Close conformity"
elif metrics['MAD'] < 0.012:
    level = "Acceptable conformity"
elif metrics['MAD'] < 0.015:
    level = "Marginally acceptable"
else:
    level = "Nonconformity"
print(f"  Level: {level}")
print(f"  Industry standard: MAD < 0.015")

print(f"\nKolmogorov-Smirnov Test: {metrics['KS_test']:.4f}")
print(f"  (Lower values indicate better fit)")

print("\n" + "="*70)

# Now test for a specific year (2024)
print("\nTESTING FOR YEAR 2024 ONLY")
print("="*70)

year_2024_numbers = []
for quarter in range(1, 5):
    quarter_dir = sec_data_dir / f"2024q{quarter}"
    if quarter_dir.exists():
        try:
            parser = SECDataParser(quarter_dir)
            financials = parser.get_company_financials(APPLE_CIK)
            if not financials.empty:
                numbers = financials['value'].dropna()
                numbers = numbers[numbers != 0]
                year_2024_numbers.extend(numbers.tolist())
        except:
            pass

if year_2024_numbers:
    metrics_2024 = calculate_benford_metrics(pd.Series(year_2024_numbers))
    print(f"\n2024 Data: {metrics_2024['n_samples']:,} numbers")
    print(f"  Chi-Square: {metrics_2024['chi_square']:.4f}")
    print(f"  P-Value: {metrics_2024['p_value']:.4f}")
    print(f"  MAD: {metrics_2024['MAD']:.4f}")
    print(f"  KS Test: {metrics_2024['KS_test']:.4f}")
else:
    print("\nNo data found for 2024")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

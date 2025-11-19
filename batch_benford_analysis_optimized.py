"""
Optimized Batch Benford's Law Analysis for 500 Companies
Loads each quarter file only once, then extracts data for all companies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sec_data_parser import SECDataParser
from benford_metrics import calculate_benford_metrics
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Main batch processing function - optimized approach
    """
    print("="*70)
    print("BATCH BENFORD'S LAW ANALYSIS - OPTIMIZED")
    print("="*70)

    # Configuration
    cik_file = Path("CIK_oznake_with_cik.csv")
    sec_data_dir = Path("./sec_data")
    output_file = Path("benford_500_companies_analysis.csv")
    years = list(range(2014, 2025))  # 2014-2024

    # Check if CIK file exists
    if not cik_file.exists():
        print(f"\n[ERROR] File '{cik_file}' not found!")
        return

    # Check if SEC data directory exists
    if not sec_data_dir.exists():
        print(f"\n[ERROR] Directory '{sec_data_dir}' not found!")
        return

    # Load company list
    print(f"\nLoading companies from: {cik_file}")
    companies_df = pd.read_csv(cik_file)
    companies_df['cik'] = companies_df['cik'].astype(str).str.zfill(10)
    print(f"[OK] Loaded {len(companies_df)} companies")

    # Create CIK set for fast lookup
    cik_set = set(companies_df['cik'].values)
    cik_to_name = dict(zip(companies_df['cik'], companies_df['company_name']))

    # Find available quarters
    quarter_dirs = sorted([d for d in sec_data_dir.glob("20*q*") if d.is_dir()])
    print(f"\nFound {len(quarter_dirs)} downloaded quarters")

    # Initialize data structure: {cik: {year: [numbers]}}
    company_data = defaultdict(lambda: defaultdict(list))

    # Process each quarter once
    print("\n" + "="*70)
    print("LOADING DATA FROM QUARTERS")
    print("="*70)

    for quarter_dir in tqdm(quarter_dirs, desc="Processing quarters"):
        quarter_name = quarter_dir.name
        year = int(quarter_name[:4])

        if year not in years:
            continue

        try:
            parser = SECDataParser(quarter_dir)

            # Extract data for each company in our list
            for cik in cik_set:
                try:
                    # get_company_financials handles CIK conversion internally
                    financials = parser.get_company_financials(cik)

                    if not financials.empty:
                        # Extract numerical values
                        numbers = financials['value'].dropna()
                        numbers = numbers[numbers != 0]

                        if len(numbers) > 0:
                            company_data[cik][year].extend(numbers.tolist())

                except Exception:
                    # Skip errors for individual companies
                    pass

        except Exception as e:
            print(f"\n[WARNING] Error processing {quarter_name}: {e}")
            continue

    # Calculate metrics for each company-year
    print("\n" + "="*70)
    print("CALCULATING BENFORD'S LAW METRICS")
    print("="*70)

    results = []

    for cik in tqdm(companies_df['cik'], desc="Calculating metrics"):
        company_name = cik_to_name.get(cik, 'Unknown')

        result = {
            'cik': cik,
            'company_name': company_name
        }

        for year in years:
            numbers = company_data[cik][year]

            if len(numbers) > 0:
                metrics = calculate_benford_metrics(pd.Series(numbers))
            else:
                # No data for this year
                metrics = {
                    'chi_square': 0,
                    'p_value': 0,
                    'MAD': 0,
                    'KS_test': 0
                }

            result[f'year_{year}_chi_square'] = metrics['chi_square']
            result[f'year_{year}_p_value'] = metrics['p_value']
            result[f'year_{year}_MAD'] = metrics['MAD']
            result[f'year_{year}_KS_test'] = metrics['KS_test']

        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    print(f"\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    results_df.to_csv(output_file, index=False)
    print(f"\n[OK] Results saved to: {output_file}")
    print(f"[OK] Total companies processed: {len(results_df)}")
    print(f"[OK] Total columns: {len(results_df.columns)}")

    # Display summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Count companies with data per year
    print("\nCompanies with data per year:")
    for year in years:
        chi_col = f'year_{year}_chi_square'
        count_with_data = (results_df[chi_col] > 0).sum()
        print(f"  {year}: {count_with_data:3d} / {len(results_df)} companies")

    # Show sample of results
    print("\nFirst 5 companies (2024 metrics):")
    sample_cols = ['cik', 'company_name', 'year_2024_chi_square', 'year_2024_p_value', 'year_2024_MAD', 'year_2024_KS_test']
    available_cols = [col for col in sample_cols if col in results_df.columns]
    print(results_df[available_cols].head().to_string())

    # Summary statistics for 2024
    print("\n2024 Metrics Summary:")
    print(f"  Chi-square: mean={results_df['year_2024_chi_square'].mean():.2f}, std={results_df['year_2024_chi_square'].std():.2f}")
    print(f"  P-value: mean={results_df['year_2024_p_value'].mean():.4f}, std={results_df['year_2024_p_value'].std():.4f}")
    print(f"  MAD: mean={results_df['year_2024_MAD'].mean():.4f}, std={results_df['year_2024_MAD'].std():.4f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

"""
Batch Benford's Law Analysis for 500 Companies
Processes multiple companies and generates comprehensive statistical analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sec_data_parser import SECDataParser
from benford_metrics import calculate_benford_metrics
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_company_data_for_year(cik, year, sec_data_dir):
    """
    Extract all financial numbers for a company for a specific year

    Args:
        cik: Company CIK number (string)
        year: Year (int, e.g., 2014)
        sec_data_dir: Path to SEC data directory

    Returns:
        pandas.Series: All numerical values for the company in that year
    """
    all_numbers = []

    # Get all quarters for this year (Q1-Q4)
    for quarter in range(1, 5):
        quarter_name = f"{year}q{quarter}"
        quarter_dir = sec_data_dir / quarter_name

        if not quarter_dir.exists():
            continue

        try:
            parser = SECDataParser(quarter_dir)
            financials = parser.get_company_financials(cik)

            if not financials.empty:
                # Extract numerical values
                numbers = financials['value'].dropna()
                numbers = numbers[numbers != 0]
                all_numbers.extend(numbers.tolist())

        except Exception as e:
            # Silently skip errors for individual quarters
            pass

    return pd.Series(all_numbers) if all_numbers else pd.Series([])


def process_company(cik, company_name, sec_data_dir, years):
    """
    Process a single company across all years

    Args:
        cik: Company CIK (string)
        company_name: Company name (string)
        sec_data_dir: Path to SEC data directory
        years: List of years to analyze

    Returns:
        dict: Results with CIK, name, and metrics for each year
    """
    result = {
        'cik': cik,
        'company_name': company_name
    }

    for year in years:
        # Get data for this year
        numbers = get_company_data_for_year(cik, year, sec_data_dir)

        # Calculate metrics
        if len(numbers) > 0:
            metrics = calculate_benford_metrics(numbers)
        else:
            # No data for this year - set all to 0
            metrics = {
                'chi_square': 0,
                'p_value': 0,
                'MAD': 0,
                'KS_test': 0
            }

        # Add to results with year prefix
        result[f'year_{year}_chi_square'] = metrics['chi_square']
        result[f'year_{year}_p_value'] = metrics['p_value']
        result[f'year_{year}_MAD'] = metrics['MAD']
        result[f'year_{year}_KS_test'] = metrics['KS_test']

    return result


def main():
    """
    Main batch processing function
    """
    print("="*70)
    print("BATCH BENFORD'S LAW ANALYSIS - 500 COMPANIES")
    print("="*70)

    # Configuration
    cik_file = Path("CIK_oznake_with_cik.csv")
    sec_data_dir = Path("./sec_data")
    output_file = Path("benford_500_companies_analysis.csv")
    years = list(range(2014, 2025))  # 2014-2024

    # Check if CIK file exists
    if not cik_file.exists():
        print(f"\n[ERROR] File '{cik_file}' not found!")
        print("Please create a CSV file with columns: cik, company_name")
        return

    # Check if SEC data directory exists
    if not sec_data_dir.exists():
        print(f"\n[ERROR] Directory '{sec_data_dir}' not found!")
        print("Please run the SEC data downloader first.")
        return

    # Load company list
    print(f"\nLoading companies from: {cik_file}")
    try:
        companies_df = pd.read_csv(cik_file)

        # Check for required columns
        if 'cik' not in companies_df.columns:
            print("[ERROR] CSV must have a 'cik' column!")
            return

        # Add company_name column if it doesn't exist
        if 'company_name' not in companies_df.columns:
            companies_df['company_name'] = 'Unknown'

        # Convert CIK to string and pad with zeros if needed
        companies_df['cik'] = companies_df['cik'].astype(str).str.zfill(10)

        print(f"[OK] Loaded {len(companies_df)} companies")

    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    # Find available quarters
    quarter_dirs = sorted([d for d in sec_data_dir.glob("20*q*") if d.is_dir()])
    print(f"\nFound {len(quarter_dirs)} downloaded quarters")
    print(f"Year range: {years[0]} - {years[-1]}")

    # Process each company
    print("\n" + "="*70)
    print("PROCESSING COMPANIES")
    print("="*70)

    results = []

    # Use tqdm for progress bar
    for _, row in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Processing"):
        cik = row['cik']
        company_name = row.get('company_name', 'Unknown')

        try:
            result = process_company(cik, company_name, sec_data_dir, years)
            results.append(result)
        except Exception as e:
            # If processing fails, add row with zeros
            print(f"\n[WARNING] Error processing CIK {cik}: {e}")
            result = {'cik': cik, 'company_name': company_name}
            for year in years:
                result[f'year_{year}_chi_square'] = 0
                result[f'year_{year}_p_value'] = 0
                result[f'year_{year}_MAD'] = 0
                result[f'year_{year}_KS_test'] = 0
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
    print(f"     - 2 metadata columns (cik, company_name)")
    print(f"     - {len(years) * 4} metric columns ({len(years)} years × 4 metrics)")

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
    print("\nFirst 5 companies (sample columns):")
    sample_cols = ['cik', 'company_name', 'year_2024_chi_square', 'year_2024_p_value', 'year_2024_MAD']
    available_cols = [col for col in sample_cols if col in results_df.columns]
    print(results_df[available_cols].head().to_string())

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

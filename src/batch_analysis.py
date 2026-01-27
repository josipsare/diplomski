"""
Batch Benford's Law Analysis for Multiple Companies

Processes multiple companies and generates comprehensive statistical analysis
for both first-digit and second-digit Benford's Law conformance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm
import warnings

from .parser import SECDataParser
from .benford import calculate_benford_metrics, calculate_second_digit_benford_metrics

warnings.filterwarnings('ignore')


def get_company_data_for_year(
    cik: str,
    year: int,
    sec_data_dir: Path
) -> pd.Series:
    """
    Extract all financial numbers for a company for a specific year.

    Args:
        cik: Company CIK number (string, zero-padded)
        year: Year (int, e.g., 2014)
        sec_data_dir: Path to SEC data directory

    Returns:
        pandas.Series: All numerical values for the company in that year
    """
    all_numbers = []

    for quarter in range(1, 5):
        quarter_name = f"{year}q{quarter}"
        quarter_dir = sec_data_dir / quarter_name

        if not quarter_dir.exists():
            continue

        try:
            parser = SECDataParser(quarter_dir)
            financials = parser.get_company_financials(cik)

            if not financials.empty:
                numbers = financials['value'].dropna()
                numbers = numbers[numbers != 0]
                all_numbers.extend(numbers.tolist())

        except Exception:
            pass

    return pd.Series(all_numbers) if all_numbers else pd.Series([], dtype=float)


def process_company(
    cik: str,
    company_name: str,
    sec_data_dir: Path,
    years: List[int]
) -> dict:
    """
    Process a single company across all years.

    Calculates both first-digit and second-digit Benford metrics.

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
        numbers = get_company_data_for_year(cik, year, sec_data_dir)

        if len(numbers) > 0:
            first_metrics = calculate_benford_metrics(numbers)
            second_metrics = calculate_second_digit_benford_metrics(numbers)
        else:
            first_metrics = {'chi_square': 0, 'p_value': 0, 'MAD': 0, 'KS_test': 0}
            second_metrics = {'chi_square': 0, 'p_value': 0, 'MAD': 0, 'KS_test': 0}

        # First-digit results
        result[f'year_{year}_chi_square'] = first_metrics['chi_square']
        result[f'year_{year}_p_value'] = first_metrics['p_value']
        result[f'year_{year}_MAD'] = first_metrics['MAD']
        result[f'year_{year}_KS_test'] = first_metrics['KS_test']

        # Second-digit results
        result[f'year_{year}_d2_chi_square'] = second_metrics['chi_square']
        result[f'year_{year}_d2_p_value'] = second_metrics['p_value']
        result[f'year_{year}_d2_MAD'] = second_metrics['MAD']
        result[f'year_{year}_d2_KS_test'] = second_metrics['KS_test']

    return result


def run_batch_analysis(
    cik_file: Union[str, Path],
    sec_data_dir: Union[str, Path],
    output_file: Union[str, Path],
    years: Optional[List[int]] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Run batch Benford analysis on multiple companies.

    Args:
        cik_file: Path to CSV file with CIK numbers
        sec_data_dir: Path to SEC data directory
        output_file: Path to save results CSV
        years: List of years to analyze (default: 2014-2024)
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with analysis results
    """
    cik_file = Path(cik_file)
    sec_data_dir = Path(sec_data_dir)
    output_file = Path(output_file)

    if years is None:
        years = list(range(2014, 2025))

    # Validate inputs
    if not cik_file.exists():
        raise FileNotFoundError(f"CIK file not found: {cik_file}")

    if not sec_data_dir.exists():
        raise FileNotFoundError(f"SEC data directory not found: {sec_data_dir}")

    # Load company list
    companies_df = pd.read_csv(cik_file)

    if 'cik' not in companies_df.columns:
        raise ValueError("CSV must have a 'cik' column")

    if 'company_name' not in companies_df.columns:
        companies_df['company_name'] = 'Unknown'

    companies_df['cik'] = companies_df['cik'].astype(str).str.zfill(10)

    # Process companies
    results = []
    iterator = tqdm(companies_df.iterrows(), total=len(companies_df), desc="Processing") \
        if show_progress else companies_df.iterrows()

    for _, row in iterator:
        cik = row['cik']
        company_name = row.get('company_name', 'Unknown')

        try:
            result = process_company(cik, company_name, sec_data_dir, years)
            results.append(result)
        except Exception as e:
            result = {'cik': cik, 'company_name': company_name}
            for year in years:
                for col in ['chi_square', 'p_value', 'MAD', 'KS_test']:
                    result[f'year_{year}_{col}'] = 0
                    result[f'year_{year}_d2_{col}'] = 0
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    return results_df


if __name__ == '__main__':
    print("=" * 70)
    print("BATCH BENFORD'S LAW ANALYSIS")
    print("=" * 70)

    results = run_batch_analysis(
        cik_file="data/input/companies.csv",
        sec_data_dir="sec_data",
        output_file="data/output/results/benford_analysis.csv"
    )

    print(f"\nProcessed {len(results)} companies")
    print("Analysis complete!")

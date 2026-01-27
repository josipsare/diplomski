#!/usr/bin/env python3
"""
Run Sector/Industry Comparison Analysis

Extends Benford analysis with sector-based peer comparison using SIC codes.
Compares companies against their industry peers rather than just Benford's
expected distribution.

Usage:
    python scripts/run_sector_analysis.py --companies data/input/companies.csv
    python scripts/run_sector_analysis.py --help
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.parser import SECDataParser
from src.sector import (
    classify_company,
    calculate_sector_baselines,
    run_sector_comparison,
    identify_sector_outliers,
    get_sector_summary
)
from src.visualization import generate_sector_visualizations


def extract_sic_codes_from_sec_data(
    sec_data_dir: Path,
    ciks: List[str],
    years: List[int],
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract SIC codes for companies from SEC quarterly data.

    Args:
        sec_data_dir: Path to SEC data directory
        ciks: List of CIKs to extract
        years: Years to search
        show_progress: Show progress bar

    Returns:
        DataFrame with company SIC classifications
    """
    sector_data = []
    seen_ciks = set()

    # Build list of quarters to check
    quarters = []
    for year in years:
        for q in range(1, 5):
            quarter_dir = sec_data_dir / f"{year}q{q}"
            if quarter_dir.exists():
                quarters.append((year, q, quarter_dir))

    # Process quarters (most recent first to get latest SIC codes)
    quarters.reverse()

    iterator = tqdm(quarters, desc="Extracting SIC codes") if show_progress else quarters

    for year, q, quarter_dir in iterator:
        try:
            parser = SECDataParser(quarter_dir)
            sic_df = parser.get_company_sic_codes(ciks)

            for _, row in sic_df.iterrows():
                cik = row['cik']
                if cik not in seen_ciks and row['sic_code']:
                    classification = classify_company(
                        cik, row['company_name'], row['sic_code']
                    )
                    sector_data.append(classification)
                    seen_ciks.add(cik)

            # Stop if we have all companies
            if len(seen_ciks) >= len(ciks):
                break

        except Exception as e:
            if show_progress:
                tqdm.write(f"  Warning: Error processing {year}q{q}: {e}")

    return pd.DataFrame(sector_data)


def main():
    parser = argparse.ArgumentParser(
        description="Run Benford analysis with sector/industry comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sector analysis with existing Benford results
    python scripts/run_sector_analysis.py --benford-file data/output/results/benford_analysis.csv

    # Run full analysis from companies list
    python scripts/run_sector_analysis.py --companies data/input/companies.csv --sec-data ./sec_data
        """
    )

    parser.add_argument(
        "--benford-file",
        default="data/output/results/benford_500_companies_analysis.csv",
        help="Path to existing Benford analysis CSV file"
    )
    parser.add_argument(
        "--companies",
        default="data/input/companies.csv",
        help="Path to companies CSV file (with CIK column)"
    )
    parser.add_argument(
        "--sec-data",
        default="./sec_data",
        help="Path to SEC data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--graphs-dir",
        default="data/output/graphs/sector",
        help="Directory to save sector visualizations"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2014,
        help="Start year for analysis (default: 2014)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year for analysis (default: 2024)"
    )
    parser.add_argument(
        "--min-companies",
        type=int,
        default=3,
        help="Minimum companies per sector for baseline calculation (default: 3)"
    )
    parser.add_argument(
        "--group-by",
        choices=['sic_division', 'sic_major_group', 'sic_industry_group'],
        default='sic_major_group',
        help="Sector grouping level (default: sic_major_group)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualization graphs"
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    graphs_dir = Path(args.graphs_dir)
    sec_data_dir = Path(args.sec_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))
    show_progress = not args.quiet

    print("=" * 70)
    print("SECTOR/INDUSTRY COMPARISON ANALYSIS")
    print("=" * 70)
    print(f"\nAnalysis period: {args.start_year} - {args.end_year}")
    print(f"Grouping level: {args.group_by}")
    print(f"Minimum companies per sector: {args.min_companies}")

    # Load Benford analysis data
    print(f"\nLoading Benford analysis data from: {args.benford_file}")
    try:
        analysis_df = pd.read_csv(args.benford_file, dtype={'cik': str})
        analysis_df['cik'] = analysis_df['cik'].astype(str).str.zfill(10)
        print(f"  Loaded {len(analysis_df)} companies")
    except FileNotFoundError:
        print(f"  ERROR: Benford analysis file not found: {args.benford_file}")
        print("  Please run batch analysis first with: python scripts/run_analysis.py")
        sys.exit(1)

    # Get list of CIKs
    ciks = analysis_df['cik'].unique().tolist()

    # Extract SIC codes from SEC data
    print(f"\nExtracting SIC codes from SEC data...")
    sector_df = extract_sic_codes_from_sec_data(
        sec_data_dir, ciks, years, show_progress
    )

    if sector_df.empty:
        print("  ERROR: Could not extract any SIC codes from SEC data")
        print("  Make sure SEC data is downloaded in:", sec_data_dir)
        sys.exit(1)

    print(f"  Classified {len(sector_df)} companies by sector")

    # Show sector distribution
    sector_counts = sector_df['major_group_name'].value_counts()
    print(f"\nTop 10 sectors by company count:")
    for sector, count in sector_counts.head(10).items():
        print(f"  {sector}: {count} companies")

    # Run sector comparison analysis
    print(f"\nCalculating sector baselines and comparisons...")
    baselines_df, comparisons_df = run_sector_comparison(
        analysis_df,
        sector_df,
        years,
        group_by=args.group_by,
        min_companies=args.min_companies
    )

    if baselines_df.empty:
        print("  ERROR: Could not calculate sector baselines")
        print("  Check that SEC data contains SIC codes and analysis data is valid")
        sys.exit(1)

    print(f"  Calculated baselines for {baselines_df['sector_name'].nunique()} sectors")
    print(f"  Generated {len(comparisons_df)} company-sector comparisons")

    # Identify outliers
    outliers_df = identify_sector_outliers(comparisons_df, z_threshold=2.0)
    extreme_outliers_df = identify_sector_outliers(comparisons_df, z_threshold=3.0)

    # Save results
    print(f"\nSaving results to: {output_dir}")

    # Save sector classifications
    sector_output = output_dir / "sector_classifications.csv"
    sector_df.to_csv(sector_output, index=False)
    print(f"  Saved: sector_classifications.csv ({len(sector_df)} companies)")

    # Save sector baselines
    baselines_output = output_dir / "sector_baselines.csv"
    baselines_df.to_csv(baselines_output, index=False)
    print(f"  Saved: sector_baselines.csv ({len(baselines_df)} sector-year records)")

    # Save company comparisons
    comparisons_output = output_dir / "sector_comparisons.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"  Saved: sector_comparisons.csv ({len(comparisons_df)} comparisons)")

    # Save outliers
    if not outliers_df.empty:
        outliers_output = output_dir / "sector_outliers.csv"
        outliers_df.to_csv(outliers_output, index=False)
        print(f"  Saved: sector_outliers.csv ({len(outliers_df)} outliers)")

    # Get and save sector summary
    summary_df = get_sector_summary(baselines_df)
    if not summary_df.empty:
        summary_output = output_dir / "sector_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"  Saved: sector_summary.csv ({len(summary_df)} sectors)")

    # Generate visualizations
    if not args.skip_visualizations:
        print(f"\nGenerating sector visualizations in: {graphs_dir}")
        graphs_dir.mkdir(parents=True, exist_ok=True)
        generate_sector_visualizations(comparisons_df, baselines_df, graphs_dir)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nCompanies analyzed: {len(analysis_df)}")
    print(f"Companies with SIC codes: {len(sector_df)}")
    print(f"Unique sectors: {baselines_df['sector_name'].nunique()}")
    print(f"Years covered: {len(years)} ({args.start_year}-{args.end_year})")

    print(f"\nOutlier Detection (companies deviating from sector peers):")
    print(f"  Outliers (|z| > 2): {len(outliers_df)} observations")
    print(f"  Extreme outliers (|z| > 3): {len(extreme_outliers_df)} observations")

    if not outliers_df.empty:
        # Get unique companies that are outliers
        unique_outlier_companies = outliers_df['company_name'].nunique()
        print(f"  Unique companies flagged: {unique_outlier_companies}")

        print(f"\nTop 5 most suspicious companies (by max z-score):")
        top_outliers = outliers_df.groupby('company_name')['z_score_vs_sector'].max().nlargest(5)
        for company, z in top_outliers.items():
            sector = outliers_df[outliers_df['company_name'] == company]['sector_name'].iloc[0]
            print(f"  {company[:40]}: z={z:.2f} ({sector})")

    print(f"\nSector Performance Summary:")
    if not summary_df.empty:
        print("\n  Best conforming sectors (lowest avg median MAD):")
        for _, row in summary_df.nsmallest(5, 'avg_median_MAD').iterrows():
            print(f"    {row['sector_name'][:40]}: MAD={row['avg_median_MAD']:.3f}")

        print("\n  Worst conforming sectors (highest avg median MAD):")
        for _, row in summary_df.nlargest(5, 'avg_median_MAD').iterrows():
            print(f"    {row['sector_name'][:40]}: MAD={row['avg_median_MAD']:.3f}")

    print("\n" + "=" * 70)
    print("Sector analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

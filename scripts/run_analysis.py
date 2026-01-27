#!/usr/bin/env python3
"""
Run Benford's Law Batch Analysis

This script processes multiple companies and generates Benford's Law
conformance metrics for both first-digit and second-digit analysis.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch_analysis import run_batch_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run Benford's Law batch analysis on SEC financial data"
    )
    parser.add_argument(
        "--companies",
        default="./data/input/companies.csv",
        help="Path to CSV file with company CIKs (default: ./data/input/companies.csv)"
    )
    parser.add_argument(
        "--sec-data",
        default="./sec_data",
        help="Path to SEC data directory (default: ./sec_data)"
    )
    parser.add_argument(
        "--output",
        default="./data/output/results/benford_analysis.csv",
        help="Path to save results CSV"
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
        "--quiet",
        action="store_true",
        help="Suppress progress bar"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BENFORD'S LAW BATCH ANALYSIS")
    print("=" * 70)
    print(f"\nCompanies file: {args.companies}")
    print(f"SEC data dir:   {args.sec_data}")
    print(f"Output file:    {args.output}")
    print(f"Year range:     {args.start_year} - {args.end_year}")
    print()

    years = list(range(args.start_year, args.end_year + 1))

    results = run_batch_analysis(
        cik_file=args.companies,
        sec_data_dir=args.sec_data,
        output_file=args.output,
        years=years,
        show_progress=not args.quiet
    )

    print(f"\nProcessed {len(results)} companies")
    print(f"Results saved to: {args.output}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

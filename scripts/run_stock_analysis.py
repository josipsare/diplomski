#!/usr/bin/env python3
"""
Stock Analysis CLI Script

Downloads stock data, calculates metrics, merges with Benford analysis,
and generates visualizations.

Usage:
    python scripts/run_stock_analysis.py --download
    python scripts/run_stock_analysis.py --analyze
    python scripts/run_stock_analysis.py --visualize
    python scripts/run_stock_analysis.py --all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.stock_downloader import (
    StockDataDownloader,
    load_symbols_from_companies,
    get_cik_symbol_mapping
)
from src.stock_metrics import (
    calculate_all_metrics,
    merge_benford_and_stock_data,
    reshape_to_long_format,
    get_arrow_plot_data
)
from src.stock_visualization import (
    plot_benford_arrow_plot,
    plot_arrow_plot_by_year,
    plot_combined_time_series,
    plot_correlation_summary,
    generate_all_stock_visualizations
)


# Default paths
DEFAULT_COMPANIES_FILE = "./data/input/companies.csv"
DEFAULT_BENFORD_FILE = "./data/output/results/benford_500_companies_analysis.csv"
DEFAULT_STOCK_CACHE = "./data/stock_data"
DEFAULT_OUTPUT_DIR = "./data/output/results"
DEFAULT_GRAPHS_DIR = "./data/output/graphs/stock"

# Analysis years
START_YEAR = 2014
END_YEAR = 2024


def download_stock_data(
    companies_file: str,
    cache_dir: str,
    force_refresh: bool = False
) -> dict:
    """Download stock data for all companies."""
    print("\n" + "="*60)
    print("STEP 1: Downloading Stock Data")
    print("="*60 + "\n")

    # Load symbols
    symbols = load_symbols_from_companies(companies_file)
    print(f"Found {len(symbols)} symbols in {companies_file}")

    # Initialize downloader
    downloader = StockDataDownloader(cache_dir=cache_dir)

    # Download all stocks
    stock_data = downloader.download_all_stocks(
        symbols=symbols,
        start_year=START_YEAR,
        end_year=END_YEAR,
        force_refresh=force_refresh
    )

    return stock_data


def calculate_stock_metrics_batch(
    stock_data: dict,
    output_file: str
) -> pd.DataFrame:
    """Calculate stock metrics for all companies."""
    print("\n" + "="*60)
    print("STEP 2: Calculating Stock Metrics")
    print("="*60 + "\n")

    years = list(range(START_YEAR, END_YEAR + 1))

    # Calculate metrics
    metrics_df = calculate_all_metrics(stock_data, years)

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved stock metrics to: {output_file}")
    print(f"Shape: {metrics_df.shape}")

    return metrics_df


def merge_with_benford(
    stock_metrics_file: str,
    benford_file: str,
    companies_file: str,
    output_file: str
) -> pd.DataFrame:
    """Merge stock metrics with Benford analysis."""
    print("\n" + "="*60)
    print("STEP 3: Merging with Benford Analysis")
    print("="*60 + "\n")

    # Load data
    stock_df = pd.read_csv(stock_metrics_file)
    print(f"Stock metrics: {stock_df.shape}")

    if not Path(benford_file).exists():
        print(f"Warning: Benford file not found: {benford_file}")
        print("Skipping merge - will use stock data only")
        return stock_df

    benford_df = pd.read_csv(benford_file, dtype={'cik': str})
    print(f"Benford analysis: {benford_df.shape}")

    # Get CIK-symbol mapping
    cik_symbol_map = get_cik_symbol_mapping(companies_file)

    # Merge
    merged = merge_benford_and_stock_data(benford_df, stock_df, cik_symbol_map)
    print(f"Merged data: {merged.shape}")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"Saved merged data to: {output_file}")

    return merged


def generate_visualizations(
    combined_file: str,
    output_dir: str
):
    """Generate all stock visualizations."""
    print("\n" + "="*60)
    print("STEP 4: Generating Visualizations")
    print("="*60 + "\n")

    # Load data
    if not Path(combined_file).exists():
        print(f"Error: Combined data file not found: {combined_file}")
        return

    combined_df = pd.read_csv(combined_file, dtype={'cik': str})

    # Reshape to long format
    id_cols = ['cik', 'company_name', 'symbol']
    id_cols = [c for c in id_cols if c in combined_df.columns]

    long_df = reshape_to_long_format(combined_df, id_cols)
    print(f"Long format data: {long_df.shape}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare arrow plot data
    arrow_data = get_arrow_plot_data(
        long_df,
        benford_metric='MAD' if 'MAD' in long_df.columns else long_df.columns[3],
        stock_metric='stock_annual_return' if 'stock_annual_return' in long_df.columns else 'annual_return'
    )

    if len(arrow_data) > 0:
        # Graph 1: Main arrow plot
        print("\n1. Generating Benford Arrow Plot...")
        plot_benford_arrow_plot(
            arrow_data,
            output_path=str(output_path / "01_benford_arrow_plot.png")
        )

        # Graph 2: Arrow plot by year
        print("2. Generating Arrow Plot by Year...")
        years = sorted(arrow_data['year'].unique())
        # Select evenly spaced years
        if len(years) >= 4:
            selected_years = [years[i] for i in range(0, len(years), len(years)//4)][:4]
        else:
            selected_years = years
        plot_arrow_plot_by_year(
            arrow_data,
            years=selected_years,
            output_path=str(output_path / "02_arrow_plot_by_year.png")
        )

    # Graph 4: Time series for suspicious companies
    print("4. Generating Time Series for Suspicious Companies...")
    if 'MAD' in long_df.columns:
        avg_mad = long_df.groupby('symbol')['MAD'].mean()
        if len(avg_mad) > 0:
            top_suspicious = avg_mad.nlargest(6).index.tolist()
            plot_combined_time_series(
                long_df,
                symbols=top_suspicious,
                output_path=str(output_path / "04_time_series_suspicious.png")
            )

    # Graph 5: Correlation summary
    print("5. Generating Correlation Summary...")
    plot_correlation_summary(
        long_df,
        output_path=str(output_path / "05_correlation_summary.png")
    )

    print(f"\nAll visualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Analysis with Benford's Law Correlation"
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download stock data for all companies'
    )
    parser.add_argument(
        '--analyze', action='store_true',
        help='Calculate stock metrics and merge with Benford data'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all steps (download, analyze, visualize)'
    )
    parser.add_argument(
        '--force-refresh', action='store_true',
        help='Force re-download even if data is cached'
    )
    parser.add_argument(
        '--companies', type=str, default=DEFAULT_COMPANIES_FILE,
        help=f'Path to companies CSV (default: {DEFAULT_COMPANIES_FILE})'
    )
    parser.add_argument(
        '--benford', type=str, default=DEFAULT_BENFORD_FILE,
        help=f'Path to Benford analysis CSV (default: {DEFAULT_BENFORD_FILE})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--graphs-dir', type=str, default=DEFAULT_GRAPHS_DIR,
        help=f'Output directory for graphs (default: {DEFAULT_GRAPHS_DIR})'
    )

    args = parser.parse_args()

    # If no action specified, show help
    if not (args.download or args.analyze or args.visualize or args.all):
        parser.print_help()
        return

    # Define file paths
    stock_cache = DEFAULT_STOCK_CACHE
    stock_metrics_file = f"{args.output_dir}/stock_metrics.csv"
    combined_file = f"{args.output_dir}/combined_analysis.csv"

    # Run requested steps
    if args.all or args.download:
        stock_data = download_stock_data(
            companies_file=args.companies,
            cache_dir=stock_cache,
            force_refresh=args.force_refresh
        )

        if args.all or args.analyze:
            metrics_df = calculate_stock_metrics_batch(
                stock_data=stock_data,
                output_file=stock_metrics_file
            )

    if (args.analyze and not args.download) or (args.all and not args.download):
        # Load from cache
        downloader = StockDataDownloader(cache_dir=stock_cache)
        symbols = load_symbols_from_companies(args.companies)

        stock_data = {}
        for symbol in tqdm(symbols, desc="Loading cached data"):
            df = downloader.load_cached_stock(symbol)
            if df is not None:
                stock_data[symbol] = df

        if stock_data:
            metrics_df = calculate_stock_metrics_batch(
                stock_data=stock_data,
                output_file=stock_metrics_file
            )

    if args.all or args.analyze:
        merge_with_benford(
            stock_metrics_file=stock_metrics_file,
            benford_file=args.benford,
            companies_file=args.companies,
            output_file=combined_file
        )

    if args.all or args.visualize:
        generate_visualizations(
            combined_file=combined_file,
            output_dir=args.graphs_dir
        )

    print("\n" + "="*60)
    print("Stock Analysis Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

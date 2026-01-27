#!/usr/bin/env python3
"""
Generate PDF/HTML Reports for Benford's Law Analysis

Creates professional reports with executive summaries, visualizations,
and company-by-company breakdowns.

Usage:
    python scripts/generate_report.py --format both
    python scripts/generate_report.py --format pdf -o my_report.pdf
    python scripts/generate_report.py --format html --top-companies 100
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import jinja2
    except ImportError:
        missing.append('jinja2')

    try:
        import weasyprint
    except ImportError:
        missing.append('weasyprint')

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Generate Benford's Law Analysis Reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate both PDF and HTML reports
    python scripts/generate_report.py --format both

    # Generate only PDF report
    python scripts/generate_report.py --format pdf -o my_report.pdf

    # Generate HTML with interactive charts
    python scripts/generate_report.py --format html --top-companies 100

    # Generate from specific data file
    python scripts/generate_report.py --benford-file data/output/results/benford_analysis.csv
        """
    )

    parser.add_argument(
        '--format', '-f',
        choices=['pdf', 'html', 'both'],
        default='html',
        help='Output format (default: html)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output filename (without extension if using --format both)'
    )
    parser.add_argument(
        '--benford-file',
        default='data/output/results/benford_500_companies_analysis.csv',
        help='Path to Benford analysis CSV file'
    )
    parser.add_argument(
        '--combined-file',
        default='data/output/results/combined_analysis.csv',
        help='Path to combined Benford + stock data CSV (optional)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/output/reports',
        help='Directory to save reports'
    )
    parser.add_argument(
        '--top-companies', '-n',
        type=int,
        default=50,
        help='Number of companies for detailed breakdown (default: 50)'
    )
    parser.add_argument(
        '--no-details',
        action='store_true',
        help='Skip individual company sections'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive charts in HTML (use static images)'
    )

    args = parser.parse_args()

    # Check dependencies for requested format
    if args.format in ['pdf', 'both']:
        missing = check_dependencies()
        if missing:
            print("Missing required dependencies for PDF generation:")
            for dep in missing:
                print(f"  - {dep}")
            print("\nInstall them with:")
            print(f"  pip install {' '.join(missing)}")

            if args.format == 'both':
                print("\nFalling back to HTML only...")
                args.format = 'html'
            else:
                sys.exit(1)

    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_name = args.output or f'benford_report_{timestamp}'

    # Remove extension if provided
    if base_name.endswith(('.pdf', '.html')):
        base_name = base_name.rsplit('.', 1)[0]

    print("=" * 70)
    print("BENFORD'S LAW ANALYSIS REPORT GENERATOR")
    print("=" * 70)

    # Check if benford file exists
    benford_path = Path(args.benford_file)
    if not benford_path.exists():
        print(f"\nERROR: Benford analysis file not found: {args.benford_file}")
        print("Please run batch analysis first with: python scripts/run_analysis.py")
        sys.exit(1)

    # Check for combined file
    combined_path = Path(args.combined_file) if args.combined_file else None
    if combined_path and not combined_path.exists():
        print(f"\nNote: Combined analysis file not found: {args.combined_file}")
        print("Stock data will not be included in reports.")
        combined_path = None

    # Import report generator modules
    from src.report_generator import (
        ReportDataCollector,
        ReportChartGenerator,
        HTMLReportGenerator,
        PDFReportGenerator
    )

    # Initialize data collector
    print(f"\nLoading data from: {benford_path}")
    collector = ReportDataCollector(
        benford_file=str(benford_path),
        combined_file=str(combined_path) if combined_path else None
    )

    summary = collector.get_portfolio_summary()
    print(f"  Loaded {summary.total_companies} companies")
    print(f"  Analysis period: {summary.analysis_period}")
    print(f"  High risk companies: {summary.companies_above_threshold}")

    # Initialize chart generator
    chart_gen = ReportChartGenerator(output_dir=str(charts_dir))

    # Generate reports
    generated_files = []

    if args.format in ['html', 'both']:
        print("\nGenerating HTML report...")
        try:
            html_gen = HTMLReportGenerator(output_dir=str(output_dir))
            html_path = html_gen.generate_report(
                data_collector=collector,
                chart_generator=chart_gen,
                output_filename=f'{base_name}.html',
                top_n_companies=args.top_companies,
                show_company_details=not args.no_details,
                interactive=not args.no_interactive
            )
            print(f"  HTML report saved: {html_path}")
            generated_files.append(html_path)
        except Exception as e:
            print(f"  ERROR generating HTML report: {e}")

    if args.format in ['pdf', 'both']:
        print("\nGenerating PDF report...")
        try:
            pdf_gen = PDFReportGenerator(output_dir=str(output_dir))
            pdf_path = pdf_gen.generate_report(
                data_collector=collector,
                chart_generator=chart_gen,
                output_filename=f'{base_name}.pdf',
                top_n_companies=args.top_companies,
                show_company_details=not args.no_details
            )
            print(f"  PDF report saved: {pdf_path}")
            generated_files.append(pdf_path)
        except ImportError as e:
            print(f"  WARNING: PDF generation requires weasyprint: {e}")
            print("  Install with: pip install weasyprint")
        except Exception as e:
            print(f"  ERROR generating PDF report: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)

    if generated_files:
        print("\nGenerated files:")
        for f in generated_files:
            print(f"  - {f}")

        print(f"\nCharts saved to: {charts_dir}")
    else:
        print("\nNo reports were generated.")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download SEC Financial Data

This script downloads quarterly financial statement data from SEC EDGAR.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader import SECDataDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download SEC quarterly financial data"
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="User-Agent string (required by SEC). Format: 'CompanyName email@example.com'"
    )
    parser.add_argument(
        "--output-dir",
        default="./sec_data",
        help="Directory to save downloaded data (default: ./sec_data)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Download specific year (all quarters)"
    )
    parser.add_argument(
        "--quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Download specific quarter (requires --year)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Start year for range download"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="End year for range download"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Download latest available quarter"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep ZIP files after extraction"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = SECDataDownloader(
        user_agent=args.user_agent,
        output_dir=args.output_dir
    )

    # Determine what to download
    if args.latest:
        print("Downloading latest quarter...")
        downloader.download_latest(keep_zip=args.keep_zip)

    elif args.year and args.quarter:
        print(f"Downloading {args.year}Q{args.quarter}...")
        downloader.download_quarter(args.year, args.quarter, keep_zip=args.keep_zip)

    elif args.year:
        print(f"Downloading all quarters for {args.year}...")
        downloader.download_year(args.year, keep_zip=args.keep_zip)

    elif args.start_year and args.end_year:
        print(f"Downloading {args.start_year} to {args.end_year}...")
        downloader.download_range(args.start_year, args.end_year, keep_zip=args.keep_zip)

    else:
        print("No download option specified. Use --latest, --year, or --start-year/--end-year")
        parser.print_help()
        sys.exit(1)

    print("\nDownload complete!")


if __name__ == "__main__":
    main()

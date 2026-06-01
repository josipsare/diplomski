#!/usr/bin/env python3
"""
Generate Benford's Law Analysis Visualizations

This script generates all visualization graphs from the analysis results.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="Generate Benford's Law analysis visualizations"
    )
    parser.add_argument(
        "--input",
        default="./data/output/results/benford_analysis.csv",
        help="Path to analysis results CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/output/graphs",
        help="Directory to save graphs"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BENFORD'S LAW VISUALIZATION GENERATOR")
    print("=" * 60)
    print(f"\nInput file:  {args.input}")
    print(f"Output dir:  {args.output_dir}")
    print()

    generate_all_visualizations(
        data_file=args.input,
        output_dir=args.output_dir
    )

    print(f"\nGraphs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

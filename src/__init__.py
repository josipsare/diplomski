"""
SEC Financial Data Benford Analysis Package

A toolkit for downloading SEC EDGAR financial data and analyzing it
for conformance to Benford's Law. Includes stock price correlation analysis.
"""

from .downloader import SECDataDownloader
from .parser import SECDataParser
from .benford import (
    calculate_benford_metrics,
    calculate_second_digit_benford_metrics,
    calculate_digit_zscores,
    calculate_anomaly_score,
    get_digit_distribution,
    interpret_results,
    interpret_second_digit_results,
    BENFORD_EXPECTED,
    BENFORD_SECOND_DIGIT_EXPECTED
)
from .stock_downloader import (
    StockDataDownloader,
    load_symbols_from_companies,
    get_cik_symbol_mapping,
    get_symbol_cik_mapping
)
from .stock_metrics import (
    calculate_annual_return,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volume_benford,
    calculate_stock_metrics_for_year,
    calculate_all_metrics,
    merge_benford_and_stock_data,
    reshape_to_long_format,
    get_arrow_plot_data
)
from .stock_visualization import (
    plot_benford_arrow_plot,
    plot_arrow_plot_by_year,
    plot_volume_benford_heatmap,
    plot_combined_time_series,
    plot_correlation_summary,
    generate_all_stock_visualizations
)

__version__ = "1.2.0"
__author__ = "Diploma Thesis Project"

__all__ = [
    # SEC Data
    "SECDataDownloader",
    "SECDataParser",
    # Benford Analysis
    "calculate_benford_metrics",
    "calculate_second_digit_benford_metrics",
    "calculate_digit_zscores",
    "calculate_anomaly_score",
    "get_digit_distribution",
    "interpret_results",
    "interpret_second_digit_results",
    "BENFORD_EXPECTED",
    "BENFORD_SECOND_DIGIT_EXPECTED",
    # Stock Data
    "StockDataDownloader",
    "load_symbols_from_companies",
    "get_cik_symbol_mapping",
    "get_symbol_cik_mapping",
    # Stock Metrics
    "calculate_annual_return",
    "calculate_volatility",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_volume_benford",
    "calculate_stock_metrics_for_year",
    "calculate_all_metrics",
    "merge_benford_and_stock_data",
    "reshape_to_long_format",
    "get_arrow_plot_data",
    # Stock Visualization
    "plot_benford_arrow_plot",
    "plot_arrow_plot_by_year",
    "plot_volume_benford_heatmap",
    "plot_combined_time_series",
    "plot_correlation_summary",
    "generate_all_stock_visualizations"
]

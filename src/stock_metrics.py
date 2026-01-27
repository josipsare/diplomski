"""
Stock Metrics Calculator

Calculates stock performance metrics and applies Benford's Law analysis
to trading volumes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .benford import calculate_benford_metrics, calculate_second_digit_benford_metrics


def calculate_annual_return(prices: pd.Series) -> float:
    """
    Calculate annual return from daily prices.

    Args:
        prices: Series of daily closing prices for a year

    Returns:
        Annual return as percentage (e.g., 15.5 for 15.5%)
    """
    if len(prices) < 2:
        return np.nan

    # Filter out NaN values
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan

    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]

    if first_price == 0 or pd.isna(first_price):
        return np.nan

    return ((last_price - first_price) / first_price) * 100


def calculate_volatility(prices: pd.Series) -> float:
    """
    Calculate annualized volatility from daily prices.

    Volatility = std(daily_returns) * sqrt(252)

    Args:
        prices: Series of daily closing prices

    Returns:
        Annualized volatility as percentage
    """
    if len(prices) < 10:
        return np.nan

    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()

    if len(daily_returns) < 10:
        return np.nan

    # Annualized volatility (252 trading days)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    return volatility


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown from daily prices.

    Max Drawdown = max((peak - price) / peak)

    Args:
        prices: Series of daily closing prices

    Returns:
        Maximum drawdown as percentage (negative value)
    """
    if len(prices) < 2:
        return np.nan

    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan

    # Calculate running maximum
    cummax = prices.cummax()

    # Calculate drawdown
    drawdown = (prices - cummax) / cummax

    # Return maximum drawdown (most negative value)
    return drawdown.min() * 100


def calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.

    Sharpe = (annual_return - risk_free_rate) / volatility

    Args:
        prices: Series of daily closing prices
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sharpe ratio
    """
    annual_return = calculate_annual_return(prices) / 100  # Convert to decimal
    volatility = calculate_volatility(prices) / 100  # Convert to decimal

    if pd.isna(annual_return) or pd.isna(volatility) or volatility == 0:
        return np.nan

    return (annual_return - risk_free_rate) / volatility


def calculate_volume_benford(volumes: pd.Series) -> Dict[str, float]:
    """
    Apply Benford's Law analysis to trading volumes.

    Args:
        volumes: Series of daily trading volumes

    Returns:
        Dictionary with Benford metrics (chi_square, p_value, MAD, KS_test, n_samples)
    """
    # Filter valid volumes (positive, non-zero)
    valid_volumes = volumes[volumes > 0].dropna()

    if len(valid_volumes) < 50:
        return {
            'chi_square': np.nan,
            'p_value': np.nan,
            'MAD': np.nan,
            'KS_test': np.nan,
            'n_samples': len(valid_volumes)
        }

    return calculate_benford_metrics(valid_volumes)


def calculate_stock_metrics_for_year(
    df: pd.DataFrame,
    year: int
) -> Dict[str, float]:
    """
    Calculate all stock metrics for a specific year.

    Args:
        df: DataFrame with stock data (Date index, Close, Volume columns)
        year: Year to analyze

    Returns:
        Dictionary with all metrics for the year
    """
    # Filter data for the year
    if isinstance(df.index, pd.DatetimeIndex):
        # Handle timezone-aware DatetimeIndex
        year_data = df[df.index.year == year]
    else:
        df_temp = df.copy()
        # Convert index to datetime, handling timezone-aware datetimes
        try:
            df_temp['Date'] = pd.to_datetime(df_temp.index, utc=True)
        except Exception:
            df_temp['Date'] = pd.to_datetime(df_temp.index)
        year_data = df_temp[df_temp['Date'].dt.year == year]

    if len(year_data) < 20:
        return {
            'annual_return': np.nan,
            'volatility': np.nan,
            'max_drawdown': np.nan,
            'sharpe_ratio': np.nan,
            'volume_benford_MAD': np.nan,
            'volume_benford_chi_square': np.nan,
            'volume_benford_p_value': np.nan,
            'avg_volume': np.nan,
            'trading_days': len(year_data)
        }

    # Get price and volume columns
    close_col = 'Close' if 'Close' in year_data.columns else 'Adj Close'
    prices = year_data[close_col] if close_col in year_data.columns else pd.Series()
    volumes = year_data['Volume'] if 'Volume' in year_data.columns else pd.Series()

    # Calculate metrics
    metrics = {
        'annual_return': calculate_annual_return(prices),
        'volatility': calculate_volatility(prices),
        'max_drawdown': calculate_max_drawdown(prices),
        'sharpe_ratio': calculate_sharpe_ratio(prices),
        'avg_volume': volumes.mean() if len(volumes) > 0 else np.nan,
        'trading_days': len(year_data)
    }

    # Benford analysis on volumes
    if len(volumes) > 50:
        benford = calculate_volume_benford(volumes)
        metrics['volume_benford_MAD'] = benford['MAD']
        metrics['volume_benford_chi_square'] = benford['chi_square']
        metrics['volume_benford_p_value'] = benford['p_value']
    else:
        metrics['volume_benford_MAD'] = np.nan
        metrics['volume_benford_chi_square'] = np.nan
        metrics['volume_benford_p_value'] = np.nan

    return metrics


def calculate_all_metrics(
    stock_data: Dict[str, pd.DataFrame],
    years: List[int]
) -> pd.DataFrame:
    """
    Calculate metrics for all stocks across all years.

    Args:
        stock_data: Dictionary mapping symbol -> DataFrame with stock data
        years: List of years to analyze

    Returns:
        DataFrame with one row per symbol, columns for each year's metrics
    """
    results = []

    for symbol, df in stock_data.items():
        row = {'symbol': symbol}

        for year in years:
            metrics = calculate_stock_metrics_for_year(df, year)

            # Add year prefix to column names
            for metric, value in metrics.items():
                row[f'year_{year}_stock_{metric}'] = value

        results.append(row)

    return pd.DataFrame(results)


def calculate_2year_deltas(
    metrics_df: pd.DataFrame,
    metric_name: str,
    years: List[int]
) -> pd.DataFrame:
    """
    Calculate 2-year changes in a metric.

    Args:
        metrics_df: DataFrame with metrics (rows = symbols)
        metric_name: Name of metric to calculate deltas for
        years: List of years

    Returns:
        DataFrame with delta columns added
    """
    df = metrics_df.copy()

    for year in years:
        year_minus_2 = year - 2

        col_current = f'year_{year}_stock_{metric_name}'
        col_previous = f'year_{year_minus_2}_stock_{metric_name}'
        delta_col = f'year_{year}_delta_{metric_name}'

        if col_current in df.columns and col_previous in df.columns:
            df[delta_col] = df[col_current] - df[col_previous]

    return df


def merge_benford_and_stock_data(
    benford_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    cik_symbol_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Merge Benford analysis data with stock metrics.

    Args:
        benford_df: DataFrame with Benford analysis (indexed by CIK)
        stock_df: DataFrame with stock metrics (indexed by symbol)
        cik_symbol_mapping: Dictionary mapping CIK -> symbol

    Returns:
        Merged DataFrame with both Benford and stock metrics
    """
    # Add symbol column to Benford data
    benford_with_symbol = benford_df.copy()
    benford_with_symbol['symbol'] = benford_with_symbol['cik'].map(cik_symbol_mapping)

    # Merge on symbol
    merged = pd.merge(
        benford_with_symbol,
        stock_df,
        on='symbol',
        how='inner'
    )

    return merged


def reshape_to_long_format(
    wide_df: pd.DataFrame,
    id_cols: List[str] = ['cik', 'company_name', 'symbol']
) -> pd.DataFrame:
    """
    Reshape wide format data to long format for visualization.

    Converts from:
        cik, company, year_2014_MAD, year_2014_return, year_2015_MAD, ...

    To:
        cik, company, year, MAD, return, ...

    Args:
        wide_df: Wide format DataFrame
        id_cols: Columns to use as identifiers

    Returns:
        Long format DataFrame
    """
    # Find all year columns
    year_cols = [c for c in wide_df.columns if c.startswith('year_')]

    # Extract unique years
    years = set()
    for col in year_cols:
        parts = col.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            years.add(int(parts[1]))

    years = sorted(years)

    # Build long format
    rows = []
    for _, row in wide_df.iterrows():
        for year in years:
            new_row = {col: row[col] for col in id_cols if col in row}
            new_row['year'] = year

            # Extract metrics for this year
            for col in year_cols:
                if col.startswith(f'year_{year}_'):
                    metric_name = col.replace(f'year_{year}_', '')
                    new_row[metric_name] = row[col]

            rows.append(new_row)

    return pd.DataFrame(rows)


def get_arrow_plot_data(
    long_df: pd.DataFrame,
    benford_metric: str = 'MAD',
    stock_metric: str = 'stock_annual_return'
) -> pd.DataFrame:
    """
    Prepare data for the arrow plot visualization.

    Args:
        long_df: Long format DataFrame with all metrics
        benford_metric: Benford metric to use for X-axis
        stock_metric: Stock metric to use for Y-axis

    Returns:
        DataFrame with columns: symbol, year, x (benford), y (return), next_year_up (bool)
    """
    # Sort by symbol and year
    df = long_df.sort_values(['symbol', 'year'])

    rows = []
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('year')

        for i in range(len(symbol_data) - 1):
            current = symbol_data.iloc[i]
            next_year = symbol_data.iloc[i + 1]

            x = current.get(benford_metric, np.nan)
            y = current.get(stock_metric, np.nan)
            next_return = next_year.get(stock_metric, np.nan)

            if pd.notna(x) and pd.notna(y) and pd.notna(next_return):
                rows.append({
                    'symbol': symbol,
                    'year': current['year'],
                    'x': x,
                    'y': y,
                    'next_year_return': next_return,
                    'next_year_up': next_return > 0,
                    'arrow_magnitude': abs(next_return)
                })

    return pd.DataFrame(rows)


# Example usage
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    print("Testing stock metrics calculation...")

    # Download test data
    aapl = yf.download("AAPL", start="2020-01-01", end="2024-01-01", progress=False)

    # Calculate metrics for each year
    for year in [2020, 2021, 2022, 2023]:
        metrics = calculate_stock_metrics_for_year(aapl, year)
        print(f"\nAAPL {year}:")
        for key, value in metrics.items():
            if pd.notna(value):
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

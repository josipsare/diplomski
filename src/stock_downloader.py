"""
Stock Price Data Downloader

Downloads historical stock price data using yfinance for companies in the analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import time
from datetime import datetime
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")


class StockDataDownloader:
    """
    Download historical stock price data via yfinance.

    Handles bulk downloads for multiple companies with caching
    to avoid redundant API calls.
    """

    def __init__(self, cache_dir: str = "./data/stock_data"):
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded stock data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting (yfinance is generally lenient but be respectful)
        self.request_delay = 0.1  # seconds between requests

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"{symbol}.csv"

    def _is_cached(self, symbol: str, start_year: int, end_year: int) -> bool:
        """Check if data is already cached with required date range."""
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return False

        try:
            df = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
            if len(df) == 0:
                return False

            cached_start = df.index.min().year
            cached_end = df.index.max().year

            # Check if cache covers required range
            return cached_start <= start_year and cached_end >= end_year
        except Exception:
            return False

    def download_stock(
        self,
        symbol: str,
        start_year: int = 2014,
        end_year: int = 2024,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download historical stock data for a single symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_year: Start year for data
            end_year: End year for data
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with OHLCV data, or None if download failed
        """
        cache_path = self._get_cache_path(symbol)

        # Check cache first
        if not force_refresh and self._is_cached(symbol, start_year, end_year):
            try:
                df = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
                return df
            except Exception:
                pass  # Fall through to download

        try:
            # Download from yfinance
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year + 1}-01-01"  # End is exclusive

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                print(f"  No data for {symbol}")
                return None

            # Reset index to have Date as column
            df = df.reset_index()

            # Rename columns to standard names
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

            # Save to cache
            df.to_csv(cache_path, index=False)

            # Rate limiting
            time.sleep(self.request_delay)

            return df.set_index('Date')

        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")
            return None

    def download_all_stocks(
        self,
        symbols: List[str],
        start_year: int = 2014,
        end_year: int = 2024,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_year: Start year for data
            end_year: End year for data
            force_refresh: If True, re-download even if cached

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Downloading stock data for {len(symbols)} companies")
        print(f"Period: {start_year} - {end_year}")
        print(f"{'='*60}\n")

        results = {}
        failed = []

        for symbol in tqdm(symbols, desc="Downloading stocks"):
            df = self.download_stock(symbol, start_year, end_year, force_refresh)
            if df is not None and not df.empty:
                results[symbol] = df
            else:
                failed.append(symbol)

        print(f"\n{'='*60}")
        print(f"Downloaded: {len(results)}/{len(symbols)} symbols")
        if failed:
            print(f"Failed ({len(failed)}): {', '.join(failed[:10])}" +
                  (f"... and {len(failed)-10} more" if len(failed) > 10 else ""))
        print(f"{'='*60}\n")

        return results

    def download_bulk(
        self,
        symbols: List[str],
        start_year: int = 2014,
        end_year: int = 2024,
        threads: bool = True
    ) -> pd.DataFrame:
        """
        Bulk download using yfinance's optimized download function.

        This is faster for many symbols as it uses multi-threading.

        Args:
            symbols: List of ticker symbols
            start_year: Start year for data
            end_year: End year for data
            threads: Whether to use multi-threading

        Returns:
            Multi-level DataFrame with all stock data
        """
        print(f"\n{'='*60}")
        print(f"Bulk downloading stock data for {len(symbols)} companies")
        print(f"Period: {start_year} - {end_year}")
        print(f"{'='*60}\n")

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year + 1}-01-01"

        # Use yfinance bulk download
        data = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            threads=threads,
            progress=True,
            repair=True  # Fix price anomalies
        )

        # Cache individual symbols
        if isinstance(data.columns, pd.MultiIndex):
            for symbol in data.columns.get_level_values(0).unique():
                try:
                    symbol_data = data[symbol].dropna(how='all')
                    if not symbol_data.empty:
                        cache_path = self._get_cache_path(symbol)
                        symbol_data.reset_index().to_csv(cache_path, index=False)
                except Exception:
                    pass

        print(f"\nBulk download complete")
        return data

    def load_cached_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from cache.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with stock data, or None if not cached
        """
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
            return df
        except Exception:
            return None

    def get_cached_symbols(self) -> List[str]:
        """Get list of symbols that are cached."""
        cached = []
        for f in self.cache_dir.glob("*.csv"):
            cached.append(f.stem)
        return cached

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.

        Args:
            symbol: If provided, clear only this symbol. Otherwise clear all.
        """
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
                print(f"Cleared cache for {symbol}")
        else:
            for f in self.cache_dir.glob("*.csv"):
                f.unlink()
            print("Cleared all cached stock data")


def load_symbols_from_companies(companies_file: str) -> List[str]:
    """
    Load stock symbols from companies CSV file.

    Args:
        companies_file: Path to companies.csv

    Returns:
        List of stock symbols
    """
    df = pd.read_csv(companies_file)

    if 'symbol' not in df.columns:
        raise ValueError("companies.csv must have a 'symbol' column")

    symbols = df['symbol'].dropna().unique().tolist()
    return symbols


def get_cik_symbol_mapping(companies_file: str) -> Dict[str, str]:
    """
    Get mapping from CIK to stock symbol.

    Args:
        companies_file: Path to companies.csv

    Returns:
        Dictionary mapping CIK -> symbol
    """
    df = pd.read_csv(companies_file, dtype={'cik': str})
    mapping = dict(zip(df['cik'], df['symbol']))
    return mapping


def get_symbol_cik_mapping(companies_file: str) -> Dict[str, str]:
    """
    Get mapping from stock symbol to CIK.

    Args:
        companies_file: Path to companies.csv

    Returns:
        Dictionary mapping symbol -> CIK
    """
    df = pd.read_csv(companies_file, dtype={'cik': str})
    mapping = dict(zip(df['symbol'], df['cik']))
    return mapping


# Example usage
if __name__ == "__main__":
    # Test with a few symbols
    downloader = StockDataDownloader(cache_dir="./data/stock_data")

    # Test single download
    print("Testing single stock download...")
    aapl = downloader.download_stock("AAPL", 2014, 2024)
    if aapl is not None:
        print(f"AAPL data: {len(aapl)} rows")
        print(aapl.head())

    # Test bulk download
    print("\nTesting bulk download...")
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    data = downloader.download_bulk(test_symbols, 2020, 2024)
    print(f"Bulk data shape: {data.shape}")

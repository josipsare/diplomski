"""Batch-download daily stock-price CSVs for the Russell 3000 universe.

Each ticker is fetched via yfinance and saved to `data/stock_data/{TICKER}.csv`
in the same schema as the bachelor's-thesis CSVs (Date, Open, High, Low, Close,
Volume, Dividends, Stock Splits — index reset, tz-aware Date).

Usage:
    python scripts/download_stock_data.py                   # all missing tickers from companies_russell3000.csv
    python scripts/download_stock_data.py --universe sp500  # restrict to bachelor's universe (companies.csv)
    python scripts/download_stock_data.py --tickers AAPL    # specific tickers
    python scripts/download_stock_data.py --max 20          # cap, useful for smoke test
    python scripts/download_stock_data.py --refresh         # re-download even if CSV exists

Skipped tickers (delisted, ticker mismatch, etc.) are logged to a sidecar
`data/stock_data/_download_failures.log`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import INPUT_DIR, STOCK_DATA_DIR

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["russell3000", "sp500"], default="russell3000",
                   help="Which companies CSV to read")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Override: download only these tickers")
    p.add_argument("--max", type=int, default=None,
                   help="Cap number of tickers downloaded (for smoke testing)")
    p.add_argument("--refresh", action="store_true",
                   help="Re-download even if CSV already exists")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Sleep between requests (seconds)")
    return p.parse_args()


def _read_universe(name: str) -> list[str]:
    if name == "russell3000":
        path = INPUT_DIR / "companies_russell3000.csv"
    else:
        path = INPUT_DIR / "companies.csv"
    df = pd.read_csv(path, dtype=str)
    col = "symbol" if "symbol" in df.columns else df.columns[2]  # bachelor's CSV uses col 2
    return df[col].dropna().astype(str).str.strip().tolist()


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one ticker via yfinance. Returns empty DataFrame if no data."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    return df


def write_csv(ticker: str, df: pd.DataFrame, out_dir: Path) -> Path:
    """Write a ticker's daily DataFrame to disk in the existing schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.tickers:
        all_tickers = list(args.tickers)
    else:
        all_tickers = _read_universe(args.universe)
        log.info("Loaded %d tickers from %s universe", len(all_tickers), args.universe)

    existing = {p.stem for p in STOCK_DATA_DIR.glob("*.csv")}
    if args.refresh:
        targets = all_tickers
    else:
        targets = [t for t in all_tickers if t not in existing]
    log.info("Of %d tickers, %d already on disk, %d to download",
             len(all_tickers), len(all_tickers) - len(targets), len(targets))

    if args.max is not None:
        targets = targets[: args.max]
        log.info("Capped to %d tickers", len(targets))

    failures_path = STOCK_DATA_DIR / "_download_failures.log"
    n_ok, n_fail = 0, 0
    failures: list[str] = []

    for i, ticker in enumerate(targets, start=1):
        try:
            df = download_one(ticker, args.start, args.end)
            if df.empty:
                failures.append(f"{ticker}: empty DataFrame from yfinance")
                n_fail += 1
            else:
                write_csv(ticker, df, STOCK_DATA_DIR)
                n_ok += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            n_fail += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
        if i % 50 == 0:
            log.info("Progress: %d/%d  ok=%d  fail=%d", i, len(targets), n_ok, n_fail)

    log.info("Done. %d downloaded, %d failed (out of %d)", n_ok, n_fail, len(targets))
    if failures:
        failures_path.write_text("\n".join(failures), encoding="utf-8")
        log.info("Failures logged to %s", failures_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

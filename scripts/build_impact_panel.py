"""Build the daily-returns + market-proxy outputs that feed the impact analysis.

Sibling to `build_panel.py`. Kept separate because the daily concat across
~440 tickers is the slow part of impact-data ingestion (~10 sec end-to-end on
the current universe; longer once we expand to Russell 3000).

Usage:
    python scripts/build_impact_panel.py            # all tickers
    python scripts/build_impact_panel.py --quick    # 5 tickers, sanity check
    python scripts/build_impact_panel.py --tickers AAPL MSFT ORCL

Outputs:
    data/output/panels/daily_returns.parquet
    data/output/panels/market_returns.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.daily_stock_loader import build_market_proxy, load_all_daily
from src.data.panel_builder import save_panel
from src.utils.paths import PANEL_DIR, STOCK_DATA_DIR

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="5 tickers only")
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    tickers = args.tickers
    if args.quick:
        tickers = tickers or ["AAPL", "MSFT", "ORCL", "MPW", "PM"]
        log.info("Quick mode: tickers=%s", tickers)

    show_progress = not args.no_progress

    log.info("Loading daily price series for %s tickers …",
             "all" if tickers is None else len(tickers))
    daily = load_all_daily(STOCK_DATA_DIR, tickers=tickers, show_progress=show_progress)
    log.info("Daily rows: %d (across %d tickers, %s to %s)",
             len(daily), daily["ticker"].nunique() if not daily.empty else 0,
             daily["date"].min().date() if not daily.empty else None,
             daily["date"].max().date() if not daily.empty else None)
    save_panel(daily, PANEL_DIR / "daily_returns.parquet")

    log.info("Building equal-weighted market proxy …")
    market = build_market_proxy(daily)
    log.info("Market series: %d trading days", len(market))
    save_panel(market, PANEL_DIR / "market_returns.parquet")

    log.info("Done. Outputs in %s", PANEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

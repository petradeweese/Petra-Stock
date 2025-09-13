"""Nightly ETL job to keep the local DB in sync with Polygon.

The job sleeps until 8:15pm US/Eastern each day, then fetches the last three
days of 15-minute bars for the configured symbols and upserts them into the
database.  Symbols are read from a newline-delimited text file specified on the
command line.
"""

import datetime as dt
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from services import polygon_client
from services.price_store import upsert_bars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nightly_etl")


def load_symbols(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def run_once(symbols: list[str]) -> None:
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=3)
    for sym in symbols:
        t0 = time.monotonic()
        df = polygon_client.fetch_polygon_prices([sym], "15m", start, end)[sym]
        rows = upsert_bars(sym, df, "15m")
        logger.info(
            "nightly symbol=%s rows=%d duration=%.2fs",
            sym,
            rows,
            time.monotonic() - t0,
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: nightly_etl.py symbols.txt", file=sys.stderr)
        sys.exit(1)
    symbols = load_symbols(Path(sys.argv[1]))
    ny = ZoneInfo("America/New_York")
    while True:
        now = dt.datetime.now(ny)
        target = now.replace(hour=20, minute=15, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        wait = (target - now).total_seconds()
        logger.info("sleeping %.0fs until next run", wait)
        time.sleep(wait)
        run_once(symbols)


if __name__ == "__main__":
    main()

"""One-time backfill of 15m bars from Polygon and store into the DB.

Usage:
    python scripts/backfill_polygon.py symbols.txt [--dry-run]
    python scripts/backfill_polygon.py --test [--dry-run]

``symbols.txt`` should contain one symbol per line.  The script fetches a full
year of 15-minute bars for each symbol and upserts them into the database.  A
per-symbol progress log is emitted including row counts and duration.

``--test`` performs a quick sanity check by fetching one day of 15-minute bars
for SPY and logging how many rows were returned and saved.
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

from services import http_client, polygon_client
from services.price_store import upsert_bars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill")

CHECKPOINT = Path("backfill_checkpoint.json")


def load_symbols(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def load_checkpoint() -> int:
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text()).get("index", 0)
        except Exception:
            return 0
    return 0


def save_checkpoint(idx: int) -> None:
    CHECKPOINT.write_text(json.dumps({"index": idx}))


async def _backfill(
    symbols: list[str],
    dry_run: bool = False,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    use_checkpoint: bool = True,
) -> None:
    if end is None:
        end = dt.datetime.now(tz=dt.timezone.utc)
    if start is None:
        start = end - dt.timedelta(days=365)

    start_idx = load_checkpoint() if use_checkpoint else 0
    for i, sym in enumerate(symbols[start_idx:], start_idx):
        t0 = time.monotonic()
        data_map = await polygon_client.fetch_polygon_prices_async(
            [sym], "15m", start, end
        )
        data = data_map[sym]
        bars = len(data)
        rows = 0
        if not dry_run:
            rows = upsert_bars(sym, data, "15m")
        logger.info(
            "backfill symbol=%s returned=%d saved=%d duration=%.2fs",
            sym,
            bars,
            rows,
            time.monotonic() - t0,
        )
        if use_checkpoint:
            save_checkpoint(i + 1)
    await http_client.aclose()


def backfill(
    symbols: list[str],
    dry_run: bool = False,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    use_checkpoint: bool = True,
) -> None:
    asyncio.run(
        _backfill(
            symbols,
            dry_run,
            start=start,
            end=end,
            use_checkpoint=use_checkpoint,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill 15m bars from Polygon")
    parser.add_argument(
        "symbols", nargs="?", help="file containing symbols, one per line"
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--test", action="store_true", dest="test")
    args = parser.parse_args(argv)

    if args.test:
        end = dt.datetime.now(tz=dt.timezone.utc)
        start = end - dt.timedelta(days=1)
        backfill(
            ["SPY"], dry_run=args.dry_run, start=start, end=end, use_checkpoint=False
        )
        return

    if not args.symbols:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    symbols = load_symbols(Path(args.symbols))
    backfill(symbols, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""One-time backfill of 15m bars from Polygon and store into the DB.

Usage:
    python scripts/backfill_polygon.py symbols.txt

``symbols.txt`` should contain one symbol per line.  The script fetches a full
year of 15-minute bars for each symbol and upserts them into the database.  A
per-symbol progress log is emitted including row counts and duration.
"""

import sys
import time
import datetime as dt
import logging
import json
from pathlib import Path

from services import polygon_client
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


def backfill(symbols: list[str], dry_run: bool = False) -> None:
    start_idx = load_checkpoint()
    end = dt.datetime.now(tz=dt.timezone.utc)
    start = end - dt.timedelta(days=365)
    for i, sym in enumerate(symbols[start_idx:], start_idx):
        t0 = time.monotonic()
        data = polygon_client.fetch_polygon_prices([sym], "15m", start, end)[sym]
        rows = 0
        if not dry_run:
            rows = upsert_bars(sym, data)
        logger.info(
            "backfill symbol=%s rows=%d duration=%.2fs",
            sym,
            rows,
            time.monotonic() - t0,
        )
        save_checkpoint(i + 1)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: backfill_polygon.py symbols.txt [--dry-run]", file=sys.stderr)
        sys.exit(1)
    symbols = load_symbols(Path(sys.argv[1]))
    dry = "--dry-run" in sys.argv[2:]
    backfill(symbols, dry_run=dry)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Remove price cache parquet files created before today."""
from __future__ import annotations

from pathlib import Path
from datetime import date

CACHE_DIR = Path('.cache/prices')


def main() -> None:
    today = date.today()
    removed = 0
    if not CACHE_DIR.exists():
        return
    for path in CACHE_DIR.glob('*.parquet'):
        try:
            if date.fromtimestamp(path.stat().st_mtime) < today:
                path.unlink()
                removed += 1
        except Exception:
            continue
    print(f"removed {removed} files")


if __name__ == '__main__':
    main()

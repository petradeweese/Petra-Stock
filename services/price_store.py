import logging
import os
import sqlite3
import time
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore[import-untyped]

import db

logger = logging.getLogger(__name__)

TABLE_15M = "bars_15m"

# Simple in-process cache for DB reads so repeated scans within a short window
# do not thrash the database.
_CACHE: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
CACHE_TTL = int(os.getenv("DB_CACHE_TTL", "120"))  # seconds


def _open_conn():
    conn = db.get_engine().raw_connection()
    if hasattr(conn, "row_factory"):
        conn.row_factory = sqlite3.Row
    return conn


def clear_cache() -> None:
    """Clear the in-process DB result cache (mainly for tests)."""
    _CACHE.clear()


def upsert_bars(symbol: str, df: pd.DataFrame, conn=None) -> int:
    """Upsert OHLCV rows into the 15m bars table and log the row count."""
    if df is None or df.empty:
        return 0
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    rows = []
    for ts, row in df.iterrows():
        open_val = row.get("Open")
        high_val = row.get("High")
        low_val = row.get("Low")
        close_val = row.get("Close")
        volume_val = row.get("Volume")
        rows.append(
            (
                symbol,
                ts.isoformat(),
                float(open_val) if pd.notna(open_val) else None,
                float(high_val) if pd.notna(high_val) else None,
                float(low_val) if pd.notna(low_val) else None,
                float(close_val) if pd.notna(close_val) else None,
                int(volume_val) if pd.notna(volume_val) else 0,
            )
        )
    close_conn = False
    if conn is None:
        conn = _open_conn()
        close_conn = True
    try:
        conn.executemany(
            f"""
            INSERT INTO {TABLE_15M}(symbol, ts, open, high, low, close, volume)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            rows,
        )
        conn.commit()
    finally:
        if close_conn:
            conn.close()
    logger.info("db_upsert symbol=%s rows=%d", symbol, len(rows))
    return len(rows)


def get_prices_from_db(
    symbols: List[str], start, end, conn=None
) -> Dict[str, pd.DataFrame]:
    """Retrieve bars for symbols between start and end timestamps with caching."""
    close_conn = False
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        key = (sym, start.isoformat(), end.isoformat())
        cached = _CACHE.get(key)
        if cached and cached[0] > time.monotonic():
            logger.info("db_cache_hit symbol=%s", sym)
            results[sym] = cached[1]
            continue
        if conn is None:
            conn = _open_conn()
            close_conn = True
        cur = conn.execute(
            (
                f"SELECT ts, open, high, low, close, volume FROM {TABLE_15M} "
                "WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts"
            ),
            (sym, start.isoformat(), end.isoformat()),
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(
                rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
        else:
            df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        _CACHE[key] = (time.monotonic() + CACHE_TTL, df)
        results[sym] = df
    if close_conn and conn is not None:
        conn.close()
    return results


def detect_gaps(symbol: str, start, end, conn=None) -> List[pd.Timestamp]:
    """Return list of expected 15m timestamps missing from DB.

    The interval is treated as ``[start, end)`` so the ``end`` timestamp is
    exclusive. This prevents an off-by-one error when the range boundary falls
    exactly on a bar start.
    """
    data = get_prices_from_db([symbol], start, end, conn=conn)[symbol]
    expected = pd.date_range(
        start=start, end=end, freq="15min", tz="UTC", inclusive="left"
    )
    have = set(data.index)
    return [ts for ts in expected if ts not in have]

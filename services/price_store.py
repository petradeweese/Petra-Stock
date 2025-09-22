import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore[import-untyped]

import db

logger = logging.getLogger(__name__)

BARS_TABLE = "bars"

# Simple in-process cache for DB reads so repeated scans within a short window
# do not thrash the database.
_CACHE: Dict[Tuple[str, str, str, str], Tuple[float, pd.DataFrame]] = {}
CACHE_TTL = int(os.getenv("DB_CACHE_TTL", "120"))  # seconds


def _open_conn():
    conn = db.get_engine().raw_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    if hasattr(conn, "row_factory"):
        conn.row_factory = sqlite3.Row
    return conn


def clear_cache() -> None:
    """Clear the in-process DB result cache (mainly for tests)."""
    _CACHE.clear()


def upsert_bars(symbol: str, df: pd.DataFrame, interval: str = "15m", conn=None) -> int:
    """Upsert OHLCV rows into the bars table and log the row count."""
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
                interval,
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
            INSERT INTO {BARS_TABLE}(symbol, interval, ts, open, high, low, close,
                                    volume)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, interval, ts) DO UPDATE SET
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
    symbols: List[str], start, end, interval: str = "15m", conn=None
) -> Dict[str, pd.DataFrame]:
    """Retrieve bars for symbols between start and end timestamps with caching."""
    close_conn = False
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        key = (sym, interval, start.isoformat(), end.isoformat())
        cached = _CACHE.get(key)
        if cached and cached[0] > time.monotonic():
            logger.debug("db_cache_hit symbol=%s", sym)
            results[sym] = cached[1]
            continue
        if conn is None:
            conn = _open_conn()
            close_conn = True
        cur = conn.execute(
            (
                f"SELECT ts, open, high, low, close, volume FROM {BARS_TABLE} "
                "WHERE symbol=? AND interval=? AND ts>=? AND ts<=? ORDER BY ts"
            ),
            (sym, interval, start.isoformat(), end.isoformat()),
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(
                rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
            df["Adj Close"] = df["Close"]
            df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        else:
            df = pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )
        _CACHE[key] = (time.monotonic() + CACHE_TTL, df)
        results[sym] = df
    if close_conn and conn is not None:
        conn.close()
    return results


def detect_gaps(
    symbol: str, start, end, interval: str = "15m", conn=None
) -> List[pd.Timestamp]:
    """Return list of expected timestamps missing from DB.

    The interval is treated as ``[start, end)`` so the ``end`` timestamp is
    exclusive. This prevents an off-by-one error when the range boundary falls
    exactly on a bar start.
    """
    data = get_prices_from_db([symbol], start, end, interval=interval, conn=conn)[
        symbol
    ]
    freq = f"{interval[:-1]}min" if interval.endswith("m") else interval
    expected = pd.date_range(
        start=start, end=end, freq=freq, tz="UTC", inclusive="left"
    )
    have = set(data.index)
    return [ts for ts in expected if ts not in have]


def get_coverage(
    symbol: str, interval: str, conn=None
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return the min/max timestamps present for the symbol/interval."""
    close_conn = False
    if conn is None:
        conn = _open_conn()
        close_conn = True
    try:
        cur = conn.execute(
            f"SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts FROM {BARS_TABLE} "
            "WHERE symbol=? AND interval=?",
            (symbol, interval),
        )
        row = cur.fetchone()
        min_ts = None
        max_ts = None
        if row:
            if isinstance(row, sqlite3.Row):
                min_ts = row["min_ts"]
                max_ts = row["max_ts"]
            else:  # fallback when row factory not set
                min_ts, max_ts = row[0], row[1]
        return (
            pd.to_datetime(min_ts, utc=True).to_pydatetime() if min_ts else None,
            pd.to_datetime(max_ts, utc=True).to_pydatetime() if max_ts else None,
        )
    finally:
        if close_conn:
            conn.close()


def bulk_coverage(
    symbols: List[str], interval: str, start: datetime, end: datetime, conn=None
) -> Dict[str, Tuple[Optional[datetime], Optional[datetime], int]]:
    """Return coverage stats for multiple symbols in one query."""
    if not symbols:
        return {}
    close_conn = False
    if conn is None:
        conn = _open_conn()
        close_conn = True
    try:
        placeholders = ",".join(["?"] * len(symbols))
        cur = conn.execute(
            (
                f"SELECT symbol, MIN(ts) AS min_ts, MAX(ts) AS max_ts, COUNT(*) AS cnt "
                f"FROM {BARS_TABLE} WHERE interval=? AND ts>=? AND ts<=? "
                f"AND symbol IN ({placeholders}) GROUP BY symbol"
            ),
            [interval, start.isoformat(), end.isoformat(), *symbols],
        )
        rows = cur.fetchall()
        out: Dict[str, Tuple[Optional[datetime], Optional[datetime], int]] = {}
        for row in rows:
            if isinstance(row, sqlite3.Row):
                sym = row["symbol"]
                min_ts = row["min_ts"]
                max_ts = row["max_ts"]
                cnt = row["cnt"]
            else:
                sym, min_ts, max_ts, cnt = row
            out[sym] = (
                pd.to_datetime(min_ts, utc=True).to_pydatetime() if min_ts else None,
                pd.to_datetime(max_ts, utc=True).to_pydatetime() if max_ts else None,
                int(cnt or 0),
            )
        for sym in symbols:
            out.setdefault(sym, (None, None, 0))
        return out
    finally:
        if close_conn:
            conn.close()


def covers(start: datetime, end: datetime, cov_min, cov_max) -> bool:
    """Return True if coverage fully spans the requested window."""
    if cov_min is None or cov_max is None:
        return False
    return cov_min <= start and cov_max >= end


def missing_ranges(
    start: datetime, end: datetime, cov_min, cov_max
) -> List[Tuple[datetime, datetime]]:
    """Return list of (start,end) tuples not covered by existing data."""
    if cov_min is None or cov_max is None:
        return [(start, end)]
    ranges: List[Tuple[datetime, datetime]] = []
    if start < cov_min:
        ranges.append((start, cov_min))
    if cov_max < end:
        ranges.append((cov_max, end))
    return [(a, b) for a, b in ranges if a < b]


def load_bars(
    symbol: str, interval: str, start: datetime, end: datetime, conn=None
) -> pd.DataFrame:
    """Helper to load bars for a single symbol."""
    return get_prices_from_db([symbol], start, end, interval=interval, conn=conn)[
        symbol
    ]

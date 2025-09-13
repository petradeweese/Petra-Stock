"""Forward test creation and updates.

Provides helper functions used by the web routes and background scheduler
for creating and updating forward tests.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pandas as pd

from services.market_data import get_prices as md_get_prices
from services.market_data import window_from_lookback
from utils import TZ, now_et


def _window_to_minutes(value: float, unit: str) -> int:
    unit = (unit or "").lower()
    if unit.startswith("min"):
        return int(value)
    if unit.startswith("hour"):
        return int(value * 60)
    if unit.startswith("day"):
        return int(value * 60 * 24)
    if unit.startswith("week"):
        return int(value * 60 * 24 * 7)
    return int(value * 60)


def create_forward_test(
    db: sqlite3.Cursor, fav: Dict[str, Any], get_prices_fn=md_get_prices
) -> None:
    """Create a forward test record for the given favorite.

    The entry bar close is used as the reference price and the resulting
    forward test is queued for future evaluation.
    """
    start, end = window_from_lookback(fav.get("lookback_years", 1.0))
    data = get_prices_fn([fav["ticker"]], fav.get("interval", "15m"), start, end).get(
        fav["ticker"]
    )
    if data is None or getattr(data, "empty", True):
        return
    last_bar = data.iloc[-1]
    ts = last_bar.name
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    entry_ts = ts.astimezone(TZ).isoformat()
    entry_price = float(last_bar["Close"])
    window_minutes = _window_to_minutes(
        fav.get("window_value", 4.0), fav.get("window_unit", "Hours")
    )
    now_iso = now_et().isoformat()
    db.execute(
        """
        INSERT INTO forward_tests
            (fav_id, ticker, direction, interval, rule, entry_price,
             target_pct, stop_pct, window_minutes, status, roi_forward,
             hit_forward, dd_forward, last_run_at, next_run_at, runs_count,
             notes, created_at, updated_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0.0, NULL, 0.0,
             NULL, NULL, 0, NULL, ?, ?)
        """,
        (
            fav["id"],
            fav["ticker"],
            fav.get("direction", "UP"),
            fav.get("interval", "15m"),
            fav.get("rule"),
            entry_price,
            fav.get("target_pct", 1.0),
            fav.get("stop_pct", 0.5),
            window_minutes,
            entry_ts,
            now_iso,
        ),
    )
    db.connection.commit()


def update_forward_tests(db: sqlite3.Cursor, get_prices_fn=md_get_prices) -> None:
    """Advance all queued/running forward tests."""
    db.execute(
        """
        SELECT id, ticker, direction, interval, created_at, entry_price,
               target_pct, stop_pct, window_minutes, status
          FROM forward_tests
         WHERE status IN ('queued','running')
        """
    )
    rows = [dict(r) for r in db.fetchall()]
    for row in rows:
        now_iso = now_et().isoformat()
        try:
            db.execute(
                "UPDATE forward_tests SET status='running', last_run_at=?, "
                "updated_at=?, runs_count=runs_count+1 WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
            start, end = window_from_lookback(1.0)
            data = get_prices_fn([row["ticker"]], row["interval"], start, end).get(
                row["ticker"]
            )
            if data is None or getattr(data, "empty", True):
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            entry_ts = pd.Timestamp(row["created_at"])
            after = data[data.index > entry_ts]
            if after.empty:
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            prices = after["Close"]
            mult = 1.0 if row["direction"] == "UP" else -1.0
            pct_series = (prices / row["entry_price"] - 1.0) * 100 * mult
            roi = float(pct_series.iloc[-1])
            mae = float(pct_series.min())
            status = "ok"
            hit_pct = None
            if row["direction"] == "UP":
                hit_cond = prices >= row["entry_price"] * (1 + row["target_pct"] / 100)
                stop_cond = prices <= row["entry_price"] * (1 - row["stop_pct"] / 100)
            else:
                hit_cond = prices <= row["entry_price"] * (1 - row["target_pct"] / 100)
                stop_cond = prices >= row["entry_price"] * (1 + row["stop_pct"] / 100)
            hit_time = prices[hit_cond].index[0] if hit_cond.any() else None
            stop_time = prices[stop_cond].index[0] if stop_cond.any() else None
            expire_ts = entry_ts + pd.Timedelta(minutes=row["window_minutes"])
            final_ts = after.index[-1]
            if (
                hit_time
                and (not stop_time or hit_time <= stop_time)
                and hit_time <= expire_ts
            ):
                roi = float(pct_series.loc[hit_time])
                hit_pct = 100.0
            elif (
                stop_time
                and (not hit_time or stop_time < hit_time)
                and stop_time <= expire_ts
            ):
                roi = float(pct_series.loc[stop_time])
                hit_pct = 0.0
            elif final_ts < expire_ts:
                status = "queued"
            dd = float(max(0.0, -mae))
            db.execute(
                """
                UPDATE forward_tests
                   SET roi_forward=?, dd_forward=?, status=?, hit_forward=?,
                       last_run_at=?, next_run_at=?, updated_at=?
                 WHERE id=?
                """,
                (
                    roi,
                    dd,
                    status,
                    hit_pct,
                    now_et().isoformat(),
                    now_et().isoformat(),
                    now_iso,
                    row["id"],
                ),
            )
        except Exception:
            db.execute(
                "UPDATE forward_tests SET status='error', last_run_at=?, "
                "updated_at=? WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
    db.connection.commit()

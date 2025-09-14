"""Utility helpers for dealing with time and market sessions.

The function ``market_is_open`` prefers the
``pandas_market_calendars`` package to accurately account for New York Stock
Exchange (XNYS) holidays and special sessions.  If the package is unavailable
the function falls back to a simplified weekend/regular-hours check.
"""

from datetime import datetime, time, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal

    _XNYS = mcal.get_calendar("XNYS")
except Exception:  # pragma: no cover - fallback when dependency missing
    mcal = None
    _XNYS = None

# Eastern Time zone used by the New York Stock Exchange
TZ = ZoneInfo("America/New_York")

# Regular trading hours (local exchange time)
OPEN_TIME = time(9, 30)
CLOSE_TIME = time(16, 0)


def now_et() -> datetime:
    """Return the current time in Eastern Time."""
    return datetime.now(timezone.utc).astimezone(TZ)


def market_is_open(ts: Optional[datetime] = None) -> bool:
    """Return ``True`` if ``ts`` falls within a trading session."""

    ts = ts or now_et()

    # Ensure the timestamp is timezone aware then convert to exchange tz
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(TZ)

    if mcal and _XNYS:
        schedule = _XNYS.schedule(start_date=ts.date(), end_date=ts.date())
        if schedule.empty:
            return False
        row = schedule.iloc[0]
        if getattr(row, "name", None).date() != ts.date():
            return False
        open_dt = row["market_open"].to_pydatetime().astimezone(TZ)
        close_dt = row["market_close"].to_pydatetime().astimezone(TZ)
        return open_dt <= ts <= close_dt

    # Fallback: assume regular hours and no holidays
    if ts.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    open_dt = datetime.combine(ts.date(), OPEN_TIME, tzinfo=TZ)
    close_dt = datetime.combine(ts.date(), CLOSE_TIME, tzinfo=TZ)
    return open_dt <= ts <= close_dt


def last_trading_close(ts: Optional[datetime] = None) -> datetime:
    """Return the most recent market close at or before ``ts``."""

    ts = ts or now_et()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts_et = ts.astimezone(TZ)
    day = ts_et.date()
    while True:
        midday = datetime.combine(day, time(12, 0), tzinfo=TZ)
        if market_is_open(midday):
            close_dt = datetime.combine(day, CLOSE_TIME, tzinfo=TZ)
            if ts_et >= close_dt:
                return close_dt.astimezone(timezone.utc)
        day -= timedelta(days=1)


def clamp_market_closed(start: datetime, end: datetime) -> Tuple[datetime, bool]:
    """Clamp ``end`` to the last trading close when market is closed.

    Returns a tuple of (new_end, was_clamped).
    """

    if market_is_open(end):
        return end, False
    last_close = last_trading_close(end)
    return last_close, True

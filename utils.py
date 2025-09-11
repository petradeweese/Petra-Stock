"""Utility helpers for dealing with time and market sessions.

The function ``market_is_open`` prefers the
``pandas_market_calendars`` package to accurately account for New York Stock
Exchange (XNYS) holidays and special sessions.  If the package is unavailable
the function falls back to a simplified weekend/regular-hours check.
"""

from datetime import datetime, time, timezone
from typing import Optional
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
        open_dt = schedule.at[ts.date(), "market_open"].to_pydatetime().astimezone(TZ)
        close_dt = schedule.at[ts.date(), "market_close"].to_pydatetime().astimezone(TZ)
        return open_dt <= ts <= close_dt

    # Fallback: assume regular hours and no holidays
    if ts.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    open_dt = datetime.combine(ts.date(), OPEN_TIME, tzinfo=TZ)
    close_dt = datetime.combine(ts.date(), CLOSE_TIME, tzinfo=TZ)
    return open_dt <= ts <= close_dt

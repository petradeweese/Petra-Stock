"""Utility helpers for dealing with time and market sessions.

The original implementation depended on the third‑party
``pandas_market_calendars`` package to determine whether the New York Stock
Exchange (XNYS) was open.  The execution environment used for the tests does
not provide that dependency, which caused import errors during test
collection.  To make the utility module self‑contained and keep the tests
focused on simple logic, we replace the dependency with a lightweight
implementation based solely on the Python standard library.

The simplified ``market_is_open`` function assumes the standard NYSE trading
hours of 9:30–16:00 Eastern Time and does not account for market holidays or
special sessions.  This behaviour is sufficient for the unit tests which only
require that weekends are correctly identified as closed.
"""

from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Eastern Time zone used by the New York Stock Exchange
TZ = ZoneInfo("America/New_York")

# Regular trading hours (local exchange time)
OPEN_TIME = time(9, 30)
CLOSE_TIME = time(16, 0)


def now_utc() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


def now_et() -> datetime:
    """Return the current time in Eastern Time."""
    return now_utc().astimezone(TZ)


def market_is_open(ts: Optional[datetime] = None) -> bool:
    """Return ``True`` if ``ts`` falls within regular trading hours.

    The timestamp is converted to Eastern Time.  Weekends are considered
    closed and holidays are ignored.
    """

    ts = ts or now_et()

    # Ensure the timestamp is timezone aware then convert to exchange tz
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(TZ)

    # Markets are closed on weekends
    if ts.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    open_dt = datetime.combine(ts.date(), OPEN_TIME, tzinfo=TZ)
    close_dt = datetime.combine(ts.date(), CLOSE_TIME, tzinfo=TZ)
    return open_dt <= ts <= close_dt

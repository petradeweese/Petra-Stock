import contextvars
import datetime as dt
from contextlib import contextmanager
from typing import Dict, List, Optional

import pandas as pd

from prometheus_client import Histogram
from config import settings
from services.data_fetcher import fetch_prices as yahoo_fetch

from .data_provider import fetch_bars
from . import price_store
from .price_store import detect_gaps, get_prices_from_db
from utils import TZ, OPEN_TIME, CLOSE_TIME, market_is_open

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal

    _XNYS = mcal.get_calendar("XNYS")
except Exception:  # pragma: no cover - fallback when dependency missing
    mcal = None
    _XNYS = None

coverage_metric = Histogram(
    "data_coverage_ratio", "Ratio of available bars to expected"
)

DEFAULT_PROVIDER = settings.data_provider or "db"

_NOW_OVERRIDE: contextvars.ContextVar[Optional[dt.datetime]] = contextvars.ContextVar(
    "market_data_now_override", default=None
)


@contextmanager
def override_window_end(end: Optional[dt.datetime]):
    """Temporarily override the ``window_from_lookback`` end timestamp."""

    token = _NOW_OVERRIDE.set(end)
    try:
        yield
    finally:
        _NOW_OVERRIDE.reset(token)


def window_from_lookback(lookback_years: float) -> tuple[dt.datetime, dt.datetime]:
    override = _NOW_OVERRIDE.get()
    if override is not None:
        if override.tzinfo is None:
            override = override.replace(tzinfo=dt.timezone.utc)
        end = override.astimezone(dt.timezone.utc)
    else:
        end = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_years * 365))
    return start, end


def _interval_to_minutes(interval: str) -> int:
    """Translate interval strings like "15m" into minutes."""
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        # Treat one trading day as 24 hours; callers handle daily separately
        return int(interval[:-1]) * 24 * 60
    return 0


def _trading_minutes(start: dt.datetime, end: dt.datetime) -> float:
    """Return the number of trading minutes between ``start`` and ``end``."""
    if mcal and _XNYS:
        schedule = _XNYS.schedule(start_date=start.date(), end_date=end.date())
        total = 0.0
        for _, row in schedule.iterrows():
            open_dt = row["market_open"].to_pydatetime()
            close_dt = row["market_close"].to_pydatetime()
            if close_dt <= start or open_dt >= end:
                continue
            day_start = max(open_dt, start)
            day_end = min(close_dt, end)
            if day_end > day_start:
                total += (day_end - day_start).total_seconds() / 60
        return total

    # Fallback: assume regular hours and skip weekends
    start_et = start.astimezone(TZ)
    end_et = end.astimezone(TZ)
    total = 0.0
    day = start_et.date()
    while day <= end_et.date():
        midday = dt.datetime.combine(day, dt.time(13, 0), tzinfo=TZ)
        if market_is_open(midday):
            open_dt = dt.datetime.combine(day, OPEN_TIME, tzinfo=TZ)
            close_dt = dt.datetime.combine(day, CLOSE_TIME, tzinfo=TZ)
            day_start = max(open_dt, start_et)
            day_end = min(close_dt, end_et)
            if day_end > day_start:
                total += (day_end - day_start).total_seconds() / 60
        day += dt.timedelta(days=1)
    return total


def expected_bar_count(start: dt.datetime, end: dt.datetime, interval: str) -> int:
    """Estimate how many price bars should exist between ``start`` and ``end``."""
    interval = interval.strip().lower()
    if interval.endswith("d"):
        if mcal and _XNYS:
            schedule = _XNYS.schedule(start_date=start.date(), end_date=end.date())
            count = 0
            for _, row in schedule.iterrows():
                open_dt = row["market_open"].to_pydatetime()
                close_dt = row["market_close"].to_pydatetime()
                if close_dt > start and open_dt < end:
                    count += 1
            return count

        start_et = start.astimezone(TZ)
        end_et = end.astimezone(TZ)
        count = 0
        day = start_et.date()
        while day <= end_et.date():
            midday = dt.datetime.combine(day, dt.time(13, 0), tzinfo=TZ)
            if market_is_open(midday):
                count += 1
            day += dt.timedelta(days=1)
        return count

    minutes = _trading_minutes(start, end)
    interval_minutes = _interval_to_minutes(interval)
    if interval_minutes <= 0:
        return 0
    return int(minutes // interval_minutes)


def get_prices(
    symbols: List[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    provider: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider != "db":
        if provider == "yahoo":
            lookback_years = (end - start).days / 365.0
            return yahoo_fetch(symbols, interval, lookback_years)
        if provider == "schwab":
            return fetch_bars(symbols, interval, start, end)
    conn = None
    try:
        conn = price_store._open_conn() if hasattr(price_store, "_open_conn") else None
        cov = price_store.bulk_coverage(symbols, interval, start, end, conn=conn)
        expected = expected_bar_count(start, end, interval)
        to_check: List[str] = []
        for sym in symbols:
            cmin, cmax, cnt = cov.get(sym, (None, None, 0))
            if cnt >= expected and price_store.covers(start, end, cmin, cmax):
                coverage_metric.observe(1.0)
            else:
                to_check.append(sym)
        try:
            results = get_prices_from_db(symbols, start, end, interval=interval, conn=conn)
        except TypeError:  # backward compatibility
            results = get_prices_from_db(symbols, start, end)
        for sym in to_check:
            try:
                gaps = detect_gaps(sym, start, end, interval=interval, conn=conn)
            except TypeError:
                gaps = detect_gaps(sym, start, end)
            if gaps:
                from scheduler import queue_gap_fill

                queue_gap_fill(sym, start, end, interval)
            df = results.get(sym, pd.DataFrame())
            bars = len(df)
            ratio = bars / expected if expected else 0.0
            coverage_metric.observe(ratio)
        return results
    finally:
        if conn is not None:
            conn.close()


def fetch_prices(
    symbols: List[str],
    interval: str,
    lookback_years: float,
    provider: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    start, end = window_from_lookback(lookback_years)
    return get_prices(symbols, interval, start, end, provider=provider)

import datetime as dt
import os
from typing import Dict, List, Optional

import pandas as pd

from prometheus_client import Histogram
from services.data_fetcher import fetch_prices as yahoo_fetch

from .polygon_client import fetch_polygon_prices
from .price_store import detect_gaps, get_prices_from_db

coverage_metric = Histogram(
    "data_coverage_ratio", "Ratio of available bars to expected"
)

DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", os.getenv("PF_DATA_PROVIDER", "db"))


def window_from_lookback(lookback_years: float) -> tuple[dt.datetime, dt.datetime]:
    end = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_years * 365))
    return start, end


def _interval_to_freq(interval: str) -> str:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return f"{int(interval[:-1])}T"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}H"
    return "1D"


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
        if provider == "polygon":
            return fetch_polygon_prices(symbols, interval, start, end)
    results = get_prices_from_db(symbols, start, end)
    for sym in symbols:
        gaps = detect_gaps(sym, start, end)
        if gaps:
            from scheduler import queue_gap_fill

            queue_gap_fill(sym, start, end, interval)
        df = results.get(sym, pd.DataFrame())
        freq = _interval_to_freq(interval)
        expected = len(pd.date_range(start=start, end=end, freq=freq, inclusive="left"))
        bars = len(df)
        ratio = bars / expected if expected else 0.0
        coverage_metric.observe(ratio)
    return results


def fetch_prices(
    symbols: List[str],
    interval: str,
    lookback_years: float,
    provider: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    start, end = window_from_lookback(lookback_years)
    return get_prices(symbols, interval, start, end, provider=provider)

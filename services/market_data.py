import os
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd

from services.data_fetcher import fetch_prices as yahoo_fetch
from .polygon_client import fetch_polygon_prices
from .price_store import get_prices_from_db

DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", os.getenv("PF_DATA_PROVIDER", "db"))


def window_from_lookback(lookback_years: float) -> tuple[dt.datetime, dt.datetime]:
    end = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_years * 365))
    return start, end


def get_prices(symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime, provider: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "yahoo":
        lookback_years = (end - start).days / 365.0
        return yahoo_fetch(symbols, interval, lookback_years)
    if provider == "polygon":
        return fetch_polygon_prices(symbols, interval, start, end)
    # default: db
    return get_prices_from_db(symbols, start, end)


def fetch_prices(symbols: List[str], interval: str, lookback_years: float, provider: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    start, end = window_from_lookback(lookback_years)
    return get_prices(symbols, interval, start, end, provider=provider)

"""Yahoo Finance provider wrapper."""
from __future__ import annotations

import datetime as dt
from typing import Dict

import pandas as pd

from services.data_fetcher import fetch_prices as fetch_prices_batch
from services.errors import DataUnavailableError

_EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
RETURNS_ADJUSTED = True


def _ensure_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _lookback_years(start: dt.datetime, end: dt.datetime) -> float:
    delta = (end - start).total_seconds()
    if delta <= 0:
        return 1 / 365
    return max(delta / _SECONDS_PER_YEAR, 1 / 365)


def _trim_window(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask]


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df.reindex(columns=_EXPECTED_COLUMNS)
    return df


def fetch_prices(symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    lookback = _lookback_years(start_utc, end_utc)
    data: Dict[str, pd.DataFrame] = fetch_prices_batch([symbol], interval, lookback)
    df = data.get(symbol)
    if df is None or df.empty:
        raise DataUnavailableError("yahoo: no data returned")
    df = _trim_window(df, start_utc, end_utc)
    if df.empty:
        raise DataUnavailableError("yahoo: empty window")
    df = _finalise(df)
    if df.empty or not set(_EXPECTED_COLUMNS).issubset(df.columns):
        raise DataUnavailableError("yahoo: bad schema")
    return df[_EXPECTED_COLUMNS]


def fetch_corporate_actions(
    symbol: str, start: dt.datetime, end: dt.datetime
) -> pd.DataFrame:  # pragma: no cover - Yahoo already adjusted
    return pd.DataFrame()


__all__ = ["fetch_prices", "fetch_corporate_actions", "RETURNS_ADJUSTED"]

import os
import math
import time
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

TTL_CACHE = 600  # 10 minutes
CACHE_ROOT = Path(".cache/quotes")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
INTRADAY_CAPS = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
}

# Reuse a single session with retries
_SESSION = requests.Session()
retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429])
adapter = HTTPAdapter(max_retries=retry)
_SESSION.mount("https://", adapter)
_SESSION.mount("http://", adapter)
_TIMEOUT = (3, 20)  # connect, read


def _period_for(interval: str, lookback_years: float) -> str:
    if interval in INTRADAY_CAPS:
        return INTRADAY_CAPS[interval]
    years = max(1, int(math.ceil(lookback_years)))
    return f"{years}y"


def _cache_file(base: Path, ticker: str, period: str) -> Path:
    return base / f"{ticker}__{period}.parquet"


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < TTL_CACHE


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _download_all(tickers: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    attempt = 0
    while tickers and attempt < 3:
        attempt += 1
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
                session=_SESSION,
                timeout=_TIMEOUT,
            )
            if isinstance(data, pd.DataFrame) and len(tickers) == 1:
                data = {tickers[0]: data}
            for t in tickers:
                df = data.get(t, pd.DataFrame()) if isinstance(data, dict) else data[t]
                out[t] = _ensure_utc(df)
            break
        except Exception as e:  # pragma: no cover - network errors
            logger.warning("download attempt=%d error=%s", attempt, e)
            time.sleep(2 ** attempt)
    for t in tickers:
        out.setdefault(t, pd.DataFrame())
    return out


def fetch_prices(tickers: List[str], interval: str, lookback_years: float) -> Dict[str, pd.DataFrame]:
    """Fetch price data for tickers with caching and batching via yfinance."""
    period = _period_for(interval, lookback_years)
    date_dir = datetime.utcnow().strftime("%Y%m%d")
    base = CACHE_ROOT / interval / date_dir
    base.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    for t in tickers:
        path = _cache_file(base, t, period)
        if _is_fresh(path):
            try:
                df = pd.read_parquet(path)
                results[t] = _ensure_utc(df)
                continue
            except Exception:
                pass
        to_download.append(t)

    if to_download:
        fetched = _download_all(to_download, period, interval)
        for t, df in fetched.items():
            results[t] = df
            try:
                df.to_parquet(_cache_file(base, t, period))
            except Exception:
                pass

    for t in tickers:
        results.setdefault(t, pd.DataFrame())
    return results

import os
import math
import time
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import asyncio

import pandas as pd

from services import http_client
from services.price_utils import normalize_price_df

logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.getenv("PF_BATCH_SIZE", "25"))
TTL_INTRADAY = int(os.getenv("PF_TTL_INTRADAY", "300"))
TTL_DAILY = int(os.getenv("PF_TTL_DAILY", "3600"))
PER_TICKER_SLEEP = float(os.getenv("PF_PER_TICKER_SLEEP", "0.0"))

CACHE_DIR = Path(".cache/prices")
CACHE_SCHEMA = 2
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION_FILE = CACHE_DIR / "VERSION"


def _ensure_cache_schema() -> None:
    """Purge cache if schema version changed."""
    ver = None
    try:
        ver = int(_CACHE_VERSION_FILE.read_text().strip())
    except Exception:
        ver = None
    if ver != CACHE_SCHEMA:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_VERSION_FILE.write_text(str(CACHE_SCHEMA))


_ensure_cache_schema()

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

_MEM_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}


def _period_for(interval: str, lookback_years: float) -> str:
    if interval in INTRADAY_CAPS:
        return INTRADAY_CAPS[interval]
    years = max(1, int(math.ceil(lookback_years)))
    return f"{years}y"


def _cache_key(ticker: str, interval: str, period: str) -> Tuple[str, str, str]:
    return (ticker, interval, period)


def _cache_file(ticker: str, interval: str, period: str) -> Path:
    fname = f"{ticker}__{interval}__{period}.parquet"
    return CACHE_DIR / fname


def _cache_ttl(interval: str) -> int:
    return TTL_INTRADAY if interval in INTRADAY_INTERVALS else TTL_DAILY


def _is_fresh(path: Path, ttl: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _normalize_ohlcv(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop(columns=["timestamp"])
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adjclose": "Adj Close",
        }
    )
    return df.dropna(how="all")


async def _download_one(ticker: str, period: str, interval: str) -> Tuple[str, pd.DataFrame]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={period}"
        "&includePrePost=false&events=div%2Csplit"
    )
    try:
        data = await http_client.get_json(url)
    except Exception as e:  # pragma: no cover - network failure
        logger.error("download error %s: %r", ticker, e)
        return ticker, pd.DataFrame()
    result = (data or {}).get("chart", {}).get("result", [])
    if not result:
        return ticker, pd.DataFrame()
    r0 = result[0]
    ts = r0.get("timestamp", [])
    quote = r0.get("indicators", {}).get("quote", [{}])[0]
    adj_raw = r0.get("indicators", {}).get("adjclose", [{}])[0]
    adj_list = adj_raw.get("adjclose", []) if isinstance(adj_raw, dict) else adj_raw
    records = []
    for i, t in enumerate(ts):
        records.append(
            {
                "timestamp": t,
                "open": quote.get("open", [None])[i] if i < len(quote.get("open", [])) else None,
                "high": quote.get("high", [None])[i] if i < len(quote.get("high", [])) else None,
                "low": quote.get("low", [None])[i] if i < len(quote.get("low", [])) else None,
                "close": quote.get("close", [None])[i] if i < len(quote.get("close", [])) else None,
                "volume": quote.get("volume", [None])[i] if i < len(quote.get("volume", [])) else None,
                "adjclose": adj_list[i] if i < len(adj_list) else None,
            }
        )
    df = _normalize_ohlcv(records)
    return ticker, _ensure_utc(df)


async def _download_batch(batch: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    tasks = [_download_one(t, period, interval) for t in batch]
    results = await asyncio.gather(*tasks)
    out: Dict[str, pd.DataFrame] = {t: df for t, df in results}
    return out


def fetch_prices(tickers: List[str], interval: str, lookback_years: float) -> Dict[str, pd.DataFrame]:
    """Fetch price data for tickers with batching, caching, and retries."""
    period = _period_for(interval, lookback_years)
    ttl = _cache_ttl(interval)
    results: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    for t in tickers:
        key = _cache_key(t, interval, period)
        mem = _MEM_CACHE.get(key)
        if mem is not None and not getattr(mem, "empty", True):
            results[t] = mem.copy()
            logger.info("mem_cache_hit=%s", t)
            continue
        path = _cache_file(t, interval, period)
        if _is_fresh(path, ttl):
            try:
                df = pd.read_parquet(path)
                df = normalize_price_df(df)
                if df is not None:
                    df = _ensure_utc(df)
                    results[t] = df
                    _MEM_CACHE[key] = df
                    logger.info("cache_hit=%s", t)
                else:
                    logger.info("cache_miss=%s", t)
                    to_download.append(t)
                    continue
            except Exception:
                logger.info("cache_miss=%s", t)
                to_download.append(t)
        else:
            logger.info("cache_miss=%s", t)
            to_download.append(t)

    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i : i + BATCH_SIZE]
        fetched = asyncio.run(_download_batch(batch, period, interval))
        for t, df in fetched.items():
            df = normalize_price_df(df)
            if df is None:
                df = pd.DataFrame()
            else:
                df = _ensure_utc(df)
            results[t] = df
            key = _cache_key(t, interval, period)
            _MEM_CACHE[key] = df
            path = _cache_file(t, interval, period)
            if not df.empty:
                try:
                    df.to_parquet(path)
                except Exception:
                    pass
            if PER_TICKER_SLEEP > 0 and len(batch) == 1:
                time.sleep(PER_TICKER_SLEEP)

    for t in tickers:
        results.setdefault(t, pd.DataFrame())
    logger.info("fetched=%d cache=%d", len(to_download), len(tickers) - len(to_download))
    return results


def clear_mem_cache() -> None:
    """Clear the in-memory memoization cache."""
    _MEM_CACHE.clear()

import os
import math
import time
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.getenv("PF_BATCH_SIZE", "25"))
TTL_INTRADAY = int(os.getenv("PF_TTL_INTRADAY", "300"))
TTL_DAILY = int(os.getenv("PF_TTL_DAILY", "3600"))
MAX_RETRIES = int(os.getenv("PF_MAX_RETRIES", "5"))
PER_TICKER_SLEEP = float(os.getenv("PF_PER_TICKER_SLEEP", "0.0"))

CACHE_DIR = Path(".cache/prices")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


def _period_for(interval: str, lookback_years: float) -> str:
    if interval in INTRADAY_CAPS:
        return INTRADAY_CAPS[interval]
    years = max(1, int(math.ceil(lookback_years)))
    return f"{years}y"


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


def _download_batch(batch: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    attempt = 0
    while batch and attempt < MAX_RETRIES:
        attempt += 1
        logger.info("batch_size=%d attempt=%d", len(batch), attempt)
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=False,
                progress=False,
            )
            if isinstance(data, pd.DataFrame) and len(batch) == 1:
                data = {batch[0]: data}
            for t in batch:
                df = data.get(t, pd.DataFrame()) if isinstance(data, dict) else data[t]
                df = _ensure_utc(df)
                out[t] = df
            break
        except Exception as e:  # network errors
            code = getattr(getattr(e, "response", None), "status_code", None)
            retry_after = (
                getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
                if hasattr(e, "response")
                else None
            )
            wait = int(retry_after) if retry_after else 2 ** (attempt - 1)
            logger.info("backoff=%s wait=%ds", code or "error", wait)
            time.sleep(wait)
            if code == 429 and len(batch) > 1:
                half = max(1, len(batch) // 2)
                first = batch[:half]
                second = batch[half:]
                out.update(_download_batch(first, period, interval))
                batch = second
                attempt = 0
            if len(batch) == 1 and PER_TICKER_SLEEP > 0:
                time.sleep(PER_TICKER_SLEEP)
    for t in batch:
        out.setdefault(t, pd.DataFrame())
    return out


def fetch_prices(tickers: List[str], interval: str, lookback_years: float) -> Dict[str, pd.DataFrame]:
    """Fetch price data for tickers with batching, caching, and retries."""
    period = _period_for(interval, lookback_years)
    ttl = _cache_ttl(interval)
    results: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    for t in tickers:
        path = _cache_file(t, interval, period)
        if _is_fresh(path, ttl):
            try:
                df = pd.read_parquet(path)
                results[t] = _ensure_utc(df)
                logger.info("cache_hit=%s", t)
            except Exception:
                logger.info("cache_miss=%s", t)
                to_download.append(t)
        else:
            logger.info("cache_miss=%s", t)
            to_download.append(t)

    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i : i + BATCH_SIZE]
        fetched = _download_batch(batch, period, interval)
        for t, df in fetched.items():
            results[t] = df
            path = _cache_file(t, interval, period)
            try:
                df.to_parquet(path)
            except Exception:
                pass

    for t in tickers:
        results.setdefault(t, pd.DataFrame())
    logger.info("fetched=%d cache=%d", len(to_download), len(tickers) - len(to_download))
    return results

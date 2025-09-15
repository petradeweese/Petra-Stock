import asyncio
import datetime as dt
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from config import settings
from services import http_client

RUN_ID = os.getenv("RUN_ID", "")
logger = logging.getLogger(__name__)


def _add_run_id(record: logging.LogRecord) -> bool:
    setattr(record, "run_id", RUN_ID)
    return True


logger.addFilter(_add_run_id)
NY_TZ = ZoneInfo("America/New_York")


def _api_key() -> str:
    return os.getenv("POLYGON_API_KEY", "")


def _include_prepost() -> bool:
    return os.getenv("POLYGON_INCLUDE_PREPOST", "false").lower() == "true"


POLY_RPS = float(os.getenv("POLY_RPS", "0.08"))
POLY_BURST = int(os.getenv("POLY_BURST", "1"))

try:  # pragma: no cover - best effort
    http_client.set_rate_limit("api.polygon.io", POLY_RPS, POLY_BURST)
    http_client.set_concurrency("api.polygon.io", POLY_BURST)
    logger.info("polygon_rate_limit rps=%.2f burst=%d", POLY_RPS, POLY_BURST)
except Exception:
    pass


def _normalize_window(
    start: dt.datetime, end: dt.datetime
) -> Tuple[dt.datetime, dt.datetime, int, int]:
    """Return NY/UTC representations of the exact requested window."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)

    if end <= start:
        end = start + dt.timedelta(minutes=1)

    utc_start = start.astimezone(dt.timezone.utc)
    utc_end = end.astimezone(dt.timezone.utc)
    start_ms = int(utc_start.timestamp() * 1000)
    end_ms = int(utc_end.timestamp() * 1000)
    ny_start = utc_start.astimezone(NY_TZ)
    ny_end = utc_end.astimezone(NY_TZ)
    return ny_start, ny_end, start_ms, end_ms


async def _fetch_single(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    multiplier: int = 15,
    timespan: str = "minute",
) -> pd.DataFrame:
    api_key = _api_key()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    ny_start, ny_end, start_ms, end_ms = _normalize_window(start, end)
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
        f"{start_ms}/{end_ms}?adjusted=true&sort=asc&limit=50000"
    )
    url = f"{base_url}&apiKey={api_key}" if api_key else base_url
    log_url = url.replace(api_key, "***") if api_key else url
    logger.info("polygon_request symbol=%s url=%s", symbol, log_url)

    all_results = []
    next_url: Optional[str] = url
    pages = 0
    t0 = time.monotonic()

    async def _get_json(url: str) -> dict:
        attempt = 0
        while True:
            status: Optional[int] = None
            try:
                return await http_client.get_json(url, headers=headers)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status not in (429,) and status < 500:
                    raise
                err: Exception = e
            except httpx.RequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                err = e
            attempt += 1
            if attempt >= settings.fetch_retry_max:
                raise err
            wait = min(
                settings.fetch_retry_cap_ms,
                settings.fetch_retry_base_ms * (2 ** (attempt - 1)),
            )
            wait += random.randint(0, settings.fetch_retry_base_ms)
            logger.warning(
                "retry attempt=%d status=%s wait_ms=%d", attempt, status, wait
            )
            await asyncio.sleep(wait / 1000)

    while next_url:
        data = await _get_json(next_url)
        pages += 1
        if not data:
            break
        results = data.get("results", [])
        all_results.extend(results)
        next_url = data.get("next_url")
    records = []
    for r in all_results:
        ts = r.get("t")
        if ts is None:
            continue
        records.append(
            {
                "ts": pd.to_datetime(ts, unit="ms", utc=True),
                "Open": r.get("o"),
                "High": r.get("h"),
                "Low": r.get("l"),
                "Close": r.get("c"),
                "Volume": r.get("v"),
            }
        )
    if not records:
        logger.info(
            "polygon_fetch symbol=%s pages=%d rows=0 duration=%.2f",
            symbol,
            pages,
            time.monotonic() - t0,
        )
        logger.info(
            "polygon_window symbol=%s ny_start=%s ny_end=%s "
            "utc_start_ms=%d utc_end_ms=%d bars_returned=0",
            symbol,
            ny_start.isoformat(),
            ny_end.isoformat(),
            start_ms,
            end_ms,
        )
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(records).set_index("ts")
    # Align to NYSE session and return UTC timestamps
    ny = ZoneInfo("America/New_York")
    df = df.tz_convert(ny)
    if not _include_prepost():
        df = df.between_time("09:30", "16:00")
    df = df.tz_convert("UTC")
    duration = time.monotonic() - t0
    rows = len(df)
    logger.info(
        "polygon_fetch symbol=%s pages=%d rows=%d duration=%.2f",
        symbol,
        pages,
        rows,
        duration,
    )
    logger.info(
        "polygon_window symbol=%s ny_start=%s ny_end=%s "
        "utc_start_ms=%d utc_end_ms=%d bars_returned=%d",
        symbol,
        ny_start.isoformat(),
        ny_end.isoformat(),
        start_ms,
        end_ms,
        rows,
    )
    return df


async def fetch_polygon_prices_async(
    symbols: List[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
) -> Dict[str, pd.DataFrame]:
    if not _api_key():
        raise RuntimeError("POLYGON_API_KEY missing")
    multiplier = 15
    timespan = "minute"
    out: Dict[str, pd.DataFrame] = {}
    chunk = dt.timedelta(days=7)
    for sym in symbols:
        dfs: List[pd.DataFrame] = []
        cur = start
        while cur < end:
            nxt = min(cur + chunk, end)
            logger.info(
                "provider_range symbol=%s start=%s end=%s",
                sym,
                cur.isoformat(),
                nxt.isoformat(),
            )
            dfs.append(await _fetch_single(sym, cur, nxt, multiplier, timespan))
            cur = nxt

        if dfs:
            df = pd.concat(dfs).sort_index()
            df = df[~df.index.duplicated(keep="first")]
            out[sym] = df
        else:  # pragma: no cover - defensive
            out[sym] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return out


def fetch_polygon_prices(
    symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime
) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper around :func:`fetch_polygon_prices_async`.

    ``asyncio.run`` raises ``RuntimeError`` when invoked from an active event
    loop.  Older code paths may call this helper from within an async context,
    so guard against that situation and provide a clearer error message.  The
    async variant should be used directly when already inside an event loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – safe to use asyncio.run.
        return asyncio.run(fetch_polygon_prices_async(symbols, interval, start, end))

    raise RuntimeError(
        "fetch_polygon_prices() cannot be called from a running event loop; "
        "use fetch_polygon_prices_async() instead",
    )

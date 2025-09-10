import os
import datetime as dt
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import logging
import time
from zoneinfo import ZoneInfo

from services import http_client

logger = logging.getLogger(__name__)


def _api_key() -> str:
    return os.getenv("POLYGON_API_KEY", "")


def _include_prepost() -> bool:
    return os.getenv("POLYGON_INCLUDE_PREPOST", "false").lower() == "true"


POLY_RPS = float(os.getenv("POLY_RPS", "0.08"))
POLY_BURST = int(os.getenv("POLY_BURST", "1"))

try:  # pragma: no cover - best effort
    http_client.set_rate_limit("api.polygon.io", POLY_RPS, POLY_BURST)
    logger.info("polygon_rate_limit rps=%.2f burst=%d", POLY_RPS, POLY_BURST)
except Exception:
    pass


async def _fetch_single(symbol: str, start: dt.datetime, end: dt.datetime, multiplier: int = 15, timespan: str = "minute") -> pd.DataFrame:
    api_key = _api_key()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
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
    while next_url:
        data = await http_client.get_json(next_url, headers=headers)
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
        logger.info("polygon_fetch symbol=%s pages=%d rows=0 duration=%.2f", symbol, pages, time.monotonic()-t0)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(records).set_index("ts")
    # Align to NYSE session and return UTC timestamps
    ny = ZoneInfo("America/New_York")
    df = df.tz_convert(ny)
    if not _include_prepost():
        df = df.between_time("09:30", "16:00")
    df = df.tz_convert("UTC")
    logger.info(
        "polygon_fetch symbol=%s pages=%d rows=%d duration=%.2f",
        symbol,
        pages,
        len(df),
        time.monotonic() - t0,
    )
    return df


def fetch_polygon_prices(symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime) -> Dict[str, pd.DataFrame]:
    if not _api_key():
        raise RuntimeError("POLYGON_API_KEY missing")
    multiplier = 15
    timespan = "minute"
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = asyncio.run(_fetch_single(sym, start, end, multiplier, timespan))
        out[sym] = df
    return out

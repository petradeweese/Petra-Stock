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
from services import http_client, price_store
from utils import OPEN_TIME, TZ, last_trading_close, market_is_open

RUN_ID = os.getenv("RUN_ID", "")
logger = logging.getLogger(__name__)


def _add_run_id(record: logging.LogRecord) -> bool:
    setattr(record, "run_id", RUN_ID)
    return True


logger.addFilter(_add_run_id)
NY_TZ = ZoneInfo("America/New_York")

FIFTEEN_MIN = dt.timedelta(minutes=15)


def _current_session_start(now: dt.datetime) -> dt.datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    now_et = now.astimezone(TZ)
    if market_is_open(now):
        session_date = now_et.date()
    else:
        last_close = last_trading_close(now)
        session_date = last_close.astimezone(TZ).date()
    open_dt = dt.datetime.combine(session_date, OPEN_TIME, tzinfo=TZ)
    return open_dt.astimezone(dt.timezone.utc)


def _clone_timeout_ctx(timeout_ctx: Optional[dict]) -> Optional[dict]:
    if timeout_ctx is None:
        return None
    cloned = dict(timeout_ctx)
    deadline = cloned.get("deadline")
    if deadline is not None:
        cloned["remaining"] = max(0.0, deadline - time.monotonic())
    elif "remaining" in cloned:
        try:
            cloned["remaining"] = max(0.0, float(cloned["remaining"]))
        except (TypeError, ValueError):
            cloned.pop("remaining", None)
    return cloned


def _as_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def compute_request_window(
    symbol: str,
    interval: str,
    default_start: dt.datetime,
    default_end: dt.datetime,
    *,
    last_bar: Optional[dt.datetime] = None,
    now: Optional[dt.datetime] = None,
) -> Tuple[dt.datetime, dt.datetime, str]:
    if interval != "15m" or not settings.scan_minimal_near_now:
        return default_start, default_end, "range"

    if last_bar is None:
        try:
            _, last_bar = price_store.get_coverage(symbol, interval)
        except Exception:
            last_bar = None

    if last_bar is None:
        return default_start, default_end, "range"

    last_bar = _as_utc(last_bar)
    default_start_utc = _as_utc(default_start)
    default_end_utc = _as_utc(default_end)

    now_utc = _as_utc(now) if now is not None else dt.datetime.now(dt.timezone.utc)

    session_start = _current_session_start(now_utc)
    if last_bar < session_start:
        return default_start, default_end, "range"

    overlaps_session = default_end_utc > session_start
    starts_at_or_after_last_bar = default_start_utc >= last_bar
    extends_beyond_last_bar = default_end_utc > last_bar

    if not (overlaps_session and starts_at_or_after_last_bar and extends_beyond_last_bar):
        return default_start, default_end, "range"

    start = max(default_start_utc, last_bar)
    end = min(default_end_utc, start + FIFTEEN_MIN)
    if end <= start:
        end = start + FIFTEEN_MIN
    return start, end, "single_bucket"


def _api_key() -> str:
    return os.getenv("POLYGON_API_KEY", "")


def _include_prepost() -> bool:
    return os.getenv("POLYGON_INCLUDE_PREPOST", "false").lower() == "true"


# Allow SCAN_* overrides so ops can tune rate limits and concurrency without
# redeploying.  Defaults fall back to the legacy POLY_* values for backwards
# compatibility.
POLY_RPS = float(os.getenv("SCAN_RPS", os.getenv("POLY_RPS", "1.0")))
POLY_BURST = int(
    os.getenv("SCAN_MAX_CONCURRENCY", os.getenv("POLY_BURST", "2"))
)

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
    *,
    timeout_ctx: Optional[dict] = None,
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
                ctx = _clone_timeout_ctx(timeout_ctx)
                if ctx is None:
                    return await http_client.get_json(url, headers=headers)
                return await http_client.get_json(
                    url, headers=headers, timeout_ctx=ctx
                )
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
    *,
    timeout_ctx: Optional[dict] = None,
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
            fetch_kwargs = {}
            if timeout_ctx is not None:
                fetch_kwargs["timeout_ctx"] = timeout_ctx
            dfs.append(
                await _fetch_single(
                    sym, cur, nxt, multiplier, timespan, **fetch_kwargs
                )
            )
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

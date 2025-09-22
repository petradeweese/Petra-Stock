import asyncio
import datetime as dt
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import settings
from services import price_store
from services import schwab_client
from services.schwab_client import SchwabAPIError, SchwabAuthError
from utils import OPEN_TIME, TZ, last_trading_close, market_is_open

RUN_ID = os.getenv("RUN_ID", "")
logger = logging.getLogger(__name__)


def _add_run_id(record: logging.LogRecord) -> bool:
    setattr(record, "run_id", RUN_ID)
    return True


logger.addFilter(_add_run_id)
NY_TZ = ZoneInfo("America/New_York")

FIFTEEN_MIN = dt.timedelta(minutes=15)

EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


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


def _include_prepost() -> bool:
    return os.getenv("SCHWAB_INCLUDE_PREPOST", "false").lower() == "true"


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


def _align_to_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ny = ZoneInfo("America/New_York")
    if df.index.tz is None:
        df.index = df.index.tz_localize(dt.timezone.utc)
    else:
        df.index = df.index.tz_convert(dt.timezone.utc)
    df = df.tz_convert(ny)
    if not _include_prepost():
        df = df.between_time("09:30", "16:00")
    return df.tz_convert("UTC")


async def _fetch_single(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    interval: str,
    timeout_ctx: Optional[dict] = None,
) -> Tuple[pd.DataFrame, str]:
    ny_start, ny_end, start_ms, end_ms = _normalize_window(start, end)
    t0 = time.monotonic()
    attempts = 0
    errors: List[str] = []

    while attempts < max(1, settings.fetch_retry_max):
        attempts += 1
        try:
            ctx = _clone_timeout_ctx(timeout_ctx)
            df = await schwab_client.get_price_history(
                symbol,
                start,
                end,
                interval,
                timeout_ctx=ctx,
            )
            df = _align_to_session(df)
            duration = time.monotonic() - t0
            rows = len(df)
            status = schwab_client.last_status()
            logger.info(
                "schwab_fetch symbol=%s interval=%s rows=%d duration=%.2f status=%s",
                symbol,
                interval,
                rows,
                duration,
                status if status is not None else "unknown",
            )
            logger.info(
                "schwab_window symbol=%s ny_start=%s ny_end=%s "
                "utc_start_ms=%d utc_end_ms=%d bars_returned=%d",
                symbol,
                ny_start.isoformat(),
                ny_end.isoformat(),
                start_ms,
                end_ms,
                rows,
            )
            if rows:
                df.attrs["provider"] = "schwab"
                return df, "schwab"
            logger.warning(
                "schwab_fetch_empty symbol=%s interval=%s status=%s duration=%.2f",
                symbol,
                interval,
                status if status is not None else "unknown",
                duration,
            )
            errors.append("empty")
            break
        except (SchwabAPIError, SchwabAuthError) as exc:
            errors.append(str(exc))
            status = getattr(exc, "status_code", None)
            duration = time.monotonic() - t0
            logger.warning(
                "schwab_fetch_error symbol=%s interval=%s status=%s duration=%.2f err=%s",
                symbol,
                interval,
                status if status is not None else "unknown",
                duration,
                exc,
            )
            if attempts >= settings.fetch_retry_max:
                logger.warning(
                    "schwab_fetch_failed symbol=%s attempts=%d", symbol, attempts
                )
                break
            wait = min(
                settings.fetch_retry_cap_ms,
                settings.fetch_retry_base_ms * (2 ** (attempts - 1)),
            )
            wait += random.randint(0, settings.fetch_retry_base_ms)
            logger.warning(
                "schwab_retry symbol=%s attempt=%d wait_ms=%d", symbol, attempts, wait
            )
            await asyncio.sleep(wait / 1000)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            duration = time.monotonic() - t0
            logger.exception(
                "schwab_fetch_exception symbol=%s interval=%s duration=%.2f",
                symbol,
                interval,
                duration,
            )
            if attempts >= settings.fetch_retry_max:
                break
            wait = min(
                settings.fetch_retry_cap_ms,
                settings.fetch_retry_base_ms * (2 ** (attempts - 1)),
            )
            wait += random.randint(0, settings.fetch_retry_base_ms)
            logger.warning(
                "schwab_retry symbol=%s attempt=%d wait_ms=%d", symbol, attempts, wait
            )
            await asyncio.sleep(wait / 1000)

    df = await _fetch_yfinance(symbol, start, end, interval)
    df = _align_to_session(df)
    duration = time.monotonic() - t0
    rows = len(df)
    logger.info(
        "yfinance_fetch symbol=%s interval=%s rows=%d duration=%.2f reason=%s",
        symbol,
        interval,
        rows,
        duration,
        ";".join(errors) or "schwab_unavailable",
    )
    logger.info(
        "yfinance_window symbol=%s ny_start=%s ny_end=%s "
        "utc_start_ms=%d utc_end_ms=%d bars_returned=%d",
        symbol,
        ny_start.isoformat(),
        ny_end.isoformat(),
        start_ms,
        end_ms,
        rows,
    )
    provider = "yfinance" if rows else "none"
    df.attrs["provider"] = provider
    return df, provider


async def _fetch_yfinance(
    symbol: str, start: dt.datetime, end: dt.datetime, interval: str
) -> pd.DataFrame:
    def _download() -> pd.DataFrame:
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                prepost=_include_prepost(),
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("yfinance_error symbol=%s err=%s", symbol, exc)
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=1, axis=1)
            except Exception:
                df = df.droplevel(0, axis=1)

        df = df.rename(
            columns={
                "Adj Close": "Adj Close",
                "adjclose": "Adj Close",
            }
        )

        for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            if col not in df.columns:
                df[col] = None

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    return await asyncio.to_thread(_download)


async def _fetch_yfinance_quote(symbol: str) -> Dict[str, object]:
    def _quote() -> Dict[str, object]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("yfinance_quote_error symbol=%s err=%s", symbol, exc)
            return {}

        price = info.get("regularMarketPrice") or info.get("previousClose")
        if price is None:
            return {}
        ts_val = info.get("regularMarketTime")
        timestamp: Optional[dt.datetime] = None
        if ts_val:
            try:
                timestamp = dt.datetime.fromtimestamp(float(ts_val), dt.timezone.utc)
            except Exception:
                timestamp = None
        return {
            "symbol": symbol,
            "price": float(price),
            "timestamp": timestamp,
            "source": "yfinance",
        }

    return await asyncio.to_thread(_quote)


async def fetch_bars_async(
    symbols: List[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    chunk = dt.timedelta(days=7)
    for sym in symbols:
        dfs: List[pd.DataFrame] = []
        providers: List[str] = []
        cur = start
        while cur < end:
            nxt = min(cur + chunk, end)
            logger.info(
                "provider_range symbol=%s interval=%s start=%s end=%s",
                sym,
                interval,
                cur.isoformat(),
                nxt.isoformat(),
            )
            fetch_kwargs = {}
            if timeout_ctx is not None:
                fetch_kwargs["timeout_ctx"] = timeout_ctx
            chunk_df, provider = await _fetch_single(
                sym,
                cur,
                nxt,
                interval=interval,
                **fetch_kwargs,
            )
            dfs.append(chunk_df)
            providers.append(provider)
            cur = nxt

        if dfs:
            df = pd.concat(dfs).sort_index()
            df = df[~df.index.duplicated(keep="first")]
            unique = {p for p in providers if p and p != "none"}
            if not unique:
                source = "none"
            elif len(unique) == 1:
                source = unique.pop()
            else:
                source = "mixed"
            df.attrs["provider"] = source
            out[sym] = df
        else:  # pragma: no cover - defensive
            empty = pd.DataFrame(columns=EXPECTED_COLUMNS)
            empty.attrs["provider"] = "none"
            out[sym] = empty
    return out


def fetch_bars(
    symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime
) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper around :func:`fetch_bars_async`.

    ``asyncio.run`` raises ``RuntimeError`` when invoked from an active event
    loop.  Older code paths may call this helper from within an async context,
    so guard against that situation and provide a clearer error message.  The
    async variant should be used directly when already inside an event loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – safe to use asyncio.run.
        return asyncio.run(fetch_bars_async(symbols, interval, start, end))

    raise RuntimeError(
        "fetch_bars() cannot be called from a running event loop; use "
        "fetch_bars_async() instead",
    )


def _window_from_lookback(lookback: float | dt.timedelta) -> Tuple[dt.datetime, dt.datetime]:
    end = dt.datetime.now(dt.timezone.utc)
    if isinstance(lookback, dt.timedelta):
        start = end - lookback
    else:
        start = end - dt.timedelta(days=float(lookback))
    return start, end


async def get_bars_async(
    symbol: str,
    interval: str,
    lookback: float | dt.timedelta,
    *,
    timeout_ctx: Optional[dict] = None,
) -> pd.DataFrame:
    start, end = _window_from_lookback(lookback)
    data = await fetch_bars_async([symbol], interval, start, end, timeout_ctx=timeout_ctx)
    return data.get(symbol, pd.DataFrame())


def get_bars(
    symbol: str,
    interval: str,
    lookback: float | dt.timedelta,
    *,
    timeout_ctx: Optional[dict] = None,
) -> pd.DataFrame:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            get_bars_async(
                symbol,
                interval,
                lookback,
                timeout_ctx=timeout_ctx,
            )
        )
    raise RuntimeError(
        "get_bars() cannot be called from a running event loop; use "
        "get_bars_async() instead",
    )


async def get_quote_async(
    symbol: str,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Dict[str, object]:
    try:
        quote = await schwab_client.get_quote(symbol, timeout_ctx=timeout_ctx)
        if quote:
            logger.info(
                "quote_source symbol=%s provider=schwab price=%.4f",
                symbol,
                float(quote.get("price", 0.0)),
            )
            return quote
    except (SchwabAPIError, SchwabAuthError) as exc:
        status = getattr(exc, "status_code", None)
        logger.warning(
            "schwab_quote_error symbol=%s status=%s err=%s",
            symbol,
            status if status is not None else "unknown",
            exc,
        )

    fallback = await _fetch_yfinance_quote(symbol)
    if fallback:
        logger.info(
            "quote_source symbol=%s provider=yfinance price=%.4f",
            symbol,
            float(fallback.get("price", 0.0)),
        )
    return fallback


def get_quote(
    symbol: str,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Dict[str, object]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(get_quote_async(symbol, timeout_ctx=timeout_ctx))
    raise RuntimeError(
        "get_quote() cannot be called from a running event loop; use "
        "get_quote_async() instead",
    )

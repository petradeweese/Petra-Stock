import asyncio
import datetime as dt
import logging
import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from prometheus_client import Counter, Histogram  # type: ignore

from config import settings
import db
from services import price_store
from services import schwab_client
from services.schwab_client import (
    HTTP_400_DISABLE_SECONDS,
    SchwabAPIError,
    SchwabAuthError,
)
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

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m"}
INTRADAY_LOOKBACK = dt.timedelta(days=59)


def _normalize_reason_label(value: Optional[str]) -> str:
    if not value:
        return "schwab_unavailable"
    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cleaned)
    safe = safe.strip("_")
    return safe or "schwab_unavailable"


schwab_fallback_total = Counter(
    "schwab_fallback_total",
    "Times Schwab data fetches fell back to alternate providers",
)
_schwab_fallback_reason_counters: Dict[str, Counter] = {}
schwab_fallback_rate = Counter(
    "schwab_fallback_rate_total",
    "Counter of Schwab fallbacks for rate calculations",
)


_AUTH_COOLDOWN_UNTIL: float = 0.0
_AUTH_COOLDOWN_REASON: str = ""
_AUTH_WARN_KEY: str = ""


def _auth_cooldown_state() -> Tuple[bool, Optional[str], float]:
    remaining = max(0.0, _AUTH_COOLDOWN_UNTIL - time.monotonic())
    if remaining > 0:
        return True, _AUTH_COOLDOWN_REASON or None, remaining
    return False, None, 0.0


def _start_auth_cooldown(
    reason: str,
    *,
    status: Optional[int],
    detail: str,
    ttl: float,
) -> None:
    global _AUTH_COOLDOWN_UNTIL, _AUTH_COOLDOWN_REASON, _AUTH_WARN_KEY
    ttl = max(0.0, ttl)
    now = time.monotonic()
    _AUTH_COOLDOWN_UNTIL = max(_AUTH_COOLDOWN_UNTIL, now + ttl)
    _AUTH_COOLDOWN_REASON = reason
    warn_key = f"{reason}:{status if status is not None else 'unknown'}"
    if _AUTH_WARN_KEY != warn_key:
        logger.warning(
            "schwab_auth_failure reason=%s status=%s detail=%s ttl=%.2f",
            reason,
            status if status is not None else "unknown",
            detail or "unknown",
            ttl,
        )
        _AUTH_WARN_KEY = warn_key


def _clear_auth_cooldown() -> None:
    global _AUTH_COOLDOWN_UNTIL, _AUTH_COOLDOWN_REASON, _AUTH_WARN_KEY
    _AUTH_COOLDOWN_UNTIL = 0.0
    _AUTH_COOLDOWN_REASON = ""
    _AUTH_WARN_KEY = ""


def _auth_backoff_seconds() -> float:
    try:
        value = float(settings.schwab_refresh_backoff_seconds or 0)
    except (TypeError, ValueError):
        value = 0.0
    return max(1.0, value)


def _increment_fallback_reason(reason: str) -> None:
    key = _normalize_reason_label(reason)
    counter = _schwab_fallback_reason_counters.get(key)
    if counter is None:
        counter = Counter(
            f"schwab_fallback_{key}_total",
            "Times Schwab data fetches fell back grouped by reason",
        )
        _schwab_fallback_reason_counters[key] = counter
    counter.inc()
schwab_retry_seconds = Histogram(
    "schwab_retry_time_seconds",
    "Time spent attempting Schwab before falling back",
)


def _ensure_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _normalize_bounds(
    start: dt.datetime, end: dt.datetime
) -> Tuple[dt.datetime, dt.datetime]:
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    if end_utc <= start_utc:
        end_utc = start_utc + dt.timedelta(minutes=1)
    return start_utc, end_utc


def _parse_db_timestamp(raw: object) -> Optional[dt.datetime]:
    if isinstance(raw, dt.datetime):
        return _ensure_utc(raw)
    if isinstance(raw, str):
        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _interval_to_freq(interval: str) -> Optional[str]:
    if not interval:
        return None
    value = interval.strip().lower()
    if value.endswith("m"):
        try:
            minutes = int(value[:-1] or "0")
        except ValueError:
            return None
        minutes = max(1, minutes)
        return f"{minutes}min"
    if value.endswith("h"):
        try:
            hours = int(value[:-1] or "0")
        except ValueError:
            return None
        hours = max(1, hours)
        return f"{hours * 60}min"
    if value.endswith("d"):
        try:
            days = int(value[:-1] or "0")
        except ValueError:
            return None
        days = max(1, days)
        return f"{days}D"
    return None


def _bars_to_dataframe(bars: Sequence[Dict[str, object]]) -> pd.DataFrame:
    if not bars:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df.index = pd.DatetimeIndex([], tz="UTC")
        return df

    sorted_rows = sorted(bars, key=lambda row: _ensure_utc(row["ts"]))  # type: ignore[arg-type]
    index = pd.DatetimeIndex([_ensure_utc(row["ts"]) for row in sorted_rows], tz="UTC")  # type: ignore[index]
    data = {
        "Open": [float(row.get("open", 0.0)) for row in sorted_rows],
        "High": [float(row.get("high", 0.0)) for row in sorted_rows],
        "Low": [float(row.get("low", 0.0)) for row in sorted_rows],
        "Close": [float(row.get("close", 0.0)) for row in sorted_rows],
        "Adj Close": [float(row.get("close", 0.0)) for row in sorted_rows],
        "Volume": [float(row.get("volume", 0.0)) for row in sorted_rows],
    }
    df = pd.DataFrame(data, index=index)
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return df


def _fetch_from_db(
    symbol: str, interval: str, start: dt.datetime, end: dt.datetime
) -> List[Dict[str, object]]:
    if end <= start:
        return []
    conn = db.get_engine().raw_connection()
    try:
        cursor = conn.execute(
            (
                "SELECT ts, open, high, low, close, volume "
                "FROM bars WHERE symbol=? AND interval=? AND ts>=? AND ts<? ORDER BY ts"
            ),
            (symbol.upper(), interval, start.isoformat(), end.isoformat()),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    results: List[Dict[str, object]] = []
    for row in rows:
        ts = _parse_db_timestamp(row[0])
        if ts is None:
            continue
        results.append(
            {
                "ts": ts,
                "open": float(row[1]) if row[1] is not None else None,
                "high": float(row[2]) if row[2] is not None else None,
                "low": float(row[3]) if row[3] is not None else None,
                "close": float(row[4]) if row[4] is not None else None,
                "volume": float(row[5]) if row[5] is not None else 0.0,
            }
        )
    return results


async def _fetch_from_provider(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Tuple[List[Dict[str, object]], str]:
    df, provider = await _fetch_single(
        symbol,
        start,
        end,
        interval=interval,
        timeout_ctx=timeout_ctx,
    )
    bars: List[Dict[str, object]] = []
    if not df.empty:
        for ts, row in df.iterrows():
            bars.append(
                {
                    "ts": _ensure_utc(ts.to_pydatetime()),
                    "open": float(row.get("Open", 0.0)),
                    "high": float(row.get("High", 0.0)),
                    "low": float(row.get("Low", 0.0)),
                    "close": float(row.get("Close", 0.0)),
                    "volume": float(row.get("Volume", 0.0)),
                }
            )
    return bars, provider


def _persist_bars(
    symbol: str, interval: str, bars: Sequence[Dict[str, object]]
) -> Tuple[int, int]:
    if not bars:
        return 0, 0
    symbol_norm = symbol.upper()
    conn = db.get_engine().raw_connection()
    try:
        cursor = conn.cursor()
        ts_values = [
            _ensure_utc(bar["ts"]).isoformat()  # type: ignore[arg-type]
            for bar in bars
        ]
        min_ts = min(ts_values)
        max_ts = max(ts_values)
        existing_rows = cursor.execute(
            (
                "SELECT ts FROM bars WHERE symbol=? AND interval=? AND ts>=? AND ts<=?"
            ),
            (symbol_norm, interval, min_ts, max_ts),
        ).fetchall()
        existing = {row[0] for row in existing_rows}

        rows_to_write: List[Tuple[object, ...]] = []
        inserted = 0
        replaced = 0
        for bar, ts_iso in zip(bars, ts_values):
            if ts_iso in existing:
                replaced += 1
            else:
                inserted += 1
            rows_to_write.append(
                (
                    symbol_norm,
                    interval,
                    ts_iso,
                    float(bar.get("open", 0.0)),
                    float(bar.get("high", 0.0)),
                    float(bar.get("low", 0.0)),
                    float(bar.get("close", 0.0)),
                    float(bar.get("volume", 0.0)),
                )
            )
        if rows_to_write:
            cursor.executemany(
                """
                INSERT INTO bars(symbol, interval, ts, open, high, low, close, volume)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, interval, ts) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
                """,
                rows_to_write,
            )
            conn.commit()
        return inserted, replaced
    finally:
        conn.close()


def _missing_ranges(
    bars: Sequence[Dict[str, object]],
    start: dt.datetime,
    end: dt.datetime,
    interval: str,
) -> List[Tuple[dt.datetime, dt.datetime]]:
    freq = _interval_to_freq(interval)
    if freq is None:
        return [(start, end)] if end > start else []
    index = pd.date_range(start=start, end=end, freq=freq, inclusive="left", tz="UTC")
    have = {_ensure_utc(row["ts"]) for row in bars}  # type: ignore[arg-type]
    gaps: List[Tuple[dt.datetime, dt.datetime]] = []
    gap_start: Optional[dt.datetime] = None
    for ts in index:
        ts_val = _ensure_utc(ts.to_pydatetime())
        if ts_val in have:
            if gap_start is not None:
                gaps.append((gap_start, ts_val))
                gap_start = None
        else:
            if gap_start is None:
                gap_start = ts_val
    if gap_start is not None and end > gap_start:
        gaps.append((gap_start, end))
    return gaps


async def _fetch_intraday_range(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Tuple[List[Dict[str, object]], str]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    cutoff = now_utc - INTRADAY_LOOKBACK
    symbol_norm = symbol.upper()

    initial_rows = _fetch_from_db(symbol_norm, interval, start, end)
    if end <= cutoff:
        if initial_rows:
            logger.info(
                "db_only_intraday symbol=%s interval=%s rows=%d",
                symbol_norm,
                interval,
                len(initial_rows),
            )
            return initial_rows, "db"
        logger.info(
            "db_only_intraday symbol=%s interval=%s rows=%d reason=out_of_lookback",
            symbol_norm,
            interval,
            len(initial_rows),
        )
        return initial_rows, "db"

    recent_start = max(start, cutoff)
    recent_rows = [row for row in initial_rows if row["ts"] >= recent_start]
    gaps = _missing_ranges(recent_rows, recent_start, end, interval)

    total_inserted = 0
    total_replaced = 0
    provider_used = "db"
    if gaps:
        aggregated_rows: List[Dict[str, object]] = []
        for gap_start, gap_end in gaps:
            if gap_end <= gap_start:
                continue
            bars, provider = await _fetch_from_provider(
                symbol_norm,
                interval,
                gap_start,
                gap_end,
                timeout_ctx=_clone_timeout_ctx(timeout_ctx),
            )
            if provider and provider != "none":
                provider_used = provider
            if not bars:
                continue
            filtered = [
                bar
                for bar in bars
                if gap_start <= _ensure_utc(bar["ts"]) < gap_end  # type: ignore[arg-type]
            ]
            aggregated_rows.extend(filtered)
        if aggregated_rows:
            inserted, replaced = _persist_bars(symbol_norm, interval, aggregated_rows)
            total_inserted += inserted
            total_replaced += replaced
            if inserted or replaced:
                logger.info(
                    "schwab_persist symbol=%s interval=%s inserted=%d replaced=%d",
                    symbol_norm,
                    interval,
                    inserted,
                    replaced,
                )
            initial_rows = _fetch_from_db(symbol_norm, interval, start, end)
        else:
            provider_used = provider_used or "none"

    if gaps and (total_inserted or total_replaced):
        old_rows = [row for row in initial_rows if row["ts"] < recent_start]
        new_rows = [row for row in initial_rows if row["ts"] >= recent_start]
        logger.info(
            "db+schwab_intraday symbol=%s interval=%s old_rows=%d new_rows=%d saved=%d",
            symbol_norm,
            interval,
            len(old_rows),
            len(new_rows),
            total_inserted + total_replaced,
        )
    elif gaps:
        logger.info(
            "db_only_intraday symbol=%s interval=%s rows=%d",
            symbol_norm,
            interval,
            len(initial_rows),
        )
    elif start >= cutoff and not gaps:
        logger.info(
            "db_only_intraday symbol=%s interval=%s rows=%d",
            symbol_norm,
            interval,
            len(initial_rows),
        )
    elif start < cutoff and not gaps:
        logger.info(
            "db_only_intraday symbol=%s interval=%s rows=%d",
            symbol_norm,
            interval,
            len(initial_rows),
        )

    return initial_rows, provider_used or "db"


async def _fetch_higher_interval_range(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Tuple[List[Dict[str, object]], str]:
    bars, provider = await _fetch_from_provider(
        symbol,
        interval,
        start,
        end,
        timeout_ctx=timeout_ctx,
    )
    filtered = [
        bar
        for bar in bars
        if start <= _ensure_utc(bar["ts"]) < end  # type: ignore[arg-type]
    ]
    bars = filtered
    if bars:
        inserted, replaced = _persist_bars(symbol, interval, bars)
        if inserted or replaced:
            logger.info(
                "schwab_persist symbol=%s interval=%s inserted=%d replaced=%d",
                symbol.upper(),
                interval,
                inserted,
                replaced,
            )
    final_rows = _fetch_from_db(symbol, interval, start, end)
    if final_rows:
        return final_rows, provider or "db"
    return bars, provider or "none"


async def _fetch_range(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Tuple[List[Dict[str, object]], str]:
    start_utc, end_utc = _normalize_bounds(start, end)
    if interval.strip().lower() in INTRADAY_INTERVALS:
        return await _fetch_intraday_range(
            symbol,
            interval,
            start_utc,
            end_utc,
            timeout_ctx=timeout_ctx,
        )
    return await _fetch_higher_interval_range(
        symbol,
        interval,
        start_utc,
        end_utc,
        timeout_ctx=timeout_ctx,
    )


async def fetch_range_async(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> List[Dict[str, object]]:
    rows, _ = await _fetch_range(
        symbol,
        interval,
        start,
        end,
        timeout_ctx=timeout_ctx,
    )
    return rows


def fetch_range(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
) -> List[Dict[str, object]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            fetch_range_async(
                symbol,
                interval,
                start,
                end,
                timeout_ctx=timeout_ctx,
            )
        )
    raise RuntimeError(
        "fetch_range() cannot be called from a running event loop; use "
        "fetch_range_async() instead",
    )


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
        trimmed = df.between_time("09:30", "16:00")
        if not trimmed.empty:
            df = trimmed
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
    error_labels: List[str] = []

    async def _finish_with_fallback(reason_hint: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        lost = time.monotonic() - t0
        reason_label = reason_hint or (error_labels[-1] if error_labels else "")
        fallback_reason = _normalize_reason_label(reason_label)
        skip_network = fallback_reason in {
            "auth_error",
            "schwab_disabled",
        }
        if skip_network:
            df = pd.DataFrame(columns=EXPECTED_COLUMNS)
            df.index = pd.DatetimeIndex([], tz="UTC")
        else:
            df = await _fetch_yfinance(symbol, start, end, interval)
            df = _align_to_session(df)
        duration = time.monotonic() - t0
        rows = len(df)
        reason_text = ";".join(errors) or "schwab_unavailable"
        logger.info(
            "yfinance_fetch symbol=%s interval=%s rows=%d duration=%.2f reason=%s "
            "ny_start=%s ny_end=%s utc_start_ms=%d utc_end_ms=%d",
            symbol,
            interval,
            rows,
            duration,
            reason_text,
            ny_start.isoformat(),
            ny_end.isoformat(),
            start_ms,
            end_ms,
        )
        provider = "yfinance" if rows and not skip_network else "none"
        df.attrs["provider"] = provider
        schwab_fallback_total.inc()
        _increment_fallback_reason(fallback_reason)
        schwab_fallback_rate.inc()
        schwab_retry_seconds.observe(lost)
        return df, provider

    cooldown_active, cooldown_reason, cooldown_remaining = _auth_cooldown_state()
    if cooldown_active:
        reason = cooldown_reason or "schwab_disabled"
        errors.append(reason)
        error_labels.append(reason)
        logger.info(
            "schwab_auth_cooldown_skip symbol=%s interval=%s reason=%s remaining=%.2f",
            symbol,
            interval,
            reason,
            cooldown_remaining,
        )
        return await _finish_with_fallback(reason)

    disabled, disabled_reason, disabled_status, _ = schwab_client.disabled_state()
    if disabled:
        reason = disabled_reason or "schwab_disabled"
        errors.append(reason)
        error_labels.append(reason)
        logger.info(
            "schwab_disabled_skip symbol=%s interval=%s reason=%s status=%s",
            symbol,
            interval,
            reason,
            disabled_status if disabled_status is not None else "unknown",
        )
        return await _finish_with_fallback(reason)

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
                "schwab_fetch symbol=%s interval=%s rows=%d duration=%.2f status=%s "
                "ny_start=%s ny_end=%s utc_start_ms=%d utc_end_ms=%d",
                symbol,
                interval,
                rows,
                duration,
                status if status is not None else "unknown",
                ny_start.isoformat(),
                ny_end.isoformat(),
                start_ms,
                end_ms,
            )
            if rows:
                _clear_auth_cooldown()
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
            error_labels.append("empty")
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
            if isinstance(exc, SchwabAuthError):
                reason_label = (exc.error_code or "auth_error").strip() or "auth_error"
                if status == 400:
                    reason_label = "http_400"
                error_labels.append(reason_label)
                base_ttl = _auth_backoff_seconds()
                ttl = max(
                    base_ttl,
                    float(HTTP_400_DISABLE_SECONDS) if status == 400 else base_ttl,
                )
                _start_auth_cooldown(
                    reason_label,
                    status=status,
                    detail=str(exc),
                    ttl=ttl,
                )
                schwab_client.disable(
                    reason=reason_label, status_code=status, ttl=ttl, error=exc
                )
                break
            if status == 400:
                error_labels.append("http_400")
                ttl = max(_auth_backoff_seconds(), float(HTTP_400_DISABLE_SECONDS))
                _start_auth_cooldown(
                    "http_400",
                    status=status,
                    detail=str(exc),
                    ttl=ttl,
                )
                schwab_client.disable(
                    reason="http_400", status_code=status, ttl=ttl, error=None
                )
                break
            if status is not None and 400 <= status < 500:
                error_labels.append(f"http_{status}")
                break
            error_labels.append("error")
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
            error_labels.append("exception")
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

    return await _finish_with_fallback()


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


class FetchResult(dict):
    """Dictionary-like container that also tracks fetch statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.ok: int = 0
        self.err: int = 0
        self.errors: Dict[str, Exception] = {}
        self.elapsed: float = 0.0


async def fetch_bars_async(
    symbols: List[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout_ctx: Optional[dict] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> FetchResult:
    total = len(symbols)
    result = FetchResult()
    if total == 0:
        return result

    try:
        max_concurrency = int(os.getenv("SCANNER_MAX_CONCURRENCY", "8"))
    except ValueError:
        max_concurrency = 8
    if max_concurrency <= 0:
        max_concurrency = 1

    semaphore = asyncio.Semaphore(max_concurrency)
    pending = total
    started = time.perf_counter()
    logger.info(
        "data_provider fetch_async_start symbols=%d interval=%s",
        total,
        interval,
    )

    async def _guarded(symbol: str) -> Tuple[str, pd.DataFrame, Optional[Exception]]:
        nonlocal pending
        provider_used = "none"
        try:
            async with semaphore:
                rows, provider_used = await _fetch_range(
                    symbol,
                    interval,
                    start,
                    end,
                    timeout_ctx=_clone_timeout_ctx(timeout_ctx),
                )
            df = _bars_to_dataframe(rows)
            provider_value = provider_used or ("db" if not df.empty else "none")
            df.attrs["provider"] = provider_value
            return symbol, df, None
        except Exception as exc:
            logger.exception(
                "data_provider fetch_async_error symbol=%s interval=%s",
                symbol,
                interval,
            )
            empty = _bars_to_dataframe([])
            empty.attrs["provider"] = provider_used or "error"
            return symbol, empty, exc
        finally:
            pending = max(0, pending - 1)
            if progress_cb:
                try:
                    progress_cb(symbol, pending, total)
                except Exception:
                    logger.exception("data_provider progress callback failed")

    tasks = [asyncio.create_task(_guarded(sym)) for sym in symbols]
    gathered = await asyncio.gather(*tasks)

    ok = 0
    err = 0
    for symbol, df, exc in gathered:
        result[symbol] = df
        if exc is None:
            ok += 1
        else:
            err += 1
            result.errors[symbol] = exc

    result.ok = ok
    result.err = err
    result.elapsed = time.perf_counter() - started
    logger.info(
        "data_provider fetch_async_done symbols=%d ok=%d err=%d elapsed=%.2fs",
        total,
        ok,
        err,
        result.elapsed,
    )
    return result


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

import contextvars
import datetime as dt
import logging
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd

from prometheus_client import Histogram
from config import ADJUST_BARS, DATA_PROVIDERS_PRIMARY, SCHWAB_ENABLED, SESSION
from services.data_fetcher import fetch_prices as yahoo_fetch  # legacy alias for tests
from services.errors import DataUnavailableError
from services.price_utils import normalize_price_df
from services.providers import (
    schwab as schwab_p,
    schwab_options,
    yahoo as yahoo_p,
    yahoo_options,
)
from services.providers.schwab_options import OptionsUnavailableError

from .polygon_client import fetch_polygon_prices
from . import price_store
from .price_store import detect_gaps, get_prices_from_db
from utils import TZ, OPEN_TIME, CLOSE_TIME, market_is_open

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal

    _XNYS = mcal.get_calendar("XNYS")
except Exception:  # pragma: no cover - fallback when dependency missing
    mcal = None
    _XNYS = None

coverage_metric = Histogram(
    "data_coverage_ratio", "Ratio of available bars to expected"
)

log = logging.getLogger(__name__)

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

_MAX_LATENCY_SAMPLES = 120
_HEALTH_WINDOW_SECONDS = 60.0
_PROVIDER_LATENCIES: dict[str, deque[float]] = {
    "schwab": deque(maxlen=_MAX_LATENCY_SAMPLES),
    "yahoo": deque(maxlen=_MAX_LATENCY_SAMPLES),
}
_PROVIDER_CALLS: dict[str, deque[float]] = {
    "schwab": deque(),
    "yahoo": deque(),
}
_PROVIDER_LAST_ERROR: dict[str, Optional[str]] = {"schwab": None, "yahoo": None}

_OPTIONS_COLUMNS = list(schwab_options.EXPECTED_COLUMNS)
_OPTION_ALIASES = {
    key: list(value) for key, value in schwab_options.ALIAS_MAP.items()
}
_OPTION_TYPE_MAP = dict(schwab_options.TYPE_MAP)
_OPTION_NUMERIC = list(schwab_options.NUMERIC_COLUMNS)


def _coerce_option_timestamp(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(float)
        mask = values > 1e11
        values = values.where(~mask, values / 1000.0)
        series = pd.Series(values, index=series.index)
    return pd.to_datetime(series, utc=True, errors="coerce")


def _coerce_option_expiry(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    values = series.astype(str).str.split(":").str[0]
    return pd.to_datetime(values, utc=True, errors="coerce")


def _record_provider_success(name: str, start_monotonic: float) -> None:
    duration = max(time.monotonic() - start_monotonic, 0.0)
    latencies = _PROVIDER_LATENCIES.setdefault(
        name, deque(maxlen=_MAX_LATENCY_SAMPLES)
    )
    latencies.append(duration)
    calls = _PROVIDER_CALLS.setdefault(name, deque())
    now = time.monotonic()
    calls.append(now)
    while calls and now - calls[0] > _HEALTH_WINDOW_SECONDS:
        calls.popleft()
    _PROVIDER_LAST_ERROR[name] = None


def _record_provider_failure(name: str, err: Exception) -> None:
    _PROVIDER_LAST_ERROR[name] = str(err)


def _retry_fetch(fetch_fn, *args, **kwargs):
    delay = 0.4
    last_err: Optional[DataUnavailableError] = None
    for attempt in range(1, 4):
        try:
            return fetch_fn(*args, **kwargs)
        except DataUnavailableError as exc:
            last_err = exc
            status = getattr(exc, "status", None)
            if status in TRANSIENT_STATUSES and attempt < 3:
                time.sleep(delay + random.uniform(0, 0.2))
                delay *= 2
                continue
            raise
    if last_err is not None:
        raise last_err
    raise DataUnavailableError("retry failed")


def normalize_corporate_actions(
    df: pd.DataFrame, actions: pd.DataFrame
) -> pd.DataFrame:
    if df.empty or actions is None:
        return df
    if not isinstance(actions, pd.DataFrame):
        actions = pd.DataFrame(actions)
    if actions.empty:
        return df

    actions = actions.copy()
    rename_map: dict[str, str] = {}
    for column in list(actions.columns):
        key = str(column).lower()
        if key in {"type", "eventtype", "action", "actiontype"}:
            rename_map[column] = "type"
        elif key in {
            "effective",
            "date",
            "executiondate",
            "effectivedate",
            "recorddate",
            "paymentdate",
        }:
            rename_map[column] = "effective"
        elif key in {"ratio", "split_ratio", "splitratio"}:
            rename_map[column] = "ratio"
        elif key in {"numerator", "splitnumerator"}:
            rename_map[column] = "numerator"
        elif key in {"denominator", "splitdenominator"}:
            rename_map[column] = "denominator"
    if rename_map:
        actions = actions.rename(columns=rename_map)

    if "effective" not in actions.columns:
        return df

    actions["effective"] = pd.to_datetime(
        actions["effective"], utc=True, errors="coerce"
    )
    actions = actions.dropna(subset=["effective"])
    if actions.empty:
        return df

    def _split_ratio(row: pd.Series) -> Optional[float]:
        for key in ("ratio",):
            if key in row and pd.notna(row[key]):
                value = row[key]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    if isinstance(value, str):
                        cleaned = value.replace(" ", "")
                        for sep in ("/", ":"):
                            if sep in cleaned:
                                parts = cleaned.split(sep)
                                if len(parts) == 2:
                                    try:
                                        num = float(parts[0])
                                        den = float(parts[1])
                                        if den:
                                            return num / den
                                    except (TypeError, ValueError, ZeroDivisionError):
                                        continue
                    continue
        num = row.get("numerator")
        den = row.get("denominator")
        if pd.notna(num) and pd.notna(den):
            try:
                num_f = float(num)
                den_f = float(den)
                if den_f:
                    return num_f / den_f
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        return None

    df_adj = df.copy()
    if isinstance(df_adj.index, pd.DatetimeIndex) and df_adj.index.tz is None:
        df_adj.index = df_adj.index.tz_localize("UTC")
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols + ["Volume"]:
        if col in df_adj.columns:
            df_adj[col] = pd.to_numeric(df_adj[col], errors="coerce")

    actions = actions.sort_values("effective")
    for _, action in actions.iterrows():
        action_type = str(action.get("type", "")).lower()
        if "split" not in action_type:
            continue
        ratio = _split_ratio(action)
        if ratio is None or ratio <= 0:
            continue
        effective = action["effective"]
        mask = df_adj.index < effective
        if not mask.any():
            continue
        df_adj.loc[mask, price_cols] = df_adj.loc[mask, price_cols] / ratio
        df_adj.loc[mask, "Volume"] = df_adj.loc[mask, "Volume"].fillna(0) * ratio

    return df_adj


def maybe_apply_corporate_actions(
    df: pd.DataFrame,
    provider_mod: Any,
    provider_name: str,
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    if df.empty or not ADJUST_BARS:
        return df
    if getattr(provider_mod, "RETURNS_ADJUSTED", False):
        return df
    fetch_actions = getattr(provider_mod, "fetch_corporate_actions", None)
    if fetch_actions is None:
        return df
    try:
        actions = fetch_actions(symbol, start, end)
    except DataUnavailableError as err:
        log.info(
            "provider_actions_failed provider=%s symbol=%s err=%s",
            provider_name,
            symbol,
            err,
        )
        return df
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "provider_actions_error provider=%s symbol=%s err=%s",
            provider_name,
            symbol,
            exc,
        )
        return df

    if actions is None:
        return df
    if not isinstance(actions, pd.DataFrame):
        actions = pd.DataFrame(actions)
    if actions.empty:
        return df
    try:
        adjusted = normalize_corporate_actions(df, actions)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "provider_actions_apply_failed provider=%s symbol=%s err=%s",
            provider_name,
            symbol,
            exc,
        )
        return df
    return adjusted


def filter_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    session_normalized = (session or "RTH").strip().upper()
    if session_normalized == "ETH":
        return df
    if session_normalized not in {"RTH", "ETH"}:
        session_normalized = "RTH"

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    idx_et = df.index.tz_convert(TZ)
    if len(idx_et) <= 1:
        return df
    if not idx_et.normalize().duplicated().any():
        return df
    indexer = idx_et.indexer_between_time(
        OPEN_TIME, CLOSE_TIME, include_start=True, include_end=True
    )
    if len(indexer) == len(df):
        return df
    return df.iloc[indexer]


def get_provider_health() -> Dict[str, Dict[str, Optional[float]]]:
    snapshot: Dict[str, Dict[str, Optional[float]]] = {}
    now = time.monotonic()
    for name in set(_PROVIDER_LAST_ERROR) | set(_PROVIDER_LATENCIES):
        latencies = list(_PROVIDER_LATENCIES.setdefault(name, deque(maxlen=_MAX_LATENCY_SAMPLES)))
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        calls = _PROVIDER_CALLS.setdefault(name, deque())
        while calls and now - calls[0] > _HEALTH_WINDOW_SECONDS:
            calls.popleft()
        snapshot[name] = {
            "avg_latency": avg_latency,
            "rpm": int(len(calls)),
            "last_error": _PROVIDER_LAST_ERROR.get(name),
        }
    return snapshot


def _fetch_with_fallback(symbol: str, interval: str, start: dt.datetime, end: dt.datetime):
    last_err: Optional[DataUnavailableError] = None
    providers = {"schwab": schwab_p, "yahoo": yahoo_p}
    for name in DATA_PROVIDERS_PRIMARY:
        if name == "schwab" and not SCHWAB_ENABLED:
            continue
        provider = providers.get(name)
        if provider is None:
            continue
        try:
            start_monotonic = time.monotonic()
            fetch = provider.fetch_prices
            if name == "schwab":
                df = _retry_fetch(fetch, symbol, interval, start, end)
            else:
                df = fetch(symbol, interval, start, end)
            df_norm = normalize_price_df(df)
            if df_norm is None:
                raise DataUnavailableError(f"{name}: empty frame")
            expected = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in expected if col not in df_norm.columns]
            if missing:
                raise DataUnavailableError(f"{name}: missing columns {missing}")
            df_norm = df_norm.reindex(columns=expected)
            df_norm = maybe_apply_corporate_actions(
                df_norm, provider, name, symbol, start, end
            )
            df_norm = df_norm.reindex(columns=expected)
            df_norm = filter_session(df_norm, SESSION)
            df_norm = df_norm.reindex(columns=expected)
            if df_norm.empty:
                raise DataUnavailableError(f"{name}: empty frame")
            _record_provider_success(name, start_monotonic)
            log.info(
                "provider_ok provider=%s symbol=%s interval=%s rows=%d",
                name,
                symbol,
                interval,
                len(df_norm),
            )
            return df_norm
        except DataUnavailableError as err:
            _record_provider_failure(name, err)
            log.info(
                "provider_failed provider=%s symbol=%s interval=%s err=%s",
                name,
                symbol,
                interval,
                err,
            )
            last_err = err
            continue
    raise DataUnavailableError(
        f"all providers failed for {symbol} {interval}: {last_err}"
    )


def normalize_options_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise OptionsUnavailableError("options: invalid frame")
    if df.empty:
        raise OptionsUnavailableError("options: empty frame")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    else:
        df = df.copy()

    rename_map: dict[str, str] = {}
    for target, aliases in _OPTION_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    for column in _OPTIONS_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    if df["symbol"].isna().all():
        df["symbol"] = symbol
    if df["underlying"].isna().all():
        df["underlying"] = symbol

    df["type"] = (
        df["type"].astype(str).str.lower().map(_OPTION_TYPE_MAP).fillna(pd.NA)
    )

    for column in _OPTION_NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "last" in df.columns:
        missing_last = df["last"].isna()
        if missing_last.any():
            df.loc[missing_last, "last"] = (
                df.loc[missing_last, ["bid", "ask"]].sum(axis=1) / 2
            )

    if "iv" in df.columns:
        df.loc[df["iv"] > 1.5, "iv"] = df.loc[df["iv"] > 1.5, "iv"] / 100.0

    df["updated_at"] = _coerce_option_timestamp(df["updated_at"])
    df["expiry"] = _coerce_option_expiry(df["expiry"])

    df = df.dropna(subset=["strike", "type", "bid", "ask"], how="any")
    df = df.dropna(subset=["expiry", "updated_at"], how="any")
    if df.empty:
        raise OptionsUnavailableError("options: no usable rows")

    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df["updated_at"] = df["updated_at"].dt.tz_convert("UTC")
    df["expiry"] = df["expiry"].dt.tz_convert("UTC")

    df = df[_OPTIONS_COLUMNS]
    df = df.sort_values(["expiry", "type", "strike"], kind="mergesort")
    df = df.reset_index(drop=True)
    return df


def get_options_chain(symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
    try:
        df_raw = schwab_options.fetch_chain(symbol, expiry)
        provider = "schwab"
    except OptionsUnavailableError as err:
        log.info(
            "options_provider_failed provider=%s symbol=%s err=%s",
            "schwab",
            symbol,
            err,
        )
        df_raw = yahoo_options.fetch_chain(symbol, expiry)
        provider = "yahoo"

    df_norm = normalize_options_df(df_raw, symbol)
    log.info(
        "options_provider_ok provider=%s symbol=%s rows=%d",
        provider,
        symbol,
        len(df_norm),
    )
    return df_norm


DEFAULT_PROVIDER = os.getenv("DATA_PROVIDER", os.getenv("PF_DATA_PROVIDER", "db"))

_NOW_OVERRIDE: contextvars.ContextVar[Optional[dt.datetime]] = contextvars.ContextVar(
    "market_data_now_override", default=None
)


@contextmanager
def override_window_end(end: Optional[dt.datetime]):
    """Temporarily override the ``window_from_lookback`` end timestamp."""

    token = _NOW_OVERRIDE.set(end)
    try:
        yield
    finally:
        _NOW_OVERRIDE.reset(token)


def window_from_lookback(lookback_years: float) -> tuple[dt.datetime, dt.datetime]:
    override = _NOW_OVERRIDE.get()
    if override is not None:
        if override.tzinfo is None:
            override = override.replace(tzinfo=dt.timezone.utc)
        end = override.astimezone(dt.timezone.utc)
    else:
        end = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_years * 365))
    return start, end


def _interval_to_minutes(interval: str) -> int:
    """Translate interval strings like "15m" into minutes."""
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        # Treat one trading day as 24 hours; callers handle daily separately
        return int(interval[:-1]) * 24 * 60
    return 0


def _trading_minutes(start: dt.datetime, end: dt.datetime) -> float:
    """Return the number of trading minutes between ``start`` and ``end``."""
    if mcal and _XNYS:
        schedule = _XNYS.schedule(start_date=start.date(), end_date=end.date())
        total = 0.0
        for _, row in schedule.iterrows():
            open_dt = row["market_open"].to_pydatetime()
            close_dt = row["market_close"].to_pydatetime()
            if close_dt <= start or open_dt >= end:
                continue
            day_start = max(open_dt, start)
            day_end = min(close_dt, end)
            if day_end > day_start:
                total += (day_end - day_start).total_seconds() / 60
        return total

    # Fallback: assume regular hours and skip weekends
    start_et = start.astimezone(TZ)
    end_et = end.astimezone(TZ)
    total = 0.0
    day = start_et.date()
    while day <= end_et.date():
        midday = dt.datetime.combine(day, dt.time(13, 0), tzinfo=TZ)
        if market_is_open(midday):
            open_dt = dt.datetime.combine(day, OPEN_TIME, tzinfo=TZ)
            close_dt = dt.datetime.combine(day, CLOSE_TIME, tzinfo=TZ)
            day_start = max(open_dt, start_et)
            day_end = min(close_dt, end_et)
            if day_end > day_start:
                total += (day_end - day_start).total_seconds() / 60
        day += dt.timedelta(days=1)
    return total


def expected_bar_count(start: dt.datetime, end: dt.datetime, interval: str) -> int:
    """Estimate how many price bars should exist between ``start`` and ``end``."""
    interval = interval.strip().lower()
    if interval.endswith("d"):
        if mcal and _XNYS:
            schedule = _XNYS.schedule(start_date=start.date(), end_date=end.date())
            count = 0
            for _, row in schedule.iterrows():
                open_dt = row["market_open"].to_pydatetime()
                close_dt = row["market_close"].to_pydatetime()
                if close_dt > start and open_dt < end:
                    count += 1
            return count

        start_et = start.astimezone(TZ)
        end_et = end.astimezone(TZ)
        count = 0
        day = start_et.date()
        while day <= end_et.date():
            midday = dt.datetime.combine(day, dt.time(13, 0), tzinfo=TZ)
            if market_is_open(midday):
                count += 1
            day += dt.timedelta(days=1)
        return count

    minutes = _trading_minutes(start, end)
    interval_minutes = _interval_to_minutes(interval)
    if interval_minutes <= 0:
        return 0
    return int(minutes // interval_minutes)


def get_prices(
    symbols: List[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    provider: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider != "db":
        if provider == "yahoo":
            lookback_years = (end - start).days / 365.0
            return yahoo_fetch(symbols, interval, lookback_years)
        if provider == "schwab":
            results: Dict[str, pd.DataFrame] = {}
            for sym in symbols:
                results[sym] = _fetch_with_fallback(sym, interval, start, end)
            return results
        if provider == "polygon":
            return fetch_polygon_prices(symbols, interval, start, end)
    conn = None
    try:
        conn = price_store._open_conn() if hasattr(price_store, "_open_conn") else None
        cov = price_store.bulk_coverage(symbols, interval, start, end, conn=conn)
        expected = expected_bar_count(start, end, interval)
        to_check: List[str] = []
        for sym in symbols:
            cmin, cmax, cnt = cov.get(sym, (None, None, 0))
            if cnt >= expected and price_store.covers(start, end, cmin, cmax):
                coverage_metric.observe(1.0)
            else:
                to_check.append(sym)
        try:
            results = get_prices_from_db(symbols, start, end, interval=interval, conn=conn)
        except TypeError:  # backward compatibility
            results = get_prices_from_db(symbols, start, end)
        for sym in to_check:
            try:
                gaps = detect_gaps(sym, start, end, interval=interval, conn=conn)
            except TypeError:
                gaps = detect_gaps(sym, start, end)
            if gaps:
                from scheduler import queue_gap_fill

                queue_gap_fill(sym, start, end, interval)
            df = results.get(sym, pd.DataFrame())
            bars = len(df)
            ratio = bars / expected if expected else 0.0
            coverage_metric.observe(ratio)
        return results
    finally:
        if conn is not None:
            conn.close()


def fetch_prices(
    symbols: List[str],
    interval: str,
    lookback_years: float,
    provider: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    start, end = window_from_lookback(lookback_years)
    return get_prices(symbols, interval, start, end, provider=provider)

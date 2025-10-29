"""Feature construction helpers for intraday forecast matching."""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import settings
from services import price_store
from services import schwab_client
from services.data_provider import fetch_bars
from services.async_utils import run_coro
from utils import OPEN_TIME, TZ


logger = logging.getLogger(__name__)


MINUTE_INTERVAL = "5m"

# (frame label, pandas frequency string, number of bars used for feature window)
FRAME_ORDER: List[Tuple[str, str, int]] = [
    ("5m", "5min", 24),
    ("30m", "30min", 6),
    ("1d", "1D", 20),
]

FEATURE_ORDER = [
    "rsi",
    "delta",
    "gamma",
    "slope",
    "volume_z",
    "iv_rank",
]


@dataclass
class FrameResult:
    """Structured results for an aggregated timeframe."""

    frame: str
    freq: str
    bars: int
    raw: Dict[str, float]
    zscores: Dict[str, float]
    count: int


@dataclass
class CachedBars:
    """Container for cached multi-resolution bar data."""

    start: dt.datetime
    end: dt.datetime
    data: pd.DataFrame
    source: str
    aggregated: Dict[str, pd.DataFrame]
    stored: bool = False
    updated: float = field(default_factory=time.monotonic)


LOOKBACK_DAYS = 60


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


_TOKEN_STATUS_LOGGED = False

_BAR_CACHE: Dict[Tuple[str, str], CachedBars] = {}
_IV_RANK_CACHE: Dict[str, Tuple[Optional[float], float]] = {}
_IV_RANK_TTL = 15 * 60.0  # seconds


def _token_path_status() -> tuple[bool, str | None, str | None]:
    path_raw = getattr(settings, "schwab_token_path", "")
    if not path_raw:
        return False, "missing", None
    candidate = Path(path_raw).expanduser()
    resolved = str(candidate)
    try:
        exists = candidate.exists()
    except OSError:
        exists = False
    if not exists:
        return False, "missing", resolved
    if not candidate.is_file():
        return False, "not_file", resolved
    if not os.access(candidate, os.R_OK):
        return False, "unreadable", resolved
    return True, None, resolved


def _log_token_path_status() -> None:
    global _TOKEN_STATUS_LOGGED
    if _TOKEN_STATUS_LOGGED:
        return
    _TOKEN_STATUS_LOGGED = True
    readable, reason, path = _token_path_status()
    if readable:
        logger.info(
            "forecast schwab_token_path_ok path=%s",
            path or getattr(settings, "schwab_token_path", ""),
        )
    elif reason == "missing":
        logger.info(
            "forecast schwab_token_path_unset",
        )
    else:
        logger.warning(
            "forecast schwab_token_path_unavailable reason=%s path=%s",
            reason or "unknown",
            path,
        )


def _resample_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    else:
        working.index = working.index.tz_convert("UTC")
    if freq.upper().endswith("D"):
        working = working.tz_convert(TZ)
        aggregated = (
            working.resample(freq, label="right", closed="right")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(subset=["Close"])
        )
        aggregated.index = aggregated.index.tz_convert("UTC")
    else:
        aggregated = (
            working.resample(freq, label="right", closed="right")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(subset=["Close"])
        )
    aggregated["Adj Close"] = aggregated["Close"]
    return aggregated


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, math.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _linear_slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return math.nan
    base = arr[0]
    if base == 0:
        base = 1.0
    normalized = arr / base - 1.0
    x = np.arange(normalized.size, dtype=float)
    x_centered = x - x.mean()
    denominator = np.sum(x_centered ** 2)
    if denominator == 0:
        return 0.0
    slope = float(np.dot(x_centered, normalized) / denominator)
    return slope


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    if series.empty or window <= 1:
        return pd.Series(index=series.index, dtype=float)
    return series.rolling(window=window, min_periods=max(2, window // 2)).apply(
        _linear_slope, raw=False
    )


def _option_field(contract: object, *names: str) -> object:
    for name in names:
        if hasattr(contract, name):
            value = getattr(contract, name)
            if value is not None:
                return value
        if isinstance(contract, dict) and name in contract:
            value = contract[name]
            if value is not None:
                return value
    return None


def _current_iv_rank(ticker: str) -> Tuple[Optional[float], str]:
    key = ticker.upper()
    cached = _IV_RANK_CACHE.get(key)
    now = time.monotonic()
    if cached and cached[1] > now:
        value = cached[0]
        return (value if value is not None else None), "cache"

    try:  # pragma: no cover - optional dependency
        from services import options_provider
    except Exception:
        _IV_RANK_CACHE[key] = (None, now + 300.0)
        return None, "unavailable"

    try:  # pragma: no cover - defensive
        chain = options_provider.get_chain(key)
    except Exception:
        _IV_RANK_CACHE[key] = (None, now + 300.0)
        return None, "unavailable"

    best: Optional[Tuple[Tuple[float, float, float], float]] = None
    for contract in chain or []:
        iv_rank_raw = _option_field(contract, "iv_rank", "ivRank", "ivr", "ivRankPercentile")
        try:
            iv_rank_val = float(iv_rank_raw)
        except (TypeError, ValueError):
            continue

        delta_raw = _option_field(contract, "delta")
        try:
            delta_val = abs(float(delta_raw))
        except (TypeError, ValueError):
            delta_val = 0.5
        distance = abs(delta_val - 0.5)

        dte_raw = _option_field(
            contract,
            "dte",
            "days_to_expiration",
            "daysToExpiration",
        )
        try:
            dte_val = abs(float(dte_raw))
        except (TypeError, ValueError):
            dte_val = 999.0

        oi_raw = _option_field(contract, "open_interest", "openInterest")
        try:
            oi_val = -float(oi_raw)
        except (TypeError, ValueError):
            oi_val = -0.0

        score = (distance, dte_val, oi_val)
        if best is None or score < best[0]:
            best = (score, iv_rank_val)

    iv_rank = best[1] if best else None
    _IV_RANK_CACHE[key] = (iv_rank, now + _IV_RANK_TTL)
    return iv_rank, "options"


def _load_cached_history(
    symbol: str, start: dt.datetime, end: dt.datetime
) -> Optional[CachedBars]:
    key = (symbol.upper(), MINUTE_INTERVAL)
    cached = _BAR_CACHE.get(key)

    fetch_start = ensure_utc(start)
    fetch_end = ensure_utc(end)

    if cached is not None:
        if cached.start <= fetch_start and cached.end >= fetch_end and not cached.data.empty:
            return cached
        fetch_start = min(fetch_start, cached.start)
        fetch_end = max(fetch_end, cached.end)

    frame, source = load_price_frame(symbol, fetch_start, fetch_end, MINUTE_INTERVAL)
    if frame is None or frame.empty:
        return cached

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    if cached is not None:
        combined = pd.concat([cached.data, frame]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        source_parts = {
            part
            for part in (cached.source or "", source or "")
            if part and part != "none"
        }
        aggregated = dict(cached.aggregated)
        stored = cached.stored
        cache = CachedBars(
            min(fetch_start, cached.start),
            max(fetch_end, cached.end),
            combined,
            "|".join(sorted(source_parts)) or cached.source or source or "none",
            aggregated,
            stored=stored,
        )
    else:
        cache = CachedBars(
            fetch_start,
            fetch_end,
            frame,
            source or "none",
            {},
        )

    _BAR_CACHE[key] = cache
    return cache


def _ensure_aggregated_frames(symbol: str, cache: CachedBars) -> None:
    missing = any(frame not in cache.aggregated for frame, _, _ in FRAME_ORDER)
    if not missing and cache.aggregated:
        return

    aggregated: Dict[str, pd.DataFrame] = {}
    base = cache.data
    for frame, freq, _ in FRAME_ORDER:
        if frame == "5m":
            aggregated[frame] = base.copy()
        else:
            aggregated[frame] = _resample_frequency(base, freq)
    cache.aggregated = aggregated

    if not cache.stored:
        try:
            thirty = aggregated.get("30m")
            daily = aggregated.get("1d")
            if thirty is not None and not thirty.empty:
                price_store.upsert_bars(symbol, thirty, interval="30m")
            if daily is not None and not daily.empty:
                price_store.upsert_bars(symbol, daily, interval="1d")
            cache.stored = True
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("forecast store_aggregated_failed ticker=%s", symbol)

    logger.info(
        "forecast multiresolution_ready ticker=%s frames=%s rows=%d source=%s",
        symbol.upper(),
        ",".join(sorted(aggregated)),
        len(base),
        cache.source or "none",
    )


def load_price_frame(
    symbol: str, start: dt.datetime, end: dt.datetime, interval: str
) -> Tuple[pd.DataFrame | None, str]:
    start_utc = ensure_utc(start)
    end_utc = ensure_utc(end)

    def _clip(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.loc[(frame.index >= start_utc) & (frame.index < end_utc)]

    def _has_coverage(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        last_ts = frame.index.max()
        if interval.endswith("m"):
            try:
                minutes = int(interval[:-1])
            except ValueError:
                minutes = 1
            if minutes < 1:
                minutes = 1
            expected_last = end_utc - pd.Timedelta(minutes=minutes)
        else:
            expected_last = end_utc - pd.Timedelta(days=1)
        if last_ts < expected_last:
            return False
        try:
            gaps = price_store.detect_gaps(
                symbol,
                start_utc,
                end_utc,
                interval=interval,
            )
        except Exception:
            return True
        return len(gaps) == 0

    initial = price_store.get_prices_from_db([symbol], start_utc, end_utc, interval=interval)
    df = _clip(initial.get(symbol, pd.DataFrame()).copy())
    source = "db" if not df.empty else "none"

    if _has_coverage(df):
        return df, source

    _log_token_path_status()
    allow_network = getattr(settings, "forecast_allow_network", True)
    if not allow_network:
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=network_disabled symbol=%s",
            symbol,
        )
        return None, "db_miss"

    readable, reason, token_path = _token_path_status()
    disabled, disabled_reason, _, disabled_error = schwab_client.disabled_state()
    reason_label = (disabled_reason or "").lower()
    config_disabled = False
    if disabled:
        SchwabAuthError = getattr(schwab_client, "SchwabAuthError", None)
        if "config" in reason_label:
            config_disabled = True
        elif SchwabAuthError is not None and isinstance(disabled_error, SchwabAuthError):
            message = str(disabled_error or "").lower()
            if "missing" in message or "config" in message:
                config_disabled = True
    if not readable and reason not in (None, "missing"):
        logger.warning(
            "forecast schwab_token_path_unavailable reason=%s path=%s disabled_reason=%s",
            reason or "unknown",
            token_path,
            disabled_reason or "",
        )
        if config_disabled:
            logger.warning(
                "forecast schwab_fallback_skipped_or_failed reason=token_%s symbol=%s token_path=%s",
                reason or "unknown",
                symbol,
                token_path,
            )
            return None, "db_miss"

    client_obj = getattr(schwab_client, "_client", None)
    ensure_coro = None
    if client_obj is not None and hasattr(client_obj, "_ensure_token"):
        ensure_coro = client_obj._ensure_token()
    try:
        if ensure_coro is not None:
            run_coro(ensure_coro)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=token_refresh_failed symbol=%s err=%s",
            symbol,
            str(exc),
        )
        return None, "db_miss"

    try:
        fetched_map = fetch_bars([symbol], interval, start_utc, end_utc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=fetch_error symbol=%s err=%s",
            symbol,
            str(exc),
        )
        return None, "db_miss"

    fetched = _clip(fetched_map.get(symbol, pd.DataFrame()).copy())
    provider = str(getattr(fetched, "attrs", {}).get("provider") or "")
    if fetched.empty:
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=fetch_empty symbol=%s",
            symbol,
        )
        return None, "db_miss"

    price_store.clear_cache()
    refreshed = price_store.get_prices_from_db([symbol], start_utc, end_utc, interval=interval)
    df = _clip(refreshed.get(symbol, pd.DataFrame()).copy())
    if df.empty:
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=fetch_empty_after_refresh symbol=%s",
            symbol,
        )
        return None, "db_miss"
    if not _has_coverage(df):
        logger.warning(
            "forecast schwab_fallback_skipped_or_failed reason=coverage_incomplete symbol=%s",
            symbol,
        )
        return None, "db_miss"

    provider_norm = provider.strip().lower()
    if not provider_norm:
        provider_norm = "schwab"
    if provider_norm == "db":
        source = "db"
    elif source == "db":
        source = "mixed"
    else:
        source = provider_norm

    logger.info(
        "forecast schwab_fallback_succeeded symbol=%s rows=%s source=%s provider=%s",
        symbol,
        len(df),
        source,
        provider_norm,
    )
    return df, source


def _zscore(value: float, history: Iterable[float]) -> float:
    series = pd.Series(list(history)).dropna()
    if series.empty:
        return math.nan
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std and std > 1e-9:
        return (value - mean) / std
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad and mad > 1e-9:
        return (value - median) / (1.4826 * mad)
    return 0.0


def _frame_features(
    df: pd.DataFrame,
    frame: str,
    freq: str,
    bars: int,
    asof: dt.datetime,
    *,
    iv_rank: Optional[float],
) -> FrameResult:
    aggregated = df.loc[df.index < asof]
    raw: Dict[str, float] = {name: math.nan for name in FEATURE_ORDER}
    zscores: Dict[str, float] = {name: math.nan for name in FEATURE_ORDER}
    if aggregated.empty or len(aggregated) < 2:
        return FrameResult(frame, freq, bars, raw, zscores, len(aggregated))

    window = aggregated.iloc[-bars:]
    closes = aggregated["Close"]
    returns = closes.pct_change().dropna()
    delta_val = float(returns.iloc[-1]) if not returns.empty else math.nan
    gamma_val = math.nan
    if len(returns) >= 2:
        prev_delta = float(returns.iloc[-2])
        gamma_val = delta_val - prev_delta
    slope_series = _rolling_slope(closes, bars)
    slope_val = float(slope_series.iloc[-1]) if not slope_series.empty else math.nan
    volume = float(window["Volume"].iloc[-1]) if len(window) else math.nan
    rsi_series = _rsi(closes)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else math.nan

    raw.update(
        {
            "rsi": rsi_val,
            "delta": delta_val,
            "gamma": gamma_val,
            "slope": slope_val,
            "volume_z": volume,
            "iv_rank": iv_rank if iv_rank is not None else math.nan,
        }
    )

    zscores["rsi"] = _zscore(rsi_val, rsi_series.iloc[:-1])
    zscores["delta"] = _zscore(delta_val, returns.iloc[:-1])
    gamma_series = returns.diff().dropna()
    zscores["gamma"] = _zscore(gamma_val, gamma_series.iloc[:-1])
    zscores["slope"] = _zscore(slope_val, slope_series.iloc[:-1])
    vol_history = aggregated["Volume"].iloc[:-1]
    zscores["volume_z"] = _zscore(volume, vol_history.tail(max(20, bars)))
    zscores["iv_rank"] = 0.0 if iv_rank is not None else math.nan

    return FrameResult(frame, freq, bars, raw, zscores, len(aggregated))


def tod_minute(asof: dt.datetime) -> int:
    asof_et = ensure_utc(asof).astimezone(TZ)
    open_dt = dt.datetime.combine(asof_et.date(), OPEN_TIME, tzinfo=TZ)
    delta = asof_et - open_dt
    return int(delta.total_seconds() // 60)


def _context_returns(df: pd.DataFrame, bars: int, asof: dt.datetime) -> float:
    aggregated = df.loc[df.index < asof]
    if len(aggregated) < bars:
        return math.nan
    window = aggregated.iloc[-bars:]
    start = float(window["Open"].iloc[0])
    end = float(window["Close"].iloc[-1])
    if start == 0:
        return math.nan
    return (end / start) - 1.0


def build_state(ticker: str, asof: dt.datetime) -> Dict[str, object]:
    """Build a multi-resolution feature vector for ``ticker`` at ``asof``."""

    if not isinstance(asof, dt.datetime):
        raise TypeError("asof must be a datetime")
    asof_utc = ensure_utc(asof)

    session_day = asof_utc.astimezone(TZ).date()
    history_start_day = session_day - dt.timedelta(days=LOOKBACK_DAYS)
    history_start = dt.datetime.combine(history_start_day, OPEN_TIME, tzinfo=TZ).astimezone(
        dt.timezone.utc
    )

    cache = _load_cached_history(ticker, history_start, asof_utc)
    if cache is None or cache.data.empty:
        return {
            "ticker": ticker.upper(),
            "asof": asof_utc.isoformat(),
            "tod_minute": tod_minute(asof_utc),
            "vec": [],
            "frames": {frame: {"n": 0} for frame, _, _ in FRAME_ORDER},
            "layout": {},
            "feature_order": FEATURE_ORDER,
            "context": {},
            "sources": {"bars": (cache.source if cache else "db_miss")},
        }

    _ensure_aggregated_frames(ticker, cache)

    spy_cache = _load_cached_history("SPY", history_start, asof_utc)
    if spy_cache is not None and not spy_cache.data.empty:
        _ensure_aggregated_frames("SPY", spy_cache)

    iv_rank_val, iv_source = _current_iv_rank(ticker)

    frames: Dict[str, Dict[str, Dict[str, float]]] = {}
    layout: Dict[str, Tuple[int, int]] = {}
    vector: List[float] = []

    offset = 0
    for frame, freq, bars in FRAME_ORDER:
        aggregated = cache.aggregated.get(frame, pd.DataFrame()) if cache.aggregated else pd.DataFrame()
        result = _frame_features(aggregated, frame, freq, bars, asof_utc, iv_rank=iv_rank_val)
        frames[frame] = {
            "raw": result.raw,
            "z": result.zscores,
            "count": result.count,
        }
        z_values = [result.zscores[name] for name in FEATURE_ORDER]
        vector.extend(float(val) if not math.isnan(val) else 0.0 for val in z_values)
        layout[frame] = (offset, offset + len(z_values))
        offset += len(z_values)

    context = {
        "spy_5m_cum_return": math.nan,
        "spy_30m_cum_return": math.nan,
        "spy_1d_cum_return": math.nan,
        "iv_rank": iv_rank_val,
        "iv_rank_source": iv_source,
        "vix_stub": None,
    }
    if spy_cache is not None and spy_cache.aggregated:
        spy_5m = spy_cache.aggregated.get("5m", pd.DataFrame())
        spy_30m = spy_cache.aggregated.get("30m", pd.DataFrame())
        spy_1d = spy_cache.aggregated.get("1d", pd.DataFrame())
        if spy_5m is not None:
            context["spy_5m_cum_return"] = _context_returns(spy_5m, 12, asof_utc)
        if spy_30m is not None:
            context["spy_30m_cum_return"] = _context_returns(spy_30m, 6, asof_utc)
        if spy_1d is not None:
            context["spy_1d_cum_return"] = _context_returns(spy_1d, 5, asof_utc)

    bars_source = cache.source or "none"
    spy_source = spy_cache.source if spy_cache is not None else "none"
    combined_sources = set(filter(None, bars_source.split("|"))) | set(
        filter(None, spy_source.split("|"))
    )
    combined_sources.discard("none")
    bars_label = "|".join(sorted(combined_sources)) if combined_sources else bars_source

    return {
        "ticker": ticker.upper(),
        "asof": asof_utc.isoformat(),
        "tod_minute": tod_minute(asof_utc),
        "vec": vector,
        "frames": frames,
        "layout": layout,
        "feature_order": FEATURE_ORDER,
        "context": context,
        "sources": {"bars": bars_label or bars_source, "iv": iv_source},
    }


__all__ = [
    "build_state",
    "FRAME_ORDER",
    "FEATURE_ORDER",
    "ensure_utc",
    "load_price_frame",
    "tod_minute",
    "LOOKBACK_DAYS",
]


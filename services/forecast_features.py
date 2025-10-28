"""Feature construction helpers for intraday forecast matching."""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from config import settings
from services import price_store
from services import schwab_client
from services.data_provider import fetch_bars
from services.async_utils import run_coro
from utils import OPEN_TIME, TZ


logger = logging.getLogger(__name__)


MINUTE_INTERVAL = "1m"

FRAME_ORDER: List[Tuple[str, int, int]] = [
    ("5m", 5, 12),
    ("10m", 10, 9),
    ("30m", 30, 6),
]

FEATURE_ORDER = [
    "last_bar_return",
    "cum_return",
    "ema20_slope",
    "realized_vol",
    "volume_z",
]


@dataclass
class FrameResult:
    """Structured results for an aggregated timeframe."""

    frame: str
    minutes: int
    bars: int
    raw: Dict[str, float]
    zscores: Dict[str, float]
    count: int


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


_TOKEN_STATUS_LOGGED = False


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
            expected_last = end_utc - pd.Timedelta(minutes=1)
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
    if not readable and reason != "missing":
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


def _resample_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.resample(f"{minutes}min", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


def _frame_features(
    df: pd.DataFrame,
    frame: str,
    minutes: int,
    bars: int,
    asof: dt.datetime,
) -> FrameResult:
    aggregated = _resample_minutes(df, minutes)
    aggregated = aggregated.loc[aggregated.index < asof]
    raw: Dict[str, float] = {name: math.nan for name in FEATURE_ORDER}
    zscores: Dict[str, float] = {name: math.nan for name in FEATURE_ORDER}
    if aggregated.empty or len(aggregated) < 2:
        return FrameResult(frame, minutes, bars, raw, zscores, len(aggregated))

    window = aggregated.iloc[-bars:]
    returns = aggregated["Close"].pct_change().dropna()
    last_bar_return = float(returns.iloc[-1]) if not returns.empty else math.nan
    cum_return = math.nan
    if len(window) >= 1:
        first_open = float(window["Open"].iloc[0])
        last_close = float(window["Close"].iloc[-1])
        if first_open:
            cum_return = (last_close / first_open) - 1.0
    ema = aggregated["Close"].ewm(span=20, adjust=False).mean()
    ema_slope_series = ema.pct_change()
    ema_slope = float(ema_slope_series.iloc[-1]) if not ema_slope_series.empty else math.nan
    realized_series = returns.rolling(window=bars, min_periods=max(2, bars // 2)).std(ddof=0)
    realized_vol = (
        float(realized_series.iloc[-1]) if not realized_series.empty else math.nan
    )
    volume = float(aggregated["Volume"].iloc[-1]) if len(aggregated) else math.nan

    raw.update(
        {
            "last_bar_return": last_bar_return,
            "cum_return": cum_return,
            "ema20_slope": ema_slope,
            "realized_vol": realized_vol,
            "volume_z": volume,
        }
    )

    zscores["last_bar_return"] = _zscore(last_bar_return, returns.iloc[:-1])
    cum_series = aggregated["Close"].pct_change(periods=bars)
    zscores["cum_return"] = _zscore(cum_return, cum_series.iloc[:-1])
    zscores["ema20_slope"] = _zscore(ema_slope, ema_slope_series.iloc[:-1])
    zscores["realized_vol"] = _zscore(realized_vol, realized_series.iloc[:-1])
    vol_history = aggregated["Volume"].iloc[:-1]
    zscores["volume_z"] = _zscore(volume, vol_history.tail(20))

    return FrameResult(frame, minutes, bars, raw, zscores, len(aggregated))


def tod_minute(asof: dt.datetime) -> int:
    asof_et = ensure_utc(asof).astimezone(TZ)
    open_dt = dt.datetime.combine(asof_et.date(), OPEN_TIME, tzinfo=TZ)
    delta = asof_et - open_dt
    return int(delta.total_seconds() // 60)


def _context_returns(df: pd.DataFrame, minutes: int, bars: int, asof: dt.datetime) -> float:
    aggregated = _resample_minutes(df, minutes)
    aggregated = aggregated.loc[aggregated.index < asof]
    if len(aggregated) < bars:
        return math.nan
    window = aggregated.iloc[-bars:]
    start = float(window["Open"].iloc[0])
    end = float(window["Close"].iloc[-1])
    if start == 0:
        return math.nan
    return (end / start) - 1.0


def build_state(ticker: str, asof: dt.datetime) -> Dict[str, object]:
    """Build intraday feature state vector for ``ticker`` at ``asof``.

    The function loads one-minute OHLCV bars from the local price store.  When
    recent data are missing, Schwab is queried, results are cached to the local
    database and the query is retried.  Features are computed for 5, 10 and
    30 minute aggregated windows using z-scored metrics.
    """

    if not isinstance(asof, dt.datetime):
        raise TypeError("asof must be a datetime")
    asof_utc = ensure_utc(asof)

    session_start_et = dt.datetime.combine(
        asof_utc.astimezone(TZ).date(), OPEN_TIME, tzinfo=TZ
    )
    session_start = session_start_et.astimezone(dt.timezone.utc)
    start = session_start - dt.timedelta(minutes=390)

    bars_df, source = load_price_frame(ticker, start, asof_utc, MINUTE_INTERVAL)
    required_window = max(minutes * bars for _, minutes, bars in FRAME_ORDER)

    if bars_df is None or len(bars_df) < required_window:
        return {
            "ticker": ticker.upper(),
            "asof": asof_utc.isoformat(),
            "tod_minute": tod_minute(asof_utc),
            "vec": [],
            "frames": {frame: {"n": 0} for frame, _, _ in FRAME_ORDER},
            "layout": {},
            "feature_order": FEATURE_ORDER,
            "context": {},
            "sources": {"bars": source or "db_miss"},
        }

    spy_df, spy_source = load_price_frame("SPY", start, asof_utc, MINUTE_INTERVAL)
    if spy_df is None:
        spy_df = pd.DataFrame()

    frames: Dict[str, Dict[str, Dict[str, float]]] = {}
    layout: Dict[str, Tuple[int, int]] = {}
    vector: List[float] = []

    offset = 0
    for frame, minutes, bars in FRAME_ORDER:
        result = _frame_features(bars_df, frame, minutes, bars, asof_utc)
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
        "spy_5m_cum_return": _context_returns(spy_df, 5, 12, asof_utc),
        "spy_30m_cum_return": _context_returns(spy_df, 30, 6, asof_utc),
        "vix_stub": None,
    }

    sources = {source, spy_source}
    sources.discard("none")
    bars_source = "|".join(sorted(sources)) if sources else "none"

    return {
        "ticker": ticker.upper(),
        "asof": asof_utc.isoformat(),
        "tod_minute": tod_minute(asof_utc),
        "vec": vector,
        "frames": frames,
        "layout": layout,
        "feature_order": FEATURE_ORDER,
        "context": context,
        "sources": {"bars": bars_source or source},
    }


__all__ = [
    "build_state",
    "FRAME_ORDER",
    "FEATURE_ORDER",
    "ensure_utc",
    "load_price_frame",
    "tod_minute",
]


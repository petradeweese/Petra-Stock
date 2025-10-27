"""Feature construction helpers for intraday forecast matching."""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from services import price_store
from services import schwab_client
from utils import OPEN_TIME, TZ


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


def _run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(func(*args, **kwargs))
    if loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(func(*args, **kwargs))
        finally:
            new_loop.close()
    return loop.run_until_complete(func(*args, **kwargs))


def _fetch_from_schwab(symbol: str, start: dt.datetime, end: dt.datetime, interval: str) -> pd.DataFrame:
    df = _run_async(schwab_client.get_price_history, symbol, start, end, interval)
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(dt.timezone.utc)
    else:
        df.index = df.index.tz_convert(dt.timezone.utc)
    df = df.sort_index()
    return df


def load_price_frame(symbol: str, start: dt.datetime, end: dt.datetime, interval: str) -> Tuple[pd.DataFrame, str]:
    start_utc = ensure_utc(start)
    end_utc = ensure_utc(end)
    initial = price_store.get_prices_from_db([symbol], start_utc, end_utc, interval=interval)
    df = initial.get(symbol, pd.DataFrame()).copy()
    if not df.empty:
        df = df.loc[(df.index >= start_utc) & (df.index < end_utc)]
    coverage_ok = False
    if not df.empty:
        last_ts = df.index.max()
        if interval.endswith("m"):
            expected_last = end_utc - pd.Timedelta(minutes=1)
        else:
            expected_last = end_utc - pd.Timedelta(days=1)
        coverage_ok = last_ts >= expected_last
        if coverage_ok:
            try:
                gaps = price_store.detect_gaps(
                    symbol,
                    start_utc,
                    end_utc,
                    interval=interval,
                )
                coverage_ok = len(gaps) == 0
            except Exception:
                coverage_ok = True
    source = "db" if not df.empty else "none"
    if not coverage_ok:
        fetched = _fetch_from_schwab(symbol, start_utc, end_utc, interval)
        if not fetched.empty:
            price_store.upsert_bars(symbol, fetched, interval=interval)
            if source == "db" and not df.empty:
                source = "mixed"
            else:
                source = "schwab"
            refreshed = price_store.get_prices_from_db(
                [symbol], start_utc, end_utc, interval=interval
            )
            df = refreshed.get(symbol, pd.DataFrame()).copy()
            if not df.empty:
                df = df.loc[(df.index >= start_utc) & (df.index < end_utc)]
        else:
            df = fetched
            source = "schwab" if source == "none" else source
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
    spy_df, spy_source = load_price_frame("SPY", start, asof_utc, MINUTE_INTERVAL)

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


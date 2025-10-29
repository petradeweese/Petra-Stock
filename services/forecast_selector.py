"""Selector utilities for intraday forecast emails."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - dependency optional in some environments
    mcal = None

from services.forecast_features import LOOKBACK_DAYS, build_state
from services.forecast_matcher import find_similar_days
from utils import CLOSE_TIME, OPEN_TIME, TZ

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "NFLX",
    "AVGO",
    "COST",
    "CRM",
    "ORCL",
    "INTC",
    "MU",
    "ASML",
    "TSM",
    "SMCI",
    "PANW",
    "SNOW",
    "NOW",
    "ABNB",
    "UBER",
    "SHOP",
    "MRVL",
    "COIN",
    "PLTR",
    "SQ",
    "PYPL",
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "XOM",
    "CVX",
    "OXY",
    "LLY",
    "UNH",
    "PFE",
    "NKE",
    "DIS",
    "CAT",
    "BA",
    "GE",
    "V",
    "MA",
]

_CHAIN_SNAPSHOT_DIR = Path(
    os.getenv("SCHWAB_CHAIN_SNAPSHOT_DIR", "data/schwab_option_chains")
)
_TRADING_MINUTES_PER_YEAR = 252 * 390

if mcal is not None:  # pragma: no cover - optional dependency
    try:
        _XNYS = mcal.get_calendar("XNYS")
    except Exception:  # pragma: no cover - best effort
        _XNYS = None
else:  # pragma: no cover - optional dependency missing
    _XNYS = None


# Subset of upcoming/recurring NYSE holidays for fallback mode.
_FALLBACK_HOLIDAYS = {
    dt.date(2024, 1, 1),
    dt.date(2024, 1, 15),
    dt.date(2024, 2, 19),
    dt.date(2024, 3, 29),
    dt.date(2024, 5, 27),
    dt.date(2024, 6, 19),
    dt.date(2024, 7, 4),
    dt.date(2024, 9, 2),
    dt.date(2024, 11, 28),
    dt.date(2024, 12, 25),
    dt.date(2025, 1, 1),
    dt.date(2025, 1, 20),
    dt.date(2025, 2, 17),
    dt.date(2025, 4, 18),
    dt.date(2025, 5, 26),
    dt.date(2025, 6, 19),
    dt.date(2025, 7, 4),
    dt.date(2025, 9, 1),
    dt.date(2025, 11, 27),
    dt.date(2025, 12, 25),
}

# Known half-days where the NYSE closes at 1pm ET. The emails should not run
# before 10:00 ET nor after the early close on these sessions.
_FALLBACK_HALF_DAYS = {
    dt.date(2024, 11, 29),
    dt.date(2024, 12, 24),
    dt.date(2025, 11, 28),
    dt.date(2025, 12, 24),
}

# Cache implied move lookups for a single run. The cache key uses the ticker and
# trading date so we do not repeatedly touch the filesystem for the same symbol.
_IV_CACHE: dict[tuple[str, dt.date], Optional[float]] = {}


def _ensure_datetime(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise TypeError("expected datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _fallback_is_trading_day(asof_et: dt.datetime) -> bool:
    if asof_et.weekday() >= 5:
        return False
    day = asof_et.date()
    if day in _FALLBACK_HOLIDAYS:
        return False
    if day in _FALLBACK_HALF_DAYS:
        if asof_et.hour < 10:
            return False
        close_dt = dt.datetime.combine(day, dt.time(13, 0), tzinfo=TZ)
        return asof_et < close_dt
    open_dt = dt.datetime.combine(day, OPEN_TIME, tzinfo=TZ)
    close_dt = dt.datetime.combine(day, CLOSE_TIME, tzinfo=TZ)
    return open_dt <= asof_et < close_dt


def is_trading_day(asof: dt.datetime) -> bool:
    """Return ``True`` when the forecast email should run at ``asof``."""

    asof_utc = _ensure_datetime(asof)
    asof_et = asof_utc.astimezone(TZ)

    if _XNYS is not None:
        try:
            schedule = _XNYS.schedule(
                start_date=asof_et.date(), end_date=asof_et.date()
            )
        except Exception:  # pragma: no cover - calendar lookup failure
            schedule = None
        if schedule is not None and not schedule.empty:
            row = schedule.iloc[0]
            open_dt = row["market_open"].to_pydatetime().astimezone(TZ)
            close_dt = row["market_close"].to_pydatetime().astimezone(TZ)
            if asof_et < open_dt or asof_et >= close_dt:
                return False
            if close_dt.hour < 16 or (close_dt.hour == 16 and close_dt.minute < 0):
                if asof_et.hour < 10:
                    return False
            return True
        return False

    return _fallback_is_trading_day(asof_et)


def _parse_snapshot_datetime(raw: str | None) -> Optional[dt.datetime]:
    if not raw:
        return None
    try:
        snap = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if snap.tzinfo is None:
        snap = snap.replace(tzinfo=dt.timezone.utc)
    return snap.astimezone(dt.timezone.utc)


def _load_chain_snapshot(ticker: str, asof: dt.datetime) -> Optional[dict]:
    if not _CHAIN_SNAPSHOT_DIR.exists():
        return None
    ticker = ticker.upper()
    best: tuple[Optional[dt.datetime], dict] | None = None
    for path in _CHAIN_SNAPSHOT_DIR.glob(f"{ticker}*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        snap_ts = None
        for key in ("asof", "as_of", "timestamp", "quote_time"):
            snap_ts = _parse_snapshot_datetime(data.get(key))
            if snap_ts is not None:
                break
        if snap_ts is None:
            continue
        if snap_ts > asof:
            continue
        if best is None or (best[0] is not None and snap_ts > best[0]):
            best = (snap_ts, data)
    return best[1] if best else None


def _iter_contracts(snapshot: dict) -> Iterable[dict]:
    for key in ("options", "contracts", "legs"):
        payload = snapshot.get(key)
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
    for key in ("calls", "puts"):
        payload = snapshot.get(key)
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry


def _pick_atm_contract(snapshot: dict, asof_et: dt.datetime) -> Optional[dict]:
    underlying = snapshot.get("underlying_price")
    if underlying is None:
        underlying = snapshot.get("underlyingPrice")
    if underlying is None:
        return None
    try:
        underlying = float(underlying)
    except (TypeError, ValueError):
        return None

    best: tuple[int, float, dict] | None = None
    for contract in _iter_contracts(snapshot):
        strike = contract.get("strike") or contract.get("strikePrice")
        if strike is None:
            continue
        try:
            strike_val = float(strike)
        except (TypeError, ValueError):
            continue
        iv = (
            contract.get("iv")
            or contract.get("implied_volatility")
            or contract.get("impliedVolatility")
            or contract.get("ivMid")
        )
        if iv is None:
            continue
        try:
            iv_val = float(iv)
        except (TypeError, ValueError):
            continue
        expiry_raw = contract.get("expiry") or contract.get("expirationDate")
        expiry_dt: Optional[dt.datetime] = None
        if isinstance(expiry_raw, (int, float)):
            try:
                expiry_dt = dt.datetime.fromtimestamp(float(expiry_raw), tz=dt.timezone.utc)
            except Exception:
                expiry_dt = None
        elif isinstance(expiry_raw, str):
            expiry_dt = _parse_snapshot_datetime(expiry_raw)
            if expiry_dt is None:
                try:
                    expiry_dt = dt.datetime.strptime(expiry_raw, "%Y-%m-%d")
                    expiry_dt = expiry_dt.replace(tzinfo=TZ)
                except ValueError:
                    expiry_dt = None
        elif isinstance(expiry_raw, dt.date):
            expiry_dt = dt.datetime.combine(expiry_raw, dt.time(16, 0), tzinfo=TZ)
        dte = contract.get("dte") or contract.get("days_to_expiration")
        if expiry_dt is None and dte is not None:
            try:
                days = float(dte)
                expiry_dt = asof_et + dt.timedelta(days=days)
            except (TypeError, ValueError):
                expiry_dt = None
        if expiry_dt is not None:
            expiry_dt = _ensure_datetime(expiry_dt)
        days_out = 999
        if expiry_dt is not None:
            days_out = max(0, (expiry_dt.astimezone(TZ).date() - asof_et.date()).days)
        # Prefer expiries within the next week.
        expiry_rank = min(days_out, 7)
        strike_diff = abs(strike_val - underlying)
        ranking = (expiry_rank, strike_diff)
        if best is None or ranking < best[:2]:
            best = (expiry_rank, strike_diff, {"iv": iv_val})
    return best[2] if best else None


def get_implied_move_pct(ticker: str, asof: dt.datetime) -> Optional[float]:
    """Return the implied end-of-day move in percent for ``ticker``."""

    asof_utc = _ensure_datetime(asof)
    asof_et = asof_utc.astimezone(TZ)
    cache_key = (ticker.upper(), asof_et.date())
    if cache_key in _IV_CACHE:
        return _IV_CACHE[cache_key]

    snapshot = _load_chain_snapshot(ticker, asof_utc)
    if not snapshot:
        _IV_CACHE[cache_key] = None
        return None

    contract = _pick_atm_contract(snapshot, asof_et)
    if not contract:
        _IV_CACHE[cache_key] = None
        return None

    iv = contract.get("iv")
    if iv is None:
        _IV_CACHE[cache_key] = None
        return None
    try:
        iv_float = float(iv)
    except (TypeError, ValueError):
        _IV_CACHE[cache_key] = None
        return None
    if math.isnan(iv_float) or math.isinf(iv_float):
        _IV_CACHE[cache_key] = None
        return None

    close_dt = dt.datetime.combine(asof_et.date(), CLOSE_TIME, tzinfo=TZ)
    minutes_remaining = (close_dt - asof_et).total_seconds() / 60.0
    if minutes_remaining <= 0:
        _IV_CACHE[cache_key] = None
        return None
    time_fraction = minutes_remaining / _TRADING_MINUTES_PER_YEAR
    move = max(0.0, iv_float * math.sqrt(time_fraction) * 100.0)
    result = round(move, 3)
    _IV_CACHE[cache_key] = result
    return result


def implied_eod_move_pct(ticker: str, asof: dt.datetime) -> Optional[float]:
    """Backward-compatible alias for :func:`get_implied_move_pct`."""

    return get_implied_move_pct(ticker, asof)


def _clean_float(value: float | None) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return round(num, 3)


def _clean_pair(values: Iterable[float]) -> Optional[list[float]]:
    sanitized: list[float] = []
    for value in values:
        cleaned = _clean_float(value)
        if cleaned is None:
            return None
        sanitized.append(cleaned)
    if not sanitized:
        return None
    return sanitized


def _bias_from_median(median_close: float) -> str:
    return "Up" if median_close >= 0 else "Down"


def select_forecast_top5(
    asof: dt.datetime, tickers: List[str] = DEFAULT_UNIVERSE
) -> List[dict]:
    """Select the top five forecast candidates for ``asof``."""

    asof_utc = _ensure_datetime(asof)
    results: list[dict] = []
    total = 0
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        total += 1
        try:
            state = build_state(ticker, asof_utc)
        except Exception:
            logger.exception("forecast_selector build_state_failed ticker=%s", ticker)
            continue
        try:
            payload = find_similar_days(
                ticker, state, asof_utc, lookback_days=LOOKBACK_DAYS
            )
        except Exception:
            logger.exception("forecast_selector matcher_failed ticker=%s", ticker)
            continue

        n = int(payload.get("n", 0) or 0)
        if n <= 0:
            continue
        summary = payload.get("summary") or {}
        median_close = _clean_float(summary.get("median_close_pct"))
        if median_close is None:
            continue
        confidence = float(payload.get("confidence") or 0.0)
        implied_move = get_implied_move_pct(ticker, asof_utc)
        magnitude = abs(median_close)
        if implied_move is None:
            edge = magnitude
        else:
            edge = max(magnitude - implied_move, 0.0)
        score_basis = edge if edge > 0 else magnitude
        score = round(confidence * score_basis, 6)
        entry = {
            "ticker": ticker,
            "asof": asof_utc.isoformat(),
            "n": n,
            "confidence": round(confidence, 6),
            "median_close_pct": median_close,
            "iqr_close_pct": _clean_pair(summary.get("iqr_close_pct") or []) or [],
            "p5_p95_close_pct": _clean_pair(summary.get("p5_p95_close_pct") or []) or [],
            "median_high_pct": _clean_float(summary.get("median_high_pct")),
            "median_low_pct": _clean_float(summary.get("median_low_pct")),
            "implied_eod_move_pct": implied_move,
            "edge": round(edge, 6) if edge is not None else None,
            "score": score,
            "bias": _bias_from_median(median_close),
            "low_sample": bool(payload.get("low_sample")),
        }
        results.append(entry)

    results.sort(
        key=lambda item: (
            -(item.get("score") or 0.0),
            -(item.get("confidence") or 0.0),
            -(item.get("n") or 0),
        )
    )
    top5 = results[:5]
    logger.info(
        "forecast_selector completed asof=%s processed=%d selected=%d tickers=%s",
        asof_utc.isoformat(),
        total,
        len(top5),
        [row["ticker"] for row in top5],
    )
    return top5


__all__ = [
    "DEFAULT_UNIVERSE",
    "get_implied_move_pct",
    "implied_eod_move_pct",
    "is_trading_day",
    "select_forecast_top5",
]

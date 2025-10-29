"""API endpoints for intraday forecast data."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from services.forecast_features import LOOKBACK_DAYS, build_state
from services.forecast_matcher import find_similar_days
from utils import TZ, now_et

router = APIRouter()

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 120.0
_ForecastKey = Tuple[str, str]
_CACHE: Dict[_ForecastKey, Tuple[float, Dict[str, object]]] = {}


def _parse_asof(asof_param: str | None) -> datetime:
    """Parse the optional ``asof`` query parameter."""

    if not asof_param:
        return now_et()
    value = asof_param.strip()
    if not value:
        return now_et()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="Invalid asof timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(TZ)


def _round_to_minute(ts: datetime) -> datetime:
    tz_aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return tz_aware.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _cache_key(ticker: str, asof: datetime) -> _ForecastKey:
    rounded = _round_to_minute(asof)
    return ticker.upper(), rounded.isoformat()


def _get_cached(key: _ForecastKey) -> Dict[str, object] | None:
    expires_data = _CACHE.get(key)
    if not expires_data:
        return None
    expires_at, payload = expires_data
    if expires_at >= time.monotonic():
        logger.info("forecast cache hit ticker=%s asof=%s", key[0], key[1])
        return payload
    _CACHE.pop(key, None)
    return None


@router.get("/api/forecast/{ticker}")
def get_forecast(
    ticker: str,
    *,
    asof: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    """Return a JSON forecast summary for ``ticker``."""

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    ticker_clean = ticker.upper()
    asof_dt = _parse_asof(asof)
    logger.info(
        "[forecast] API hit: %s asof=%s limit=%s",
        ticker_clean,
        asof_dt.isoformat(),
        limit,
    )

    cache_key = _cache_key(ticker_clean, asof_dt)
    cached = _get_cached(cache_key)
    if cached is None:
        try:
            asof_state = build_state(ticker_clean, asof_dt)
            forecast = find_similar_days(
                ticker_clean, asof_state, asof_dt, lookback_days=LOOKBACK_DAYS
            )
        except Exception as exc:  # pragma: no cover - safeguard
            error_message = str(exc)
            logger.exception(
                "forecast get_forecast_failed",
                extra={
                    "ticker": ticker_clean,
                    "asof": asof_dt.isoformat(),
                    "limit": limit,
                    "err": error_message,
                },
            )
            error_payload = {
                "ticker": ticker_clean,
                "asof": asof_dt.astimezone(timezone.utc).isoformat(),
                "n": 0,
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "low_sample": True,
                "summary": {},
                "matches": [],
                "error": error_message[:300],
                "sources": {"bars": "error"},
            }
            forecast = error_payload
        else:
            _CACHE[cache_key] = (time.monotonic() + _CACHE_TTL_SECONDS, forecast)
    else:
        forecast = cached

    matches = list(forecast.get("matches", []))
    matches = matches[:limit]
    response = dict(forecast)
    response["ticker"] = ticker_clean
    response["asof"] = asof_dt.astimezone(timezone.utc).isoformat()
    response["matches"] = matches
    response.setdefault("n", len(forecast.get("matches", [])))
    return JSONResponse(response)


__all__ = ["router"]


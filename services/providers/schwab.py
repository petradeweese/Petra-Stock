"""Schwab market data provider implementation."""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import httpx
import pandas as pd

from config import SCHWAB_RPS
from services.errors import DataUnavailableError as _BaseDataUnavailableError

logger = logging.getLogger(__name__)


class DataUnavailableError(_BaseDataUnavailableError):
    """Raised when Schwab market data cannot be returned."""


@dataclass(frozen=True)
class IntervalSpec:
    granularity: str
    multiplier: int
    resample_rule: Optional[str] = None


_EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_TIMESTAMP_FIELDS = (
    "datetime",
    "timestamp",
    "time",
    "t",
    "start",
    "startTime",
    "datetimeUTC",
)
_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "tradevolume": "Volume",
    "trade_volume": "Volume",
}
_INTERVAL_MAP: MutableMapping[str, IntervalSpec] = {
    "1m": IntervalSpec("minute", 1),
    "5m": IntervalSpec("minute", 5),
    "15m": IntervalSpec("minute", 15),
    "30m": IntervalSpec("minute", 30),
    "1h": IntervalSpec("minute", 30, "60min"),
    "1d": IntervalSpec("day", 1),
}

_SESSION_LOCK = threading.Lock()
_SESSION: Optional[httpx.Client] = None
_TOKEN_LOCK = threading.Lock()
_ACCESS_TOKEN: Optional[str] = None
_TOKEN_EXPIRES_AT = 0.0
_RATE_LOCK = threading.Lock()
_RATE_REQUESTS: deque[float] = deque()
_REQUEST_TIMEOUT = float(os.getenv("SCHWAB_TIMEOUT", "10"))
RETURNS_ADJUSTED = False


def _env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise DataUnavailableError(f"schwab: missing env {name}")
    return val


def _bars_path() -> str:
    path = os.getenv("SCHWAB_BARS_PATH", "").strip()
    if not path:
        raise DataUnavailableError("schwab: SCHWAB_BARS_PATH not configured")
    if not path.startswith("/"):
        path = "/" + path
    return path


def _corp_actions_path() -> Optional[str]:
    path = os.getenv("SCHWAB_CORP_ACTIONS_PATH", "").strip()
    if not path:
        return None
    if not path.startswith("/"):
        path = "/" + path
    return path


def _to_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _throttle() -> None:
    rate = max(0, SCHWAB_RPS)
    if rate <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        window = 1.0
        while _RATE_REQUESTS and now - _RATE_REQUESTS[0] >= window:
            _RATE_REQUESTS.popleft()
        if len(_RATE_REQUESTS) >= rate:
            sleep_for = window - (now - _RATE_REQUESTS[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while _RATE_REQUESTS and now - _RATE_REQUESTS[0] >= window:
                _RATE_REQUESTS.popleft()
        _RATE_REQUESTS.append(now)


def get_session() -> httpx.Client:
    """Return a shared HTTPX session for Schwab API calls."""

    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        base_url = os.getenv("SCHWAB_API_BASE", "").strip()
        if not base_url:
            raise DataUnavailableError("schwab: SCHWAB_API_BASE not configured")
        _SESSION = httpx.Client(
            base_url=base_url.rstrip("/"),
            http2=True,
            timeout=_REQUEST_TIMEOUT,
        )
        return _SESSION


def _refresh_token(session: httpx.Client) -> Tuple[str, float]:
    client_id = _env("SCHWAB_CLIENT_ID")
    client_secret = _env("SCHWAB_SECRET")
    refresh_token = _env("SCHWAB_REFRESH_TOKEN")
    oauth_url = _env("SCHWAB_OAUTH_URL")
    redirect_uri = _env("SCHWAB_REDIRECT_URI")

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }
    response = session.post(
        oauth_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=_REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        err = DataUnavailableError(
            f"schwab: oauth status {response.status_code}"
        )
        setattr(err, "status", response.status_code)
        raise err
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise DataUnavailableError("schwab: missing access token")
    expires_in = data.get("expires_in", 1800)
    try:
        ttl = float(expires_in)
    except (TypeError, ValueError):
        ttl = 1800.0
    return token, ttl


def _get_access_token(session: httpx.Client) -> str:
    global _ACCESS_TOKEN, _TOKEN_EXPIRES_AT
    with _TOKEN_LOCK:
        now = time.time()
        if _ACCESS_TOKEN and now < _TOKEN_EXPIRES_AT - 30:
            return _ACCESS_TOKEN
        token, ttl = _refresh_token(session)
        _ACCESS_TOKEN = token
        _TOKEN_EXPIRES_AT = now + max(60.0, ttl - 30.0)
        return _ACCESS_TOKEN


def get_access_token(session: Optional[httpx.Client] = None) -> str:
    """Return a valid OAuth access token for Schwab API requests."""

    if session is None:
        session = get_session()
    return _get_access_token(session)


def _extract_rows(payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise DataUnavailableError("schwab: invalid payload")
    data = payload
    if "data" in data and isinstance(data["data"], Mapping):
        data = data["data"]  # type: ignore[assignment]
    for key in ("bars", "candles", "results"):
        rows = data.get(key) if isinstance(data, Mapping) else None
        if isinstance(rows, Sequence):
            return rows  # type: ignore[return-value]
    if isinstance(data, Sequence):
        return data  # type: ignore[return-value]
    raise DataUnavailableError("schwab: no bars in response")


def _extract_actions(payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    data = payload
    if "data" in data and isinstance(data["data"], Mapping):
        data = data["data"]  # type: ignore[assignment]
    for key in ("actions", "corporateActions", "events"):
        rows = data.get(key) if isinstance(data, Mapping) else None
        if isinstance(rows, Sequence):
            return rows  # type: ignore[return-value]
    if isinstance(data, Sequence):
        return data  # type: ignore[return-value]
    return []


def _normalize_bars(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not rows:
        raise DataUnavailableError("schwab: empty result")
    df = pd.DataFrame(rows)
    ts_col = next((col for col in _TIMESTAMP_FIELDS if col in df.columns), None)
    if ts_col is None:
        raise DataUnavailableError("schwab: missing timestamp column")
    index = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if index.isna().all():
        raise DataUnavailableError("schwab: invalid timestamps")
    df = df.drop(columns=[ts_col])
    valid = ~index.isna()
    df = df.loc[valid]
    index = index[valid]
    rename_map = {
        col: _COLUMN_MAP[str(col).lower()]
        for col in df.columns
        if str(col).lower() in _COLUMN_MAP
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in _EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_EXPECTED_COLUMNS]
    for col in _EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    if df.empty:
        raise DataUnavailableError("schwab: empty result")
    df["Volume"] = df["Volume"].fillna(0).astype(float)
    df.index = index
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


def _resample(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if not rule or df.empty:
        return df
    grouped = df.resample(rule, label="right", closed="right").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    grouped = grouped.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    grouped["Volume"] = grouped["Volume"].fillna(0)
    grouped = grouped[~grouped.index.duplicated(keep="last")]
    grouped = grouped.sort_index()
    return grouped


def _fetch_raw_bars(
    session: httpx.Client,
    token: str,
    symbol: str,
    spec: IntervalSpec,
    start: dt.datetime,
    end: dt.datetime,
) -> Sequence[Mapping[str, object]]:
    params = {
        "symbol": symbol,
        "granularity": spec.granularity,
        "multiplier": spec.multiplier,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    headers = {"Authorization": f"Bearer {token}"}
    _throttle()
    logger.debug(
        "schwab_request symbol=%s granularity=%s multiplier=%s start=%s end=%s",
        symbol,
        spec.granularity,
        spec.multiplier,
        start.isoformat(),
        end.isoformat(),
    )
    response = session.get(
        _bars_path(),
        params=params,
        headers=headers,
        timeout=_REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        err = DataUnavailableError(
            f"schwab: http {response.status_code}"
        )
        setattr(err, "status", response.status_code)
        raise err
    payload = response.json()
    return _extract_rows(payload)


def fetch_corporate_actions(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    path = _corp_actions_path()
    if not path:
        return pd.DataFrame()
    session = get_session()
    token = get_access_token(session)
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    params = {
        "symbol": symbol,
        "start": start_utc.date().isoformat(),
        "end": end_utc.date().isoformat(),
    }
    headers = {"Authorization": f"Bearer {token}"}
    _throttle()
    response = session.get(path, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
    if response.status_code != 200:
        err = DataUnavailableError(f"schwab: corp http {response.status_code}")
        setattr(err, "status", response.status_code)
        raise err
    payload = response.json()
    rows = _extract_actions(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def _resolve_interval(interval: str) -> IntervalSpec:
    key = interval.strip().lower()
    spec = _INTERVAL_MAP.get(key)
    if spec is None:
        raise DataUnavailableError(f"schwab: unsupported interval {interval}")
    return spec


def fetch_prices(symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Return OHLCV bars for ``symbol`` between ``start`` and ``end``."""

    if end <= start:
        raise DataUnavailableError("schwab: invalid time window")
    session = get_session()
    token = get_access_token(session)
    spec = _resolve_interval(interval)
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    rows = _fetch_raw_bars(session, token, symbol, spec, start_utc, end_utc)
    df = _normalize_bars(rows)
    df = df.loc[(df.index >= start_utc) & (df.index <= end_utc)]
    df = _resample(df, spec.resample_rule)
    df = df.loc[(df.index >= start_utc) & (df.index <= end_utc)]
    if df.empty:
        raise DataUnavailableError("schwab: empty result")
    df.index = df.index.tz_convert("UTC")
    return df[_EXPECTED_COLUMNS]


def fetch_options_chain(*args, **kwargs):  # pragma: no cover - compatibility shim
    from . import schwab_options as _schwab_options

    return _schwab_options.fetch_chain(*args, **kwargs)


__all__ = [
    "fetch_prices",
    "fetch_corporate_actions",
    "fetch_options_chain",
    "get_session",
    "get_access_token",
    "RETURNS_ADJUSTED",
]

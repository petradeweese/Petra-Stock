"""Schwab options chain provider implementation."""
from __future__ import annotations

import logging
import os
from typing import Mapping, MutableMapping, Optional, Sequence

import httpx
import pandas as pd

from services.providers import schwab

logger = logging.getLogger(__name__)


class OptionsUnavailableError(Exception):
    """Raised when Schwab options data is unavailable."""


EXPECTED_COLUMNS = [
    "symbol",
    "underlying",
    "expiry",
    "strike",
    "type",
    "bid",
    "ask",
    "last",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "openInterest",
    "volume",
    "updated_at",
]

ALIAS_MAP: Mapping[str, Sequence[str]] = {
    "symbol": ("symbol", "optionSymbol", "option_symbol", "occSymbol", "contractSymbol"),
    "underlying": ("underlying", "underlyingSymbol", "underlying_symbol"),
    "expiry": (
        "expiry",
        "expiration",
        "expirationDate",
        "expiration_date",
        "expiryDate",
        "expirationdate",
        "expiration_date",
    ),
    "strike": ("strike", "strikePrice", "strike_price"),
    "type": ("type", "putCall", "put_call", "callPut", "optionType", "option_type"),
    "bid": ("bid", "bidPrice", "bid_price"),
    "ask": ("ask", "askPrice", "ask_price"),
    "last": ("last", "lastPrice", "last_price", "mark", "markPrice"),
    "iv": ("iv", "impliedVolatility", "implied_volatility"),
    "delta": ("delta",),
    "gamma": ("gamma",),
    "theta": ("theta",),
    "vega": ("vega",),
    "openInterest": ("openInterest", "open_interest"),
    "volume": ("volume", "totalVolume", "tradeVolume"),
    "updated_at": (
        "updated_at",
        "updatedAt",
        "quoteTime",
        "quote_time",
        "quoteDateTime",
        "tradeTimeInLong",
        "time",
        "timestamp",
        "lastTradeDate",
    ),
}

TYPE_MAP = {
    "call": "call",
    "calls": "call",
    "c": "call",
    "put": "put",
    "puts": "put",
    "p": "put",
}

NUMERIC_COLUMNS = [
    "strike",
    "bid",
    "ask",
    "last",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "openInterest",
    "volume",
]

_REQUEST_TIMEOUT = float(os.getenv("SCHWAB_TIMEOUT", "10"))


def _options_path() -> str:
    path = os.getenv("SCHWAB_OPTIONS_PATH", "").strip()
    if not path:
        raise OptionsUnavailableError("schwab: SCHWAB_OPTIONS_PATH not configured")
    if not path.startswith("/"):
        path = "/" + path
    return path


def _normalise_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    series = pd.to_numeric(series, errors="coerce")
    return series


def _coerce_timestamp(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    if pd.api.types.is_numeric_dtype(values):
        # Determine if timestamps are milliseconds
        mask = values > 1e11
        converted = values.copy()
        converted[mask] = converted[mask] / 1000.0
        values = converted
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts


def _coerce_expiry(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    cleaned = values.astype(str).str.split(":").str[0]
    ts = pd.to_datetime(cleaned, utc=True, errors="coerce")
    return ts


def _fill_from_aliases(df: pd.DataFrame, target: str) -> None:
    for alias in ALIAS_MAP.get(target, ()):  # type: ignore[arg-type]
        if alias in df.columns and target not in df.columns:
            df[target] = df[alias]
            return
    if target not in df.columns:
        df[target] = pd.NA


def _extract_from_maps(data: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    for key, opt_type in (("callExpDateMap", "call"), ("putExpDateMap", "put")):
        mapping = data.get(key)
        if not isinstance(mapping, Mapping):
            continue
        for expiry_key, strike_map in mapping.items():
            if not isinstance(strike_map, Mapping):
                continue
            for strike_key, contracts in strike_map.items():
                if not isinstance(contracts, Sequence):
                    continue
                for contract in contracts:
                    if not isinstance(contract, Mapping):
                        continue
                    item = dict(contract)
                    item.setdefault("type", opt_type)
                    item.setdefault("expiry", expiry_key)
                    item.setdefault("strike", strike_key)
                    rows.append(item)
    return rows


def _flatten_payload(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    data = payload
    if "data" in data and isinstance(data["data"], Mapping):
        data = data["data"]  # type: ignore[assignment]
    if isinstance(data, Mapping):
        if "options" in data and isinstance(data["options"], Sequence):
            return [row for row in data["options"] if isinstance(row, Mapping)]  # type: ignore[list-item]
        if "optionChain" in data and isinstance(data["optionChain"], Mapping):
            return _extract_from_maps(data["optionChain"])
        if "chains" in data and isinstance(data["chains"], Sequence):
            return [row for row in data["chains"] if isinstance(row, Mapping)]  # type: ignore[list-item]
    if isinstance(data, Sequence):
        return [row for row in data if isinstance(row, Mapping)]  # type: ignore[list-item]
    raise OptionsUnavailableError("schwab: invalid options payload")


def _prepare_dataframe(rows: Sequence[Mapping[str, object]], symbol: str) -> pd.DataFrame:
    if not rows:
        raise OptionsUnavailableError("schwab: empty options result")
    df = pd.DataFrame(rows)
    # Normalise aliases
    for target in EXPECTED_COLUMNS:
        _fill_from_aliases(df, target)
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = symbol
    if "underlying" not in df.columns or df["underlying"].isna().all():
        df["underlying"] = symbol
    if "type" in df.columns:
            df["type"] = (
                df["type"].astype(str).str.lower().map(TYPE_MAP).fillna(pd.NA)
            )
    if "expiry" in df.columns:
        df["expiry"] = _coerce_expiry(df["expiry"])
    if "updated_at" in df.columns:
        df["updated_at"] = _coerce_timestamp(df["updated_at"])
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _normalise_series(df[col])
    if "iv" in df.columns:
        df.loc[df["iv"] > 1.5, "iv"] = df.loc[df["iv"] > 1.5, "iv"] / 100.0
    df = df.dropna(subset=["strike", "type", "bid", "ask"], how="any")
    if df.empty:
        raise OptionsUnavailableError("schwab: no usable option rows")
    for col in ("openInterest", "volume"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
    now = pd.Timestamp.now(tz="UTC")
    if "updated_at" in df.columns:
        df["updated_at"] = df["updated_at"].fillna(now)
    else:
        df["updated_at"] = now
    if "expiry" in df.columns:
        df = df.dropna(subset=["expiry"], how="any")
    df = df[EXPECTED_COLUMNS]
    df = df.reset_index(drop=True)
    df = df.sort_values(["expiry", "type", "strike", "bid"], kind="mergesort")
    return df


def _fetch_raw_chain(
    session,
    token: str,
    symbol: str,
    expiry: Optional[str],
) -> Mapping[str, object]:
    params: MutableMapping[str, str] = {"symbol": symbol}
    if expiry:
        params["expiry"] = expiry
    else:
        params["includeWeekly"] = "true"
        params["includeMonthly"] = "true"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = session.get(
            _options_path(),
            params=params,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        )
    except Exception as exc:  # pragma: no cover - network failure
        raise OptionsUnavailableError(f"schwab: request error: {exc}") from exc
    if response.status_code != 200:
        raise OptionsUnavailableError(f"schwab: http {response.status_code}")
    return response.json()


def get_session() -> httpx.Client:
    """Return the shared Schwab HTTP client for options requests."""

    return schwab.get_session()


def fetch_chain(symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
    """Return the Schwab options chain for ``symbol``."""

    try:
        session = get_session()
        token = schwab.get_access_token(session)
    except Exception as exc:  # pragma: no cover - defensive
        raise OptionsUnavailableError(f"schwab: auth error: {exc}") from exc

    payload = _fetch_raw_chain(session, token, symbol, expiry)
    rows = _flatten_payload(payload)
    df = _prepare_dataframe(rows, symbol)
    return df


__all__ = [
    "OptionsUnavailableError",
    "fetch_chain",
    "get_session",
    "EXPECTED_COLUMNS",
    "ALIAS_MAP",
    "TYPE_MAP",
    "NUMERIC_COLUMNS",
]

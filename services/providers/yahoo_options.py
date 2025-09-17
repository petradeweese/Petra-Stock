"""Yahoo Finance fallback options provider."""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
import yfinance as yf

from services.providers.schwab_options import OptionsUnavailableError

logger = logging.getLogger(__name__)

_EXPECTED_COLUMNS = [
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


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _list_expiries(ticker: yf.Ticker, expiry: Optional[str]) -> List[str]:
    if expiry:
        return [expiry]
    expiries = getattr(ticker, "options", None)
    if not expiries:
        raise OptionsUnavailableError("yahoo: no expiries available")
    if isinstance(expiries, (list, tuple)):
        return list(expiries[:2]) or list(expiries)
    return [str(expiries)]


def _fetch_for_expiry(ticker: yf.Ticker, expiry: str) -> pd.DataFrame:
    try:
        chain = ticker.option_chain(expiry)
    except Exception as exc:  # pragma: no cover - network failure
        raise OptionsUnavailableError(f"yahoo: option_chain error: {exc}") from exc
    frames: List[pd.DataFrame] = []
    for attr, opt_type in (("calls", "call"), ("puts", "put")):
        data = getattr(chain, attr, None)
        if data is None or data.empty:
            continue
        df = data.copy()
        df["type"] = opt_type
        df["expiry"] = expiry
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_EXPECTED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _prepare(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        raise OptionsUnavailableError("yahoo: empty chain")
    rename_map = {
        "contractSymbol": "symbol",
        "lastPrice": "last",
        "impliedVolatility": "iv",
        "openInterest": "openInterest",
        "lastTradeDate": "updated_at",
    }
    df = df.rename(columns=rename_map)
    df["underlying"] = symbol
    missing = [col for col in _EXPECTED_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df[_EXPECTED_COLUMNS]


def fetch_chain(symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
    ticker = _ticker(symbol)
    expiries = _list_expiries(ticker, expiry)
    frames: List[pd.DataFrame] = []
    for exp in expiries:
        exp_str = str(exp)
        df = _fetch_for_expiry(ticker, exp_str)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise OptionsUnavailableError("yahoo: no option data")
    result = pd.concat(frames, ignore_index=True)
    result = _prepare(result, symbol)
    logger.debug(
        "yahoo_options_fetch symbol=%s expiry=%s rows=%d",
        symbol,
        expiry,
        len(result),
    )
    return result


__all__ = ["fetch_chain"]

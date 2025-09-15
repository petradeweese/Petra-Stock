"""Lightweight option chain provider with in-memory caching.

The real project would fetch option chains from an external data vendor.
For tests we keep the implementation minimal and allow the function to be
monkeypatched.  A tiny TTL based cache is provided to avoid repeated lookups
within a short window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import List

__all__ = ["OptionContract", "get_chain"]


@dataclass
class OptionContract:
    occ: str
    side: str  # "call" or "put"
    strike: float
    expiry: date
    bid: float
    ask: float
    mid: float
    last: float
    open_interest: int
    volume: int
    delta: float
    gamma: float
    theta: float
    vega: float
    spread_pct: float
    dte: int
    iv_rank: float


_CACHE: dict[str, tuple[float, List[OptionContract]]] = {}
_TTL = 10.0  # seconds


def get_chain(ticker: str) -> List[OptionContract]:  # pragma: no cover - network stub
    """Return a list of :class:`OptionContract` objects for ``ticker``.

    The default implementation returns an empty list and is expected to be
    monkeypatched in tests.  Results are cached for a short period to mirror
    the behaviour of the production service.
    """

    now = time.time()
    cached = _CACHE.get(ticker)
    if cached and now - cached[0] < _TTL:
        return cached[1]

    data: List[OptionContract] = []
    _CACHE[ticker] = (now, data)
    return data

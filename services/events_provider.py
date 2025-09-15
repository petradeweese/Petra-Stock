"""Simple events provider returning upcoming earnings/dividend/FOMC dates.

The implementation is intentionally minimal and returns an empty list by
default.  Tests may monkeypatch :func:`next_events` to supply synthetic data.
"""
from __future__ import annotations

from datetime import date
import time
from typing import List, Dict

__all__ = ["next_events"]

_CACHE: dict[str, tuple[float, List[Dict[str, str]]]] = {}
_TTL = 10.0


def next_events(ticker: str) -> List[Dict[str, str]]:  # pragma: no cover - network stub
    """Return a list of upcoming events for ``ticker``.

    Each event is a mapping containing ``type`` and ``date`` keys.  ``date`` is
    expected to be in ISO-8601 format (``YYYY-MM-DD``).  Results are cached for a
    short period to avoid repeated lookups.
    """

    now = time.time()
    cached = _CACHE.get(ticker)
    if cached and now - cached[0] < _TTL:
        return cached[1]

    data: List[Dict[str, str]] = []
    _CACHE[ticker] = (now, data)
    return data

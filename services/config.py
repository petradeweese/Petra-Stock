"""Runtime flags and environment-driven configuration helpers."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip() or default)
    except (TypeError, ValueError):
        return int(default)
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


DEBUG_SIMULATION: bool = os.getenv("DEBUG_SIMULATION", "0") == "1"
SMS_MAX_PER_MONTH: int = max(1, _env_int("SMS_MAX_PER_MONTH", 50))
BUSINESS_PHONE: str = _env_str("BUSINESS_PHONE", "+1 4705584503")


__all__ = [
    "BUSINESS_PHONE",
    "DEBUG_SIMULATION",
    "SMS_MAX_PER_MONTH",
]

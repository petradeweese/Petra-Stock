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
BUSINESS_PHONE: str = _env_str("BUSINESS_PHONE", "+1 (555) 555-1212")
BUSINESS_ADDRESS_1: str = _env_str("BUSINESS_ADDRESS_1", "123 Example Street")
BUSINESS_ADDRESS_2: str = _env_str("BUSINESS_ADDRESS_2", "Suite 100")
BUSINESS_CITY: str = _env_str("BUSINESS_CITY", "City")
BUSINESS_REGION: str = _env_str("BUSINESS_REGION", "ST")
BUSINESS_POSTAL: str = _env_str("BUSINESS_POSTAL", "00000")


__all__ = [
    "BUSINESS_ADDRESS_1",
    "BUSINESS_ADDRESS_2",
    "BUSINESS_CITY",
    "BUSINESS_PHONE",
    "BUSINESS_POSTAL",
    "BUSINESS_REGION",
    "DEBUG_SIMULATION",
    "SMS_MAX_PER_MONTH",
]

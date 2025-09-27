"""Lightweight scheduler utilities."""

from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime, timedelta, timezone
from typing import Any

from services import market_calendar, scans
from services.favorites_alerts import _dedupe_key as _fav_dedupe_key
from services.favorites_alerts import mark_sent_key, was_sent_key


logger = logging.getLogger(__name__)


def _interval_to_minutes(interval: str | None) -> int:
    if not interval:
        return 15
    value = str(interval).strip().lower()
    if not value:
        return 15
    try:
        if value.endswith("m"):
            return max(1, int(float(value[:-1])))
        if value.endswith("h"):
            return max(1, int(float(value[:-1]) * 60))
        if value.endswith("d"):
            return max(1, int(float(value[:-1]) * 24 * 60))
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 15


def _last_bar_close(now_utc: datetime, interval_minutes: int) -> datetime:
    seconds = max(60, interval_minutes * 60)
    epoch = int(now_utc.timestamp())
    snapped = epoch - (epoch % seconds)
    return datetime.fromtimestamp(snapped, tz=timezone.utc)


def _is_bar_closed(now_utc: datetime, interval: str | None) -> tuple[bool, datetime]:
    minutes = _interval_to_minutes(interval)
    last_close = _last_bar_close(now_utc, minutes)
    return now_utc >= last_close + timedelta(seconds=10), last_close


def _align_to_bar(now_utc: datetime, interval: str | None) -> datetime:
    minutes = _interval_to_minutes(interval)
    return _last_bar_close(now_utc, minutes)


def _dedupe_key(fav_id: Any, interval: Any, bar_dt_utc: Any, outcome: Any | None = None) -> str | None:
    base = _fav_dedupe_key(fav_id, interval, bar_dt_utc)
    if base and outcome not in (None, ""):
        suffix = str(outcome).strip().lower()
        if suffix:
            return f"{base}|{suffix}"
    return base


_LAST_TICK_BOUNDARY: Optional[datetime] = None


def _tick(now: datetime) -> None:
    """Evaluate periodic jobs for the current ``now`` timestamp."""

    if not market_calendar.is_open(now):
        return
    interval = "15m"
    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    global _LAST_TICK_BOUNDARY

    if now.minute % 15 != 0:
        return

    bar_closed, last_close = _is_bar_closed(now_utc, interval)
    skipped_reason = None

    if not bar_closed:
        skipped_reason = "bar_open"
    elif _LAST_TICK_BOUNDARY == last_close:
        skipped_reason = "already_processed"

    logger.info(
        "favorites_tick",
        extra={
            "bar_close": last_close.replace(microsecond=0).isoformat(),
            "now": now_utc.replace(microsecond=0).isoformat(),
            "grace_s": 10,
            "skipped": skipped_reason,
        },
    )

    if skipped_reason:
        return

    _LAST_TICK_BOUNDARY = last_close
    scans.run_autoscan_batch()


__all__ = ["_tick", "_align_to_bar", "_dedupe_key", "mark_sent_key", "was_sent_key"]

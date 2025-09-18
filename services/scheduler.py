"""Lightweight scheduler utilities."""

from datetime import datetime

from services import market_calendar, scans


def _tick(now: datetime) -> None:
    """Evaluate periodic jobs for the current ``now`` timestamp."""

    if not market_calendar.is_open(now):
        return
    if now.minute % 15 == 0 and now.second < 5:
        scans.run_autoscan_batch()


__all__ = ["_tick"]

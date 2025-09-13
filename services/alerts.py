from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def _trading_days_between(a: datetime, b: datetime) -> int:
    a_d = pd.Timestamp(a.date())
    b_d = pd.Timestamp(b.date())
    start = min(a_d, b_d)
    end = max(a_d, b_d)
    # include both endpoints then subtract one to get distance
    return max(0, len(pd.bdate_range(start, end)) - 1)


def in_earnings_blackout(dates: Iterable[datetime], now: datetime, window: int = 7) -> bool:
    for d in dates:
        if _trading_days_between(d, now) <= window:
            return True
    return False


def alert_due(fav: dict, bar_time: datetime, now: datetime) -> bool:
    last_signal = fav.get("last_signal_bar") or ""
    if last_signal and last_signal == bar_time.isoformat():
        return False
    cooldown = int(fav.get("cooldown_minutes") or 0)
    last_notified = fav.get("last_notified_ts")
    if last_notified:
        try:
            last_dt = datetime.fromisoformat(last_notified)
            if (now - last_dt).total_seconds() < cooldown * 60:
                return False
        except Exception:
            pass
    return True

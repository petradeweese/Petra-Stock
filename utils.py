from datetime import datetime, timezone
from typing import Optional

import pandas_market_calendars as mcal

XNYS = mcal.get_calendar("XNYS")  # New York Stock Exchange
TZ = XNYS.tz  # exchange timezone


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(TZ)


def market_is_open(ts: Optional[datetime] = None) -> bool:
    ts = ts or now_et()
    sched = XNYS.schedule(start_date=ts.date(), end_date=ts.date())
    if sched.empty:
        return False
    open_ts = sched.iloc[0]["market_open"].to_pydatetime().astimezone(TZ)
    close_ts = sched.iloc[0]["market_close"].to_pydatetime().astimezone(TZ)
    return open_ts <= ts <= close_ts

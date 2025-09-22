import datetime as dt

import pandas as pd

from services import data_provider


def test_week_chunking(monkeypatch):
    calls = []

    async def fake_fetch(symbol, start, end, *, interval, timeout_ctx=None):
        calls.append((start, end, interval))
        return pd.DataFrame(), "schwab"

    monkeypatch.setattr(data_provider, "_fetch_single", fake_fetch)

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=15)

    data_provider.fetch_bars(["AAA"], "15m", start, end)
    assert len(calls) == 3  # 15 days -> 3 chunks
    assert all(entry[2] == "15m" for entry in calls)

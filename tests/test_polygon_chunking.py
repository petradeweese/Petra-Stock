import datetime as dt

import pandas as pd

from services import polygon_client


def test_week_chunking(monkeypatch):
    calls = []

    async def fake_fetch(symbol, start, end, multiplier, timespan):
        calls.append((start, end))
        return pd.DataFrame()

    monkeypatch.setattr(polygon_client, "_fetch_single", fake_fetch)
    monkeypatch.setenv("POLYGON_API_KEY", "x")

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=15)

    polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    assert len(calls) == 3  # 15 days -> 3 chunks

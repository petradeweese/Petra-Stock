import datetime as dt

import pandas as pd

from services import polygon_client


def test_week_chunking(monkeypatch):
    windows = []

    async def fake_fetch(symbol, start, end, multiplier, timespan):
        ny_start, ny_end, *_ = polygon_client._normalize_window(start, end)
        windows.append((ny_start, ny_end))
        return pd.DataFrame()

    monkeypatch.setattr(polygon_client, "_fetch_single", fake_fetch)
    monkeypatch.setenv("POLYGON_API_KEY", "x")

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=15)

    polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    assert len(windows) == 3  # 15 days -> 3 chunks
    for i, (ws, we) in enumerate(windows):
        assert (we - ws) <= dt.timedelta(days=7)
        if i > 0:
            assert ws == windows[i - 1][1]

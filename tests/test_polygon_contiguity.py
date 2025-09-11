import datetime as dt

import pandas as pd

from services import polygon_client


def test_dst_contiguity(monkeypatch):
    async def fake_fetch(symbol, start, end, multiplier, timespan):
        idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
        return pd.DataFrame({"Open": range(len(idx))}, index=idx)

    monkeypatch.setattr(polygon_client, "_fetch_single", fake_fetch)
    monkeypatch.setenv("POLYGON_API_KEY", "x")

    start = dt.datetime(2024, 3, 8, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 3, 12, tzinfo=dt.timezone.utc)

    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    idx = data["AAA"].index
    diffs = idx.to_series().diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(minutes=15)

import datetime as dt

import pandas as pd

from services import data_provider


def test_dst_contiguity(monkeypatch):
    async def fake_fetch(symbol, start, end, *, interval, timeout_ctx=None):
        idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
        df = pd.DataFrame({"Open": range(len(idx))}, index=idx)
        return df, "schwab"

    monkeypatch.setattr(data_provider, "_fetch_single", fake_fetch)

    start = dt.datetime(2024, 3, 8, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 3, 12, tzinfo=dt.timezone.utc)

    data = data_provider.fetch_bars(["AAA"], "15m", start, end)
    idx = data["AAA"].index
    diffs = idx.to_series().diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(minutes=15)

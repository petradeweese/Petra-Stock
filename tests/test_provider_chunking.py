import datetime as dt

from services import data_provider


def test_week_chunking(monkeypatch):
    calls = []

    async def fake_range(symbol, interval, start, end, *, timeout_ctx=None):
        calls.append((symbol, interval, start, end))
        return [], "db"

    monkeypatch.setattr(data_provider, "_fetch_range", fake_range)

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=15)

    data_provider.fetch_bars(["AAA"], "15m", start, end)
    assert len(calls) == 1
    symbol, interval, rstart, rend = calls[0]
    assert symbol == "AAA"
    assert interval == "15m"
    assert rstart == start
    assert rend == end

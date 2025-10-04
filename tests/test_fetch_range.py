import asyncio
import datetime as dt

from services import data_provider
import db


def _clear_bars() -> None:
    conn = db.get_engine().raw_connection()
    try:
        conn.execute("DELETE FROM bars")
        conn.commit()
    finally:
        conn.close()


def test_intraday_db_first(monkeypatch):
    _clear_bars()
    calls: list[tuple[dt.datetime, dt.datetime]] = []

    async def fake_provider(symbol, interval, start, end, *, timeout_ctx=None):
        calls.append((start, end))
        ts = start
        bars = []
        while ts < end:
            bars.append(
                {
                    "ts": ts,
                    "open": 1.0,
                    "high": 1.5,
                    "low": 0.5,
                    "close": 1.2,
                    "volume": 100.0,
                }
            )
            ts += dt.timedelta(minutes=15)
        return bars, "schwab"

    monkeypatch.setattr(data_provider, "_fetch_from_provider", fake_provider)

    now = dt.datetime.now(dt.timezone.utc)
    end = now
    start = end - dt.timedelta(hours=1)

    first = asyncio.run(
        data_provider.fetch_range_async("AAPL", "15m", start, end)
    )
    assert len(first) == 4
    assert len(calls) == 1

    second = asyncio.run(
        data_provider.fetch_range_async("AAPL", "15m", start, end)
    )
    assert len(second) == 4
    assert len(calls) == 1  # served from DB on second pass

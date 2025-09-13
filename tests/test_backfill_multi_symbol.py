import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.backfill_polygon as backfill
from services import http_client, polygon_client


def test_backfill_two_symbols(monkeypatch, caplog):
    monkeypatch.setenv("POLYGON_API_KEY", "test")

    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2024-01-01", periods=1, freq="15T", tz="UTC"),
    )

    async def fake_fetch(symbols, interval, start, end):
        sym = symbols[0]
        logger = logging.getLogger("services.polygon_client")
        logger.info("polygon_fetch symbol=%s pages=1 rows=1 duration=0.00", sym)
        return {sym: df}

    monkeypatch.setattr(polygon_client, "fetch_polygon_prices_async", fake_fetch)

    saved = {}

    def fake_upsert(sym, data, interval="15m"):
        saved.setdefault(sym, 0)
        saved[sym] += len(data)
        return len(data)

    monkeypatch.setattr(backfill, "upsert_bars", fake_upsert)

    caplog.set_level(logging.INFO)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    client = http_client.get_client()
    backfill.backfill(["SPY", "QQQ"], start=start, end=end, use_checkpoint=False)
    assert client.is_closed
    assert saved["SPY"] > 0
    assert saved["QQQ"] > 0
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("backfill symbol=SPY returned=1 saved=1" in m for m in messages)
    assert any("backfill symbol=QQQ returned=1 saved=1" in m for m in messages)

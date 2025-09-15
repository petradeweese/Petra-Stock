import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.backfill_polygon as backfill
from services import http_client, polygon_client


def test_quick_test_mode(monkeypatch, caplog):
    monkeypatch.setenv("POLYGON_API_KEY", "test")

    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [1, 2],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="15T", tz="UTC"),
    )

    async def fake_fetch(symbols, interval, start, end, **kwargs):
        assert symbols == ["SPY"]
        assert interval == "15m"
        assert end - start <= dt.timedelta(days=1, minutes=1)
        return {"SPY": df}

    monkeypatch.setattr(polygon_client, "fetch_polygon_prices_async", fake_fetch)

    saved = {}

    def fake_upsert(sym, data, interval="15m"):
        saved["sym"] = sym
        saved["rows"] = len(data)
        return len(data)

    monkeypatch.setattr(backfill, "upsert_bars", fake_upsert)

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(sys, "argv", ["backfill_polygon.py", "--test"])
    client = http_client.get_client()
    backfill.main()
    assert client.is_closed

    assert saved["sym"] == "SPY"
    assert saved["rows"] == 2
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("backfill symbol=SPY returned=2 saved=2" in m for m in messages)

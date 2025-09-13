import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from services import polygon_client
from services.price_store import detect_gaps, upsert_bars


def _make_page(start_ts: pd.Timestamp, count: int) -> dict:
    times = pd.date_range(start_ts, periods=count, freq="15min")
    results = [
        {
            "t": int(ts.timestamp() * 1000),
            "o": 1,
            "h": 1,
            "l": 1,
            "c": 1,
            "v": 1,
        }
        for ts in times
    ]
    return {"results": results}


def test_full_day_bar_count(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    start_ts = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    page = _make_page(start_ts, 26)

    async def fake_get_json(url, headers=None):
        return page

    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)
    start = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)
    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 26

    db.DB_PATH = str(tmp_path / "full.db")
    db.init_db()
    upsert_bars("AAA", df, "15m")
    end_ts = start_ts + pd.Timedelta(minutes=15 * 26)
    gaps = detect_gaps("AAA", start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []


def test_half_day_bar_count(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    start_ts = pd.Timestamp("2024-11-29 14:30", tz="UTC")
    page = _make_page(start_ts, 13)

    async def fake_get_json(url, headers=None):
        return page

    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)
    start = dt.datetime(2024, 11, 29, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 11, 30, tzinfo=dt.timezone.utc)
    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 13

    db.DB_PATH = str(tmp_path / "half.db")
    db.init_db()
    upsert_bars("AAA", df, "15m")
    end_ts = start_ts + pd.Timedelta(minutes=15 * 13)
    gaps = detect_gaps("AAA", start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []

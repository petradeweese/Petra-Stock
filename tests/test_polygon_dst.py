import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import polygon_client


def _capture_ms(monkeypatch, start: dt.datetime, end: dt.datetime) -> int:
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    captured = {}

    async def fake_get_json(url, headers=None):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)
    polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    path = captured["url"].split("?")[0]
    segments = path.split("/")
    return int(segments[-2])  # start_ms


def test_dst_offsets(monkeypatch):
    start = dt.datetime(2024, 9, 3, 12, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    start_ms = _capture_ms(monkeypatch, start, end)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    assert start_dt == pd.Timestamp(start)

    start = dt.datetime(2024, 1, 3, 12, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    start_ms = _capture_ms(monkeypatch, start, end)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    assert start_dt == pd.Timestamp(start)


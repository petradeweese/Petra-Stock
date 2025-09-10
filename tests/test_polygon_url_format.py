import datetime as dt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import polygon_client


def test_polygon_url_uses_unix_ms_and_api_key(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "testkey")
    captured = {}

    async def fake_get_json(url, headers=None):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)

    assert "apiKey=testkey" in captured["url"]
    path = captured["url"].split("?")[0]
    segments = path.split("/")
    assert segments[-1].isdigit()
    assert segments[-2].isdigit()

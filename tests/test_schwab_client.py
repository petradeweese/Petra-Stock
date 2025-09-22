import asyncio
import datetime as dt
import json

import pytest

from config import settings
from services import schwab_client
from services.schwab_client import SchwabAPIError


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode()
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


def test_refresh_flow(monkeypatch):
    captured: list[dict] = []

    async def fake_request(method, url, **kwargs):
        captured.append(kwargs.get("data", {}))
        return DummyResponse(200, {"access_token": "abc", "expires_in": 600})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    token = asyncio.run(client._refresh_access_token())
    assert captured and captured[0]["refresh_token"] == settings.schwab_refresh_token
    assert token.access_token == "abc"
    remaining = token.expires_at - dt.datetime.now(dt.timezone.utc)
    assert abs(remaining.total_seconds() - 540) < 5  # 60s skew applied


def test_price_history_happy_path(monkeypatch):
    schwab_client.clear_cached_token()
    candles = [
        {
            "datetime": 1_700_000_000_000,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10,
        }
    ]

    async def fake_request(method, url, **kwargs):
        if method == "POST":
            return DummyResponse(200, {"access_token": "token", "expires_in": 600})
        return DummyResponse(200, {"candles": candles})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    df = asyncio.run(client.get_price_history("SPY", start, end, "minute"))
    assert list(df.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert len(df) == 1
    assert df.index[0].tzinfo is not None and df.index[0].tzinfo.utcoffset(df.index[0]) == dt.timedelta(0)
    assert df.iloc[0]["Open"] == 1.0


def test_quote_happy_path(monkeypatch):
    schwab_client.clear_cached_token()
    payload = {
        "data": {
            "quotes": {
                "MSFT": {
                    "lastPrice": 350.12,
                    "quoteTimeInLong": 1_700_000_000_000,
                }
            }
        }
    }

    async def fake_request(method, url, **kwargs):
        if method == "POST":
            return DummyResponse(200, {"access_token": "token", "expires_in": 600})
        return DummyResponse(200, payload)

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    quote = asyncio.run(schwab_client.get_quote("MSFT"))
    assert quote["source"] == "schwab"
    assert quote["price"] == pytest.approx(350.12)
    assert quote["timestamp"].tzinfo is dt.timezone.utc


def test_price_history_error(monkeypatch):
    schwab_client.clear_cached_token()

    async def fake_request(method, url, **kwargs):
        if method == "POST":
            return DummyResponse(200, {"access_token": "token", "expires_in": 600})
        return DummyResponse(500, {"error": "rate limit"})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    with pytest.raises(SchwabAPIError):
        asyncio.run(client.get_price_history("AAPL", start, end, "minute"))

    schwab_client.clear_cached_token()

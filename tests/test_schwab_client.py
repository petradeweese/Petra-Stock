import asyncio
import asyncio
import datetime as dt
import grp
import json
import os
import pwd
import stat
import urllib.parse

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


def test_refresh_flow(monkeypatch, tmp_path):
    token_path = tmp_path / "schwab_tokens.json"
    monkeypatch.setattr(settings, "schwab_token_path", str(token_path), raising=False)
    monkeypatch.setattr(settings, "schwab_auth_mode", "basic", raising=False)

    captured: list[dict] = []

    async def fake_request(method, url, **kwargs):
        captured.append(
            {
                "data": kwargs.get("data"),
                "content": kwargs.get("content"),
                "headers": kwargs.get("headers", {}),
            }
        )
        return DummyResponse(200, {"access_token": "abc", "expires_in": 600})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    token = asyncio.run(client._refresh_access_token())

    assert captured
    assert len(captured) == 1
    request = captured[0]
    assert request["data"]
    parsed = request["data"]
    assert set(parsed.keys()) == {
        "grant_type",
        "refresh_token",
        "redirect_uri",
    }
    assert parsed.get("refresh_token") == settings.schwab_refresh_token
    assert parsed.get("grant_type") == "refresh_token"
    assert "client_id" not in parsed
    assert "client_secret" not in parsed
    auth_header = request["headers"].get("Authorization", "")
    assert auth_header.startswith("Basic ")
    assert (
        request["headers"].get("Content-Type")
        == "application/x-www-form-urlencoded"
    )

    assert token.access_token == "abc"
    remaining = token.expires_at - dt.datetime.now(dt.timezone.utc)
    assert abs(remaining.total_seconds() - 540) < 5  # 60s skew applied

    assert token_path.exists()
    stored = json.loads(token_path.read_text())
    assert stored["access_token"] == "abc"
    assert stored["refresh_token"] == settings.schwab_refresh_token
    assert "obtained_at" in stored
    stats = os.stat(token_path)
    assert stat.S_IMODE(stats.st_mode) == 0o600
    assert stats.st_uid == pwd.getpwnam("root").pw_uid
    try:
        ubuntu_gid = grp.getgrnam("ubuntu").gr_gid
    except KeyError:
        ubuntu_gid = None
    if ubuntu_gid is not None:
        assert stats.st_gid == ubuntu_gid


def test_refresh_flow_body_mode(monkeypatch, tmp_path):
    token_path = tmp_path / "schwab_tokens.json"
    monkeypatch.setattr(settings, "schwab_token_path", str(token_path), raising=False)
    monkeypatch.setattr(settings, "schwab_auth_mode", "body", raising=False)

    captured: list[dict] = []

    async def fake_request(method, url, **kwargs):
        captured.append(
            {
                "content": kwargs.get("content", ""),
                "headers": kwargs.get("headers", {}),
            }
        )
        return DummyResponse(200, {"access_token": "def", "expires_in": 300})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    token = asyncio.run(client._refresh_access_token())

    assert captured
    request = captured[0]
    parsed = urllib.parse.parse_qs(request["content"])
    assert parsed.get("client_id", [None])[0] == client._oauth_client_id
    assert parsed.get("client_secret", [None])[0] == settings.schwab_client_secret
    assert "Authorization" not in request["headers"]
    assert token.access_token == "def"


def test_refresh_invalid_grant_triggers_reauth(monkeypatch):
    async def fake_request(method, url, **kwargs):
        return DummyResponse(400, {"error": "invalid_grant"})

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    original_token = settings.schwab_refresh_token

    try:
        client.set_refresh_token("stale-token")
        settings.schwab_refresh_token = "stale-token"

        with pytest.raises(schwab_client.SchwabAuthError):
            asyncio.run(client._refresh_access_token())

        assert settings.schwab_refresh_token == ""
    finally:
        settings.schwab_refresh_token = original_token
        client.set_refresh_token(original_token)


def test_refresh_failure_cached(monkeypatch):
    calls = 0

    async def fake_request(method, url, **kwargs):
        nonlocal calls
        calls += 1
        return DummyResponse(
            400, {"error": "invalid_client", "error_description": "bad credentials"}
        )

    monkeypatch.setattr(schwab_client.http_client, "request", fake_request)
    client = schwab_client.SchwabClient()
    original_token = settings.schwab_refresh_token

    try:
        client.set_refresh_token("stale-token")
        settings.schwab_refresh_token = "stale-token"

        with pytest.raises(schwab_client.SchwabAuthError):
            asyncio.run(client._refresh_access_token())

        assert calls == 1

        with pytest.raises(schwab_client.SchwabAuthError):
            asyncio.run(client._refresh_access_token())

        assert calls == 1
    finally:
        settings.schwab_refresh_token = original_token
        client.set_refresh_token(original_token)


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

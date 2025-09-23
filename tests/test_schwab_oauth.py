from __future__ import annotations

import json
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from app import app
from config import settings
from db import get_db
from services import http_client
from services import schwab_client


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode()
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _extract_state(location: str) -> str:
    parsed = urllib.parse.urlparse(location)
    query = urllib.parse.parse_qs(parsed.query)
    return query.get("state", [""])[0]


def test_login_redirects_with_pkce(client: TestClient):
    resp = client.get("/schwab/login", follow_redirects=False)
    assert resp.status_code in (302, 303, 307)
    location = resp.headers.get("location")
    assert location
    parsed = urllib.parse.urlparse(location)
    assert parsed.path.endswith("/oauth2/v1/authorize")
    query = urllib.parse.parse_qs(parsed.query)
    assert query.get("client_id") == [settings.schwab_client_id]
    assert query.get("redirect_uri") == [settings.schwab_redirect_uri]
    assert query.get("response_type") == ["code"]
    assert query.get("code_challenge_method") == ["S256"]
    assert query.get("code_challenge")
    assert query.get("state")


def test_callback_exchanges_code_and_stores_refresh_token(monkeypatch, client: TestClient):
    login = client.get("/schwab/login", follow_redirects=False)
    state = _extract_state(login.headers["location"])

    captured: dict[str, object] = {}
    original_token = settings.schwab_refresh_token

    async def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["data"] = kwargs.get("data")
        return DummyResponse(200, {"refresh_token": "new-refresh", "access_token": "token"})

    monkeypatch.setattr(http_client, "request", fake_request)

    resp = client.get("/callback", params={"state": state, "code": "auth-code"})
    assert resp.status_code == 200
    assert "Schwab linked" in resp.text

    assert captured["method"] == "POST"
    assert captured["url"] == schwab_client.TOKEN_URL
    payload = captured["data"]
    assert isinstance(payload, dict)
    assert payload["code"] == "auth-code"
    assert payload["code_verifier"]

    gen = get_db()
    cursor = next(gen)
    try:
        cursor.execute(
            "SELECT provider, refresh_token FROM oauth_tokens WHERE provider=?", ("schwab",)
        )
        row = cursor.fetchone()
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
        gen.close()

    assert row is not None
    assert row["provider"] == "schwab"
    assert row["refresh_token"] == "new-refresh"
    assert settings.schwab_refresh_token == "new-refresh"

    # Restore configuration for subsequent tests.
    settings.schwab_refresh_token = original_token
    setattr(settings, "SCHWAB_REFRESH_TOKEN", original_token)
    schwab_client.update_refresh_token(original_token)


def test_callback_rejects_invalid_state(client: TestClient):
    resp = client.get("/callback", params={"state": "bogus", "code": "x"})
    assert resp.status_code == 400

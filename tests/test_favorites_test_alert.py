import sqlite3
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes


def setup_app(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    telem = {}
    monkeypatch.setattr(routes, "log_telemetry", lambda e: telem.update(e))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    return client, telem


def test_favorites_test_alert_mms(tmp_path, monkeypatch):
    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False):
        called.update({"symbol": symbol, "channel": channel, "compact": compact})
        return True, f"{symbol} {direction}\nWhy this contract: Δ 0.00"

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, telem = setup_app(tmp_path, monkeypatch)
    res = client.post(
        "/favorites/test_alert",
        json={"symbol": "AAPL", "channel": "mms", "compact": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert "Why this contract" in data["body"]
    assert called["channel"] == "mms"
    assert telem == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "mms",
        "compact": False,
        "ok": True,
    }


def test_favorites_test_alert_email(tmp_path, monkeypatch):
    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False):
        called.update({"symbol": symbol, "channel": channel, "compact": compact})
        return True, f"{symbol} {direction}\nWhy this contract: Δ 0.00"

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, telem = setup_app(tmp_path, monkeypatch)
    res = client.post(
        "/favorites/test_alert",
        json={"symbol": "AAPL", "channel": "email", "compact": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert "Why this contract" in data["body"]
    assert called["channel"] == "email"
    assert called["compact"] is True
    assert telem == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "email",
        "compact": True,
        "ok": True,
    }

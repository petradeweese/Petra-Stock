import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes


def test_test_alert_endpoint(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()

    called = {}

    def fake_enrich(ticker, direction):
        called["ticker"] = ticker
        called["direction"] = direction
        return True

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    res = client.post("/favorites/test_alert", json={"ticker": "AAPL", "direction": "UP"})
    assert res.status_code == 200
    assert res.json() == {"status": "sent", "ticker": "AAPL", "direction": "UP"}
    assert called == {"ticker": "AAPL", "direction": "UP"}

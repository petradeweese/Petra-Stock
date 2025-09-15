import sqlite3
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _setup(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def test_forward_list_favorites_present(tmp_path, monkeypatch):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('BBB','DOWN','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    conn.commit()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    # Simulate missing price data so forward tests aren't created; page should
    # still list favorites as queued items.
    monkeypatch.setattr(routes, "get_prices", lambda tickers, interval, start, end: {})
    monkeypatch.setattr(routes, "check_guardrails", lambda ticker: (True, []))
    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    data = res.json()
    assert len(data["favorites"]) == 2
    page = client.get("/forward")
    assert "AAA" in page.text and "BBB" in page.text


def test_forward_empty_state(tmp_path):
    _setup(tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    assert res.json()["favorites"] == []
    page = client.get("/forward")
    assert "No favorites to test" in page.text

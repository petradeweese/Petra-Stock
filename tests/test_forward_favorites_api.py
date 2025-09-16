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
    monkeypatch.setattr(routes, "get_prices", lambda tickers, interval, start, end: {})
    monkeypatch.setattr(routes, "check_guardrails", lambda ticker: (True, []))
    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    data = res.json()
    assert len(data["favorites"]) == 2
    assert {fav["lookback_years"] for fav in data["favorites"]} == {1.0}
    page = client.get("/forward")
    assert 'id="forward-tbody"' in page.text
    assert 'Run Forward Tests' in page.text
    assert 'static/js/forward.js' in page.text


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


def test_forward_run_endpoint(tmp_path, monkeypatch):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav1 = cur.lastrowid
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('BBB','DOWN','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav2 = cur.lastrowid
    conn.commit()
    called: list[int] = []

    def fake_create(db_cursor, fav):
        called.append(int(fav["id"]))

    monkeypatch.setattr(routes, "_create_forward_test", fake_create)
    monkeypatch.setattr(routes, "_update_forward_tests", lambda db: None)

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.post("/api/forward/run", json={"favorite_ids": [fav2]})
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert called == [fav2]

    called.clear()
    res = client.post("/api/forward/run", json={})
    assert res.status_code == 200
    body = res.json()
    assert body["queued"] == 2
    assert body["count"] == 2
    assert sorted(called) == sorted([fav1, fav2])


def test_forward_run_invalid_ids(tmp_path, monkeypatch):
    _setup(tmp_path)
    monkeypatch.setattr(routes, "_create_forward_test", lambda db, fav: None)
    monkeypatch.setattr(routes, "_update_forward_tests", lambda db: None)

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.post("/api/forward/run", json={"favorite_ids": ["abc"]})
    assert res.status_code == 400
    assert res.json()["ok"] is False

    res = client.post("/api/forward/run", json={"favorite_ids": [999]})
    assert res.status_code == 404
    assert res.json()["ok"] is False

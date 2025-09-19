import json
import sqlite3
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes
from services import forward_summary
from services.forward_runs import (
    forward_rule_hash,
    get_forward_history,
    log_forward_entry,
    log_forward_exit,
)


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
    assert all(fav["forward"] is None for fav in data["favorites"])
    for fav in data["favorites"]:
        summary = fav.get("summary")
        assert summary is not None
        assert summary["n"] == 0
        assert summary["hits"] == 0
        assert summary["hit_rate"] == 0.0
        assert summary["hit_lb95"] == 0.0
        assert summary["avg_roi"] is None
        assert summary["median_tt_bars"] is None
        assert summary.get("avg_dd") is None
        assert summary["mode"] == "off"
        assert summary["half_life_days"] == pytest.approx(
            routes.settings.forward_recency_halflife_days
        )
        assert summary["unweighted"]["n"] == 0
        assert summary["weighted"]["n"] == 0
    page = client.get("/forward")
    assert 'id="forward-tbody"' in page.text
    assert 'Run Forward Tests' in page.text
    assert 'static/js/forward.js' in page.text


def test_forward_favorites_include_forward_metrics(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',5.0)"
    )
    fav_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO forward_tests(
            fav_id, ticker, direction, interval, rule, version, entry_price,
            target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward,
            dd_forward, last_run_at, runs_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fav_id,
            'AAA',
            'UP',
            '15m',
            'r',
            2,
            100.0,
            1.0,
            0.5,
            60,
            'ok',
            5.5,
            75.0,
            2.0,
            '2024-01-01T10:30:00',
            3,
            '2024-01-01T10:00:00',
            '2024-01-01T10:30:00',
        ),
    )
    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    favorites = res.json()["favorites"]
    assert len(favorites) == 1
    assert favorites[0]["support_count"] == 1
    assert favorites[0]["support_display"] == "1"
    forward = favorites[0]["forward"]
    assert forward is not None
    assert forward["status"] == "ok"
    assert forward["version"] == 2
    assert forward["roi_pct"] == 5.5
    assert forward["hit_pct"] == 75.0
    assert forward["dd_pct"] == 2.0
    assert forward["runs_count"] == 3
    assert forward["created_at"] == "2024-01-01T10:00:00"
    assert forward["updated_at"] == "2024-01-01T10:30:00"
    assert forward["last_run_at"] == "2024-01-01T10:30:00"
    assert forward["events"] and forward["events"][0]["t"] == "entry"
    assert favorites[0]["forward_history"] == []
    summary = favorites[0]["summary"]
    assert summary["n"] == 0
    assert summary["hits"] == 0
    assert summary["hit_rate"] == 0.0
    assert summary["hit_lb95"] == 0.0
    assert summary["avg_roi"] is None
    assert summary["median_tt_bars"] is None
    assert summary.get("avg_dd") is None


def test_forward_events_shape(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_open = cur.lastrowid
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('BBB','DOWN','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_hit = cur.lastrowid

    cur.execute(
        """
        INSERT INTO forward_tests(
            fav_id, ticker, direction, interval, rule, version, entry_price,
            target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward,
            dd_forward, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fav_open,
            "AAA",
            "UP",
            "15m",
            "r",
            1,
            100.0,
            1.0,
            0.5,
            60,
            "running",
            0.0,
            None,
            0.0,
            "2025-01-02T13:15:00-05:00",
            "2025-01-02T13:30:00-05:00",
        ),
    )

    cur.execute(
        """
        INSERT INTO forward_tests(
            fav_id, ticker, direction, interval, rule, version, entry_price,
            target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward,
            dd_forward, time_to_hit, exit_reason, bars_to_exit, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fav_hit,
            "BBB",
            "DOWN",
            "15m",
            "r",
            1,
            200.0,
            1.0,
            0.5,
            60,
            "ok",
            5.0,
            100.0,
            0.5,
            45.0,
            "target",
            7,
            "2025-01-02T13:15:00-05:00",
            "2025-01-02T14:45:00-05:00",
        ),
    )

    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    favorites = res.json()["favorites"]
    assert len(favorites) == 2

    fav_map = {fav["ticker"]: fav for fav in favorites}

    open_events = fav_map["AAA"]["forward"]["events"]
    assert len(open_events) == 1
    assert open_events[0]["t"] == "entry"
    assert open_events[0]["ts"] == "2025-01-02T18:30:00Z"
    assert open_events[0]["px"] == 100.0

    hit_events = fav_map["BBB"]["forward"]["events"]
    assert len(hit_events) == 2
    assert hit_events[0]["t"] == "entry"
    assert hit_events[1]["t"] == "hit"
    assert hit_events[1]["ts"] == "2025-01-02T19:15:00Z"
    assert hit_events[1]["roi"] == 0.05
    assert hit_events[1]["tt_bars"] == 7

    entry_price = 200.0
    slip = routes.FORWARD_SLIPPAGE
    entry_fill = entry_price * (1 + slip * -1)
    exit_fill = entry_fill * (1 + (5.0 / 100.0) / -1)
    expected_exit = exit_fill / (1 - slip * -1)
    assert hit_events[1]["px"] == expected_exit

    for fav in favorites:
        summary = fav["summary"]
        assert summary["n"] == 0
        assert summary["hits"] == 0
        assert summary["hit_rate"] == 0.0
        assert summary["hit_lb95"] == 0.0
        assert summary["avg_roi"] is None
        assert summary["median_tt_bars"] is None
        assert summary.get("avg_dd") is None


def test_forward_history_included(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_id = cur.lastrowid
    conn.commit()

    current_hash = forward_rule_hash("r")
    log_forward_entry(
        cur,
        fav_id,
        "2025-09-16T13:15:00-04:00",
        188.42,
        current_hash,
    )
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-16T13:15:00-04:00",
        "2025-09-16T14:45:00-04:00",
        190.7,
        "target",
        0.0123,
        7,
        0.0045,
    )

    log_forward_entry(
        cur,
        fav_id,
        "2025-09-15T09:30:00-04:00",
        187.1,
        forward_rule_hash("old"),
    )
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-15T09:30:00-04:00",
        "2025-09-15T10:45:00-04:00",
        185.0,
        "stop",
        -0.008,
        5,
        0.006,
    )
    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    favorites = res.json()["favorites"]
    assert len(favorites) == 1
    history = favorites[0]["forward_history"]
    assert len(history) == 2
    assert history[0]["entry_ts"] == "2025-09-16T13:15:00-04:00"
    assert history[0]["outcome"] == "hit"
    assert history[0]["roi"] == 0.0123
    assert history[0]["tt_bars"] == 7
    assert history[0]["rule_mismatch"] is False
    assert history[1]["rule_mismatch"] is True

    summary = favorites[0]["summary"]
    assert summary["n"] == 2
    assert summary["hits"] == 1
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["hit_lb95"] == pytest.approx(routes._wilson_lb95(1, 2))
    assert summary["avg_roi"] == pytest.approx((0.0123 - 0.008) / 2)
    assert summary["median_tt_bars"] == 6
    assert summary["avg_dd"] == pytest.approx((0.0045 + 0.006) / 2)
    assert summary["mode"] == "off"
    assert summary["weighted"]["n"] == 2
    assert summary["weighted"]["hits"] == 1
    assert summary["weighted"]["hit_rate"] == pytest.approx(0.5)
    assert summary["weighted"]["hit_lb95"] == pytest.approx(routes._wilson_lb95(1, 2))
    assert summary["weighted"]["avg_roi"] == pytest.approx((0.0123 - 0.008) / 2)
    assert summary["weighted"]["median_tt_bars"] == 6
    assert summary["weighted"]["avg_dd"] == pytest.approx((0.0045 + 0.006) / 2)

    direct_history = get_forward_history(str(fav_id), limit=5)
    assert [row["entry_ts"] for row in direct_history] == [
        "2025-09-16T13:15:00-04:00",
        "2025-09-15T09:30:00-04:00",
    ]


def test_forward_summary_recency_weighting(tmp_path, monkeypatch):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_id = cur.lastrowid
    conn.commit()

    current_hash = forward_rule_hash("r")
    log_forward_entry(
        cur,
        fav_id,
        "2025-09-19T10:00:00-04:00",
        190.0,
        current_hash,
    )
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-19T10:00:00-04:00",
        "2025-09-19T12:00:00-04:00",
        192.0,
        "hit",
        0.02,
        4,
        0.005,
    )

    log_forward_entry(
        cur,
        fav_id,
        "2025-08-21T09:30:00-04:00",
        188.0,
        current_hash,
    )
    log_forward_exit(
        cur,
        fav_id,
        "2025-08-21T09:30:00-04:00",
        "2025-08-21T12:00:00-04:00",
        184.0,
        "stop",
        -0.01,
        9,
        0.007,
    )
    conn.commit()

    monkeypatch.setattr(routes.settings, "forward_recency_mode", "exp", raising=False)
    monkeypatch.setattr(routes.settings, "FORWARD_RECENCY_MODE", "exp", raising=False)
    monkeypatch.setattr(routes.settings, "forward_recency_halflife_days", 10.0, raising=False)
    monkeypatch.setattr(routes.settings, "FORWARD_RECENCY_HALFLIFE_DAYS", 10.0, raising=False)

    fixed_now = datetime(2025, 9, 20, 12, 0, tzinfo=routes.TZ)
    monkeypatch.setattr(routes, "now_et", lambda: fixed_now)
    forward_summary._summary_cache.clear()
    monkeypatch.setattr(forward_summary, "now_et", lambda: fixed_now)

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    favorites = res.json()["favorites"]
    assert len(favorites) == 1
    summary = favorites[0]["summary"]

    weight_recent = 0.5 ** (1 / 10.0)
    weight_old = 0.5 ** (30 / 10.0)
    weight_sum = weight_recent + weight_old
    expected_hit_rate = weight_recent / weight_sum
    expected_avg_roi = (weight_recent * 0.02 + weight_old * -0.01) / weight_sum
    expected_avg_dd = (weight_recent * 0.005 + weight_old * 0.007) / weight_sum

    assert summary["mode"] == "exp"
    assert summary["half_life_days"] == pytest.approx(10.0)
    assert summary["n"] == 2
    assert summary["hits"] == 1
    assert summary["unweighted"]["n"] == 2
    assert summary["weighted"]["n"] == pytest.approx(weight_sum)
    assert summary["weighted"]["hits"] == pytest.approx(weight_recent)
    assert summary["weighted"]["hit_rate"] == pytest.approx(expected_hit_rate)
    assert summary["weighted"]["hit_lb95"] == pytest.approx(routes._wilson_lb95(1, 1))
    assert summary["weighted"]["avg_roi"] == pytest.approx(expected_avg_roi)
    assert summary["weighted"]["median_tt_bars"] == 4
    assert summary["weighted"]["avg_dd"] == pytest.approx(expected_avg_dd)


def test_forward_runs_api_pagination(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_id = cur.lastrowid
    current_hash = forward_rule_hash("r")
    log_forward_entry(cur, fav_id, "2025-09-16T13:15:00-04:00", 188.42, current_hash)
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-16T13:15:00-04:00",
        "2025-09-16T14:45:00-04:00",
        190.7,
        "hit",
        0.0123,
        7,
        0.004,
    )
    log_forward_entry(cur, fav_id, "2025-09-15T09:30:00-04:00", 187.1, current_hash)
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-15T09:30:00-04:00",
        "2025-09-15T10:45:00-04:00",
        185.0,
        "stop",
        -0.008,
        5,
        0.006,
    )
    log_forward_entry(cur, fav_id, "2025-09-14T09:30:00-04:00", 186.0, current_hash)
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-14T09:30:00-04:00",
        "2025-09-14T10:30:00-04:00",
        186.5,
        "timeout",
        0.0025,
        4,
        0.001,
    )
    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get(f"/api/forward/{fav_id}?limit=2")
    assert res.status_code == 200
    first_page = res.json()
    assert isinstance(first_page, list)
    assert len(first_page) == 2
    assert first_page[0]["entry_ts"] == "2025-09-16T13:15:00-04:00"
    assert first_page[0]["roi"] == pytest.approx(0.0123)
    assert first_page[0].get("favorite_id") is None
    assert first_page[0]["rule_mismatch"] is False
    assert first_page[1]["entry_ts"] == "2025-09-15T09:30:00-04:00"
    first_etag = res.headers.get("etag")
    assert first_etag

    res = client.get(f"/api/forward/{fav_id}?limit=2&offset=2")
    assert res.status_code == 200
    next_page = res.json()
    assert len(next_page) == 1
    assert next_page[0]["entry_ts"] == "2025-09-14T09:30:00-04:00"
    assert next_page[0]["outcome"] == "timeout"

    res = client.get(f"/api/forward/{fav_id}?limit=0")
    assert res.status_code == 200
    assert len(res.json()) == 3

    res = client.get(f"/api/forward/{fav_id}?limit=5000")
    assert res.status_code == 200
    assert len(res.json()) == 3

    res = client.get(
        f"/api/forward/{fav_id}?limit=2",
        headers={"If-None-Match": first_etag},
    )
    assert res.status_code == 304
    assert res.headers.get("etag") == first_etag

    res = client.get("/api/forward/9999")
    assert res.status_code == 404


def test_forward_runs_export_csv(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,target_pct,stop_pct,window_value,window_unit,lookback_years) VALUES('AAA','UP','15m','r',1.0,0.5,4,'Hours',1.0)"
    )
    fav_id = cur.lastrowid
    hash_val = forward_rule_hash("r")
    log_forward_entry(cur, fav_id, "2025-09-16T13:15:00-04:00", 188.42, hash_val)
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-16T13:15:00-04:00",
        "2025-09-16T14:45:00-04:00",
        190.7,
        "hit",
        0.0123,
        7,
        0.004,
    )
    log_forward_entry(cur, fav_id, "2025-09-15T09:30:00-04:00", 187.1, hash_val)
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-15T09:30:00-04:00",
        "2025-09-15T10:45:00-04:00",
        185.0,
        "stop",
        -0.008,
        5,
        0.006,
    )
    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get(f"/forward/export.csv?favorite_id={fav_id}&limit=5")
    assert res.status_code == 200
    assert res.headers.get("content-type", "").startswith("text/csv")
    lines = [line for line in res.text.strip().splitlines() if line]
    assert lines[0] == "symbol,direction,entry_ts,entry_px,exit_ts,exit_px,outcome,roi,tt_bars,dd,rule_hash"
    assert len(lines) == 3
    assert lines[1].startswith("AAA,UP,2025-09-16")

    res = client.get("/forward/export.csv?favorite_id=9999")
    assert res.status_code == 404


def test_forward_runs_owner_validation(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute("ALTER TABLE favorites ADD COLUMN owner_id TEXT")
    cur.execute(
        """
        INSERT INTO favorites(
            ticker,direction,interval,rule,target_pct,stop_pct,
            window_value,window_unit,lookback_years,owner_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        ("AAA", "UP", "15m", "r", 1.0, 0.5, 4, "Hours", 1.0, "user-1"),
    )
    fav_id = cur.lastrowid
    conn.commit()

    log_forward_entry(cur, fav_id, "2025-09-16T13:15:00Z", 188.0, forward_rule_hash("r"))
    log_forward_exit(
        cur,
        fav_id,
        "2025-09-16T13:15:00Z",
        "2025-09-16T14:45:00Z",
        190.0,
        "hit",
        0.01,
        5,
        0.002,
    )
    conn.commit()

    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user_id = request.headers.get("x-user-id") or "user-1"
        return await call_next(request)

    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get(f"/api/forward/{fav_id}", headers={"x-user-id": "user-1"})
    assert res.status_code == 200
    res = client.get(f"/api/forward/{fav_id}", headers={"x-user-id": "user-2"})
    assert res.status_code == 404

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
    assert "No forward runs yet—your history appears as windows complete." in page.text


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


def test_forward_favorites_use_snapshot_params(tmp_path):
    conn, cur = _setup(tmp_path)
    settings = {
        "target_pct": "2.5",
        "stop_pct": "0.9",
        "window_value": "8",
        "window_unit": "Minutes",
        "lookback_years": "3.0",
    }
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,settings_json_snapshot) VALUES (?,?,?,?,?)",
        ("CCC", "UP", "15m", "r", json.dumps(settings)),
    )
    conn.commit()

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/api/forward/favorites")
    assert res.status_code == 200
    favorites = res.json()["favorites"]
    assert len(favorites) == 1
    fav = favorites[0]
    assert fav["target_pct"] == 2.5
    assert fav["stop_pct"] == 0.9
    assert fav["window_value"] == 8.0
    assert fav["window_unit"] == "Minutes"
    assert fav["lookback_years"] == 3.0

import json
import sqlite3
import logging
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles

import db
from db import get_db
from routes import router
from services.overnight import OvernightRunner, in_window


def test_window_enforcement():
    dt = datetime(2024, 1, 1, 2, 0)
    assert in_window(dt, "01:00", "08:00")
    dt = datetime(2024, 1, 1, 9, 0)
    assert not in_window(dt, "01:00", "08:00")
    dt = datetime(2024, 1, 1, 23, 0)
    assert in_window(dt, "22:00", "02:00")
    dt = datetime(2024, 1, 2, 3, 0)
    assert not in_window(dt, "22:00", "02:00")


def test_sequential_order():
    batch_id = "b1"
    gen = get_db()
    db = next(gen)
    db.execute(
        "INSERT INTO overnight_batches(id,status,start_override) VALUES(?, 'queued', 1)",
        (batch_id,),
    )
    items = [
        ("i1", 1, {"ticker": "T1"}),
        ("i2", 2, {"ticker": "T2"}),
        ("i3", 3, {"ticker": "T3"}),
    ]
    for iid, pos, payload in items:
        db.execute(
            "INSERT INTO overnight_items(id,batch_id,position,payload_json,status) VALUES(?,?,?,?, 'queued')",
            (iid, batch_id, pos, json.dumps(payload)),
        )
    db.connection.commit()
    gen.close()
    order: list[str] = []

    def _stub_scan(payload: dict, silent: bool) -> int:
        order.append(payload["ticker"])
        return int(payload["ticker"][1:])

    runner = OvernightRunner(_stub_scan)
    runner.run_batch(batch_id)

    gen = get_db()
    db = next(gen)
    rows = db.execute(
        "SELECT position, status, run_id FROM overnight_items ORDER BY position",
    ).fetchall()
    gen.close()
    assert order == ["T1", "T2", "T3"]
    assert [r[1] for r in rows] == ["complete", "complete", "complete"]
    assert [r[2] for r in rows] == [1, 2, 3]


def test_overnight_uses_scanner_payload(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    form = {
        "scan_type": "single",
        "ticker": "AAPL",
        "interval": "15m",
        "direction": "UP",
        "target_pct": "1.5",
    }
    res = client.post("/overnight/batches", json={"items": [form]})
    batch_id = res.json()["batch_id"]
    captured: list[dict] = []

    def _scan(p: dict, silent: bool) -> int:
        captured.append(p)
        return 1

    runner = OvernightRunner(_scan)
    runner.run_batch(batch_id)
    from services.scanner_params import coerce_scan_params
    assert captured[0] == coerce_scan_params(form)


def test_overnight_save_and_reload(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    form = {
        "scan_type": "single",
        "ticker": "MSFT",
        "interval": "15m",
        "direction": "DOWN",
        "target_pct": "2.0",
        "delta": "0.3",
        "theta_day": "0.1",
    }
    res = client.post("/overnight/batches", json={"items": [form]})
    batch_id = res.json()["batch_id"]
    data = client.get(f"/overnight/batches/{batch_id}").json()
    payload = json.loads(data["items"][0]["payload_json"])
    from services.scanner_params import coerce_scan_params

    assert payload == coerce_scan_params(form)


def _setup_app(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    app = FastAPI()
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app


def test_overnight_page_has_full_form(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.get("/overnight")
    assert res.status_code == 200
    text = res.text
    assert "Scan Type" in text
    assert "Delta (assumed)" in text


def test_overnight_add_remove_renumber(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    payload = {
        "label": "b",
        "items": [{"scan_type": "scan150"}, {"scan_type": "scan150"}],
    }
    res = client.post("/overnight/batches", json=payload)
    batch_id = res.json()["batch_id"]
    client.post(f"/overnight/batches/{batch_id}/items", json={"scan_type": "scan150"})
    items = client.get(f"/overnight/batches/{batch_id}").json()["items"]
    second_id = items[1]["id"]
    client.delete(f"/overnight/items/{second_id}")
    items = client.get(f"/overnight/batches/{batch_id}").json()["items"]
    mapping = [{"item_id": items[1]["id"], "position": 1}]
    client.post(f"/overnight/batches/{batch_id}/reorder", json=mapping)
    items = client.get(f"/overnight/batches/{batch_id}").json()["items"]
    positions = [it["position"] for it in items]
    assert positions == [1, 2]


def test_start_now(tmp_path, caplog):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res1 = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}]})
    b1 = res1.json()["batch_id"]
    res2 = client.post("/overnight/batches", json={"items": [{"ticker": "T2"}]})
    b2 = res2.json()["batch_id"]
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE overnight_batches SET status='running' WHERE id=?", (b1,))
    conn.commit()
    conn.close()
    with caplog.at_level(logging.INFO):
        resp = client.post(f"/overnight/batches/{b2}/start_now")
    assert resp.json()["status"] == "queued_after_current"
    assert any("overnight_override_start" in r.message for r in caplog.records)
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE overnight_batches SET status='complete' WHERE id=?", (b1,))
    conn.commit()
    conn.close()
    order: list[str] = []

    def _scan(p: dict, silent: bool) -> int:
        order.append(p["ticker"])
        return int(p["ticker"][1:])

    runner = OvernightRunner(_scan)
    now = datetime(2024, 1, 1, 9, 0)
    runner.run_ready(now)
    assert order == ["T2"]
    gen = get_db()
    conn_db = next(gen)
    val = conn_db.execute(
        "SELECT start_override FROM overnight_batches WHERE id=?", (b2,)
    ).fetchone()[0]
    gen.close()
    assert val == 0


def test_silent_mode(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}]})
    batch_id = res.json()["batch_id"]
    overlay: list[bool] = []

    def _scan(payload: dict, silent: bool) -> int:
        overlay.append(not silent)
        return 123

    _scan({"ticker": "T0"}, False)
    runner = OvernightRunner(_scan)
    runner.run_batch(batch_id)
    assert overlay == [True, False]


def test_archive_to_favorites(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    # create archived run
    rows = [
        {
            "ticker": "AAA",
            "direction": "UP",
            "avg_roi_pct": 1.0,
            "hit_pct": 50.0,
            "support": 1,
            "avg_dd_pct": 0.1,
            "stability": 0.1,
            "rule": "r1",
        }
    ]
    client.post("/archive/save", json={"params": {}, "rows": rows})
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    run_id = cur.execute("SELECT id FROM runs").fetchone()[0]
    conn.close()
    html = client.get(f"/results/{run_id}").text
    assert "⭐" in html
    fav_payload = {
        "ticker": "AAA",
        "direction": "UP",
        "rule": "r1",
    }
    resp = client.post("/favorites/add", json=fav_payload)
    assert resp.json()["ok"]
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM favorites")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_prefs_update(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    resp = client.get("/overnight/prefs")
    assert resp.json()["window_start"] == "01:00"
    client.post(
        "/overnight/prefs", json={"window_start": "02:00", "window_end": "03:00"}
    )
    resp = client.get("/overnight/prefs")
    data = resp.json()
    assert data["window_start"] == "02:00" and data["window_end"] == "03:00"


def test_csv_export(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post(
        "/overnight/batches", json={"items": [{"scan_type": "single", "ticker": "XYZ"}]}
    )
    batch_id = res.json()["batch_id"]
    runner = OvernightRunner(lambda p, s: 42)
    runner.run_batch(batch_id)
    conn = sqlite3.connect(db.DB_PATH)
    run_id = conn.execute(
        "SELECT run_id FROM overnight_items WHERE batch_id=?", (batch_id,)
    ).fetchone()[0]
    conn.close()
    csv_text = client.get(f"/overnight/batches/{batch_id}/csv").text
    assert f"1,single,XYZ,{run_id}" in csv_text


def test_window_close_mid_item(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post(
        "/overnight/batches", json={"items": [{"ticker": "T1"}, {"ticker": "T2"}]}
    )
    batch_id = res.json()["batch_id"]
    order: list[str] = []

    def _scan(p: dict, s: bool) -> int:
        order.append(p["ticker"])
        return int(p["ticker"][1:])

    runner = OvernightRunner(_scan)

    def fake_window(db):
        return ("00:00", "00:00")

    runner._get_window = fake_window  # type: ignore
    runner.run_batch(batch_id)
    gen = get_db()
    conn_db = next(gen)
    rows = conn_db.execute(
        "SELECT position, status FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    batch_status = conn_db.execute(
        "SELECT status FROM overnight_batches WHERE id=?", (batch_id,)
    ).fetchone()[0]
    gen.close()
    assert order == ["T1"]
    assert rows[0][1] == "complete" and rows[1][1] == "queued"
    assert batch_status == "paused"


def test_pause_resume_cancel(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}]})
    batch_id = res.json()["batch_id"]
    order: list[str] = []

    def _scan(p: dict, s: bool) -> int:
        order.append(p["ticker"])
        return int(p["ticker"][1:])

    runner = OvernightRunner(_scan)
    now = datetime(2024, 1, 1, 2, 0)
    client.post(f"/overnight/batches/{batch_id}/pause")
    assert runner.run_ready(now) is None
    client.post(f"/overnight/batches/{batch_id}/resume")
    runner.run_ready(now)
    assert order == ["T1"]
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T2"}]})
    b2 = res.json()["batch_id"]
    client.post(f"/overnight/batches/{b2}/cancel")
    status = client.get(f"/overnight/batches/{b2}").json()["batch"]["status"]
    assert status == "canceled"


def test_restart_resume_interrupted(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}, {"ticker": "T2"}]})
    batch_id = res.json()["batch_id"]
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    first_id = cur.execute(
        "SELECT id FROM overnight_items WHERE batch_id=? AND position=1",
        (batch_id,),
    ).fetchone()[0]
    cur.execute(
        "UPDATE overnight_items SET status='running', started_at='2024-01-01T00:00:00' WHERE id=?",
        (first_id,),
    )
    cur.execute(
        "UPDATE overnight_batches SET status='running' WHERE id=?",
        (batch_id,),
    )
    conn.commit()
    conn.close()
    order: list[str] = []

    def _scan(p: dict, s: bool) -> int:
        order.append(p["ticker"])
        return int(p["ticker"][1:])

    runner = OvernightRunner(_scan)
    runner.run_ready(datetime(2024, 1, 1, 2, 0))
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT position, status FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    conn.close()
    assert order == ["T2"]
    assert rows[0][1] == "failed"
    assert rows[1][1] == "complete"


def test_cancel_marks_remaining(tmp_path, monkeypatch):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post(
        "/overnight/batches", json={"items": [{"ticker": "T1"}, {"ticker": "T2"}]}
    )
    batch_id = res.json()["batch_id"]
    order: list[str] = []

    def _scan(p: dict, s: bool) -> int:
        order.append(p["ticker"])
        if p["ticker"] == "T1":
            client.post(f"/overnight/batches/{batch_id}/cancel")
        return int(p["ticker"][1:])

    import services.overnight as ov_mod

    monkeypatch.setattr(ov_mod, "in_window", lambda *a, **k: True)
    runner = OvernightRunner(_scan)
    runner.run_batch(batch_id)
    gen = get_db()
    conn_db = next(gen)
    rows = conn_db.execute(
        "SELECT position, status FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    gen.close()
    assert order == ["T1"]
    assert [r[1] for r in rows] == ["complete", "canceled"]


def test_runner_state_endpoint(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    resp = client.get("/overnight/batches")
    data = resp.json()
    assert data["runner"]["state"] == "idle"


def test_restart_resume_completed_run(tmp_path):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}, {"ticker": "T2"}]})
    batch_id = res.json()["batch_id"]
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs(id, started_at, finished_at) VALUES(1,'2024-01-01','2024-01-01')"
    )
    first_id = cur.execute(
        "SELECT id FROM overnight_items WHERE batch_id=? AND position=1",
        (batch_id,),
    ).fetchone()[0]
    cur.execute(
        "UPDATE overnight_items SET status='running', run_id=1 WHERE id=?",
        (first_id,),
    )
    cur.execute(
        "UPDATE overnight_batches SET status='running' WHERE id=?",
        (batch_id,),
    )
    conn.commit()
    conn.close()
    order: list[str] = []

    def _scan(p: dict, s: bool) -> int:
        order.append(p["ticker"])
        return int(p["ticker"][1:])

    runner = OvernightRunner(_scan)
    runner.run_ready(datetime(2024, 1, 1, 2, 0))
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT position, status FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    conn.close()
    assert order == ["T2"]
    assert rows[0][1] == "complete"
    assert rows[1][1] == "complete"


def test_telemetry_fields(tmp_path, caplog):
    app = _setup_app(tmp_path)
    client = TestClient(app)
    res = client.post("/overnight/batches", json={"items": [{"ticker": "T1"}]})
    batch_id = res.json()["batch_id"]

    def _scan(p: dict, s: bool) -> int:
        return 42

    runner = OvernightRunner(_scan)
    with caplog.at_level(logging.INFO):
        runner.run_batch(batch_id)
    records = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    item = next(m for m in records if m.get("type") == "overnight_item")
    assert item["run_id"] == 42
    assert item["silent"] is True
    assert "position" in item and "queue_wait" in item["timings_ms"] and "scan" in item["timings_ms"]

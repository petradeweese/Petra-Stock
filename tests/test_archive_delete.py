import sqlite3

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
from routes.archive import router as archive_router


def _setup(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def _make_app():
    app = FastAPI()
    app.include_router(archive_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app


def test_archive_delete_single_run(tmp_path):
    conn, cur = _setup(tmp_path)
    cur.execute(
        """INSERT INTO runs(started_at,scan_type,params_json,universe,finished_at,hit_count,settings_json)
            VALUES('2024','scan150','{}','AAPL','2024',1,'{}')"""
    )
    run_id = cur.lastrowid
    cur.execute(
        """INSERT INTO run_results(run_id,ticker,direction,avg_roi_pct,hit_pct,support,avg_tt,avg_dd_pct,stability,rule)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (run_id, "AAPL", "UP", 0.1, 50.0, 1, 0.0, 0.0, 0.0, "r"),
    )
    conn.commit()
    app = _make_app()
    client = TestClient(app)

    res = client.delete(f"/api/archive/{run_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True

    cur.execute("SELECT * FROM runs WHERE id=?", (run_id,))
    assert cur.fetchone() is None
    cur.execute("SELECT * FROM run_results WHERE run_id=?", (run_id,))
    assert cur.fetchone() is None


def test_archive_delete_missing_run(tmp_path):
    _setup(tmp_path)
    app = _make_app()
    client = TestClient(app)

    res = client.delete("/api/archive/999")
    assert res.status_code == 404
    assert res.json()["ok"] is False


def test_archive_clear_all(tmp_path):
    conn, cur = _setup(tmp_path)
    for ticker in ("AAA", "BBB"):
        cur.execute(
            """INSERT INTO runs(started_at,scan_type,params_json,universe,finished_at,hit_count,settings_json)
                VALUES('2024','scan150','{}',?, '2024',1,'{}')""",
            (ticker,),
        )
        run_id = cur.lastrowid
        cur.execute(
            """INSERT INTO run_results(run_id,ticker,direction,avg_roi_pct,hit_pct,support,avg_tt,avg_dd_pct,stability,rule)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (run_id, ticker, "UP", 0.1, 50.0, 1, 0.0, 0.0, 0.0, "r"),
        )
    conn.commit()

    app = _make_app()
    client = TestClient(app)

    res = client.post("/api/archive/clear")
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["cleared"] == 2

    cur.execute("SELECT COUNT(*) FROM runs")
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM run_results")
    assert cur.fetchone()[0] == 0

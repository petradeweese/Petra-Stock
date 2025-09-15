import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
from routes.archive import router as archive_router


def test_results_page_renders_for_run(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
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
    app = FastAPI()
    app.include_router(archive_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    res = client.get(f"/results/{run_id}")
    assert res.status_code == 200

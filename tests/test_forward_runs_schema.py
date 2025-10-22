import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes


def test_forward_favorites_bootstraps_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db, "_ENGINE", None, raising=False)
    db.init_db()

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.get("/api/forward/favorites")
    assert response.status_code == 200

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forward_runs'")
        assert cur.fetchone() is not None
    finally:
        conn.close()

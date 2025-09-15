from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def test_history_removed(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    page = client.get("/archive")
    assert "History" not in page.text
    res = client.get("/history", follow_redirects=False)
    assert res.status_code in (404, 301, 302, 308)

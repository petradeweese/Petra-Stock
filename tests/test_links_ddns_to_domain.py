from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _client(tmp_path):
    db.DB_PATH = str(tmp_path / "links.db")
    db.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return TestClient(app)


def test_no_ddns_links_in_public_pages(tmp_path):
    client = _client(tmp_path)
    for path in ["/", "/about", "/contact", "/privacy", "/terms", "/sms-consent"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "ddns" not in res.text.lower()
        assert "petrastock.ddns.net" not in res.text

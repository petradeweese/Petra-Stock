import sqlite3

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _setup_app(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,lookback_years,min_support,support_snapshot) VALUES('AAA','UP','15m','r',1.5,30,45)"
    )
    conn.commit()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return TestClient(app)


def test_favorites_list_shows_lookback_hits(tmp_path):
    client = _setup_app(tmp_path)
    res = client.get("/favorites")
    assert res.status_code == 200
    text = res.text
    assert "Lookback" in text
    assert "Hits" in text
    assert "1.5" in text
    assert ">45<" in text


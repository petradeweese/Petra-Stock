import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
import routes
from routes import favorites_delete_duplicates


def test_delete_duplicates(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES ('AAA','UP','15m','r1')"
    )
    cur.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES ('AAA','UP','15m','r1')"
    )
    cur.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES ('BBB','DOWN','15m','r2')"
    )
    conn.commit()

    favorites_delete_duplicates(db=cur)

    cur.execute("SELECT COUNT(*) FROM favorites")
    count = cur.fetchone()[0]
    assert count == 2
    conn.close()


def test_add_favorite_ref_avg_dd(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    res = client.post(
        "/favorites/add",
        json={"ticker": "AAA", "direction": "UP", "rule": "r1", "ref_avg_dd": 5.0},
    )
    assert res.status_code == 200
    assert res.json()["ok"]

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ref_avg_dd FROM favorites WHERE ticker='AAA'")
    val = cur.fetchone()[0]
    assert val == 0.05
    conn.close()


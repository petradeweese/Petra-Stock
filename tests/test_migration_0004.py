import json
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _alembic_config(db_path: Path) -> Config:
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def test_migration_0004_backfills_lookback(tmp_path):
    db_path = tmp_path / "mig.db"
    db.DB_PATH = str(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "0003")

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(ticker,direction,interval,rule,lookback_years,support_snapshot) VALUES(?,?,?,?,?,?)",
        (
            "AAA",
            "UP",
            "15m",
            "r1",
            0.2,
            json.dumps({"count": 20, "lookback_years": 2.5}),
        ),
    )
    conn.commit()

    command.upgrade(cfg, "head")

    cur.execute("SELECT lookback_years FROM favorites WHERE ticker='AAA'")
    lookback = cur.fetchone()[0]
    conn.close()

    assert lookback == 2.5


def test_migration_0004_handles_missing_support_snapshot(tmp_path):
    db_path = tmp_path / "legacy.db"
    db.DB_PATH = str(db_path)

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            interval TEXT NOT NULL,
            rule TEXT NOT NULL,
            lookback_years REAL,
            min_support INTEGER,
            params_json TEXT,
            roi_snapshot REAL,
            hit_pct_snapshot REAL,
            dd_pct_snapshot REAL,
            rule_snapshot TEXT,
            settings_json_snapshot TEXT,
            snapshot_at TEXT
        )
        """
    )
    cur.execute(
        """
        INSERT INTO favorites(
            ticker, direction, interval, rule, lookback_years, min_support, params_json
        )
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            "BBB",
            "DOWN",
            "15m",
            "r2",
            0.0,
            25,
            json.dumps({"lookback_years": 3.5}),
        ),
    )
    conn.commit()
    conn.close()

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(favorites)")
    cols = {row[1] for row in cur.fetchall()}
    assert "support_snapshot" in cols

    cur.execute(
        "SELECT lookback_years, support_snapshot FROM favorites WHERE ticker='BBB'"
    )
    lookback, snapshot = cur.fetchone()
    conn.close()

    assert lookback == 3.5
    assert snapshot is None


def test_migration_0004_smoke_app_without_snapshot(tmp_path):
    db_path = tmp_path / "legacy_app.db"
    db.DB_PATH = str(db_path)

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            interval TEXT NOT NULL,
            rule TEXT NOT NULL,
            lookback_years REAL,
            min_support INTEGER,
            params_json TEXT,
            roi_snapshot REAL,
            hit_pct_snapshot REAL,
            dd_pct_snapshot REAL,
            rule_snapshot TEXT,
            settings_json_snapshot TEXT,
            snapshot_at TEXT
        )
        """
    )
    cur.execute(
        """
        INSERT INTO favorites(
            ticker, direction, interval, rule, lookback_years, min_support, params_json
        )
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            "AAA",
            "UP",
            "15m",
            "r1",
            0.0,
            30,
            json.dumps({"lookback_years": 4.0}),
        ),
    )
    conn.commit()
    conn.close()

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "head")

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)

    res = client.get("/favorites")
    assert res.status_code == 200
    body = res.text
    assert "Favorites" in body
    assert "4y" in body


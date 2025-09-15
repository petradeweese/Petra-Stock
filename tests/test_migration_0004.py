import json
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

import db


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


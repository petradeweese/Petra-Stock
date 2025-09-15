import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

import db


def _alembic_config(db_path: Path) -> Config:
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def test_migration_0005_adds_deleted_at(tmp_path):
    db_path = tmp_path / "mig.db"
    db.DB_PATH = str(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "0004")

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO overnight_batches(id,status) VALUES(?, 'complete')",
        ("batch-one",),
    )
    conn.commit()
    conn.close()

    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    columns = [row[1] for row in cur.execute("PRAGMA table_info(overnight_batches)")]
    assert "deleted_at" in columns
    cur.execute(
        "UPDATE overnight_batches SET deleted_at='2024-01-01T00:00:00' WHERE id=?",
        ("batch-one",),
    )
    conn.commit()
    cur.execute("SELECT deleted_at FROM overnight_batches WHERE id=?", ("batch-one",))
    assert cur.fetchone()[0] == "2024-01-01T00:00:00"
    conn.close()

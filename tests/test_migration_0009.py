from pathlib import Path
import sqlite3

from alembic import command
from alembic.config import Config

import db


def _alembic_config(db_path: Path) -> Config:
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def test_migration_0009_creates_and_drops_tables(tmp_path):
    db_path = tmp_path / "paper_mig.db"
    db.DB_PATH = str(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "0008")
    command.upgrade(cfg, "0009")

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
    assert cur.fetchone() is not None
    cur.execute("PRAGMA table_info(paper_trades)")
    cols = {row[1] for row in cur.fetchall()}
    assert {"executed_at", "interval", "ticker"}.issubset(cols)
    cur.execute("PRAGMA index_list('paper_trades')")
    indexes = {row[1] for row in cur.fetchall()}
    assert "ix_paper_trades_ticker" in indexes
    assert "ix_paper_trades_executed_at" in indexes

    command.downgrade(cfg, "0008")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
    assert cur.fetchone() is None
    conn.close()

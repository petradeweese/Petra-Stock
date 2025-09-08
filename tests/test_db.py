import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db


def test_init_db_creates_settings(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    gen = db.get_db()
    cur = next(gen)
    settings = db.get_settings(cur)
    assert settings["id"] == 1
    gen.close()

import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
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


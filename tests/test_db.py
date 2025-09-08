import sqlite3
import app


def _prepare_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    for stmt in app.SCHEMA[:2]:
        cur.executescript(stmt)
    return conn, cur


def test_get_and_set_settings():
    conn, cur = _prepare_db()
    row = app.get_settings(cur)
    assert row["id"] == 1

    boundary = "2024-01-01T00:00:00"
    app.set_last_run(boundary, cur)
    cur.execute("SELECT last_boundary FROM settings WHERE id=1")
    assert cur.fetchone()[0] == boundary
    conn.close()

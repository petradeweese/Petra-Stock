import sqlite3

import pandas as pd

import db
import routes


def test_forward_version_increment(monkeypatch, tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(id, ticker, direction, interval, rule, target_pct, stop_pct, window_value, window_unit) "
        "VALUES (1,'AAA','UP','15m','r1',1.0,0.5,1,'Hours')"
    )
    conn.commit()

    ts = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame({"Close": [100.0], "High": [100.0], "Low": [100.0]}, index=[ts])
    monkeypatch.setattr(routes, "get_prices", lambda *a, **k: {"AAA": df})
    monkeypatch.setattr(routes, "check_guardrails", lambda t: (True, []))

    fav = {
        "id": 1,
        "ticker": "AAA",
        "direction": "UP",
        "interval": "15m",
        "rule": "r1",
        "target_pct": 1.0,
        "stop_pct": 0.5,
        "window_value": 1,
        "window_unit": "Hours",
    }
    routes._create_forward_test(cur, fav)

    fav["target_pct"] = 2.0
    routes._create_forward_test(cur, fav)

    cur.execute("SELECT version, status FROM forward_tests ORDER BY id")
    rows = cur.fetchall()
    assert rows[0][0] == 1
    assert rows[0][1] == "closed"
    assert rows[1][0] == 2
    conn.close()

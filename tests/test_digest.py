import sqlite3

import pandas as pd

import db
from routes import compile_weekly_digest


def test_compile_weekly_digest(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(id, ticker, direction, interval, rule, target_pct, stop_pct, window_value, window_unit) "
        "VALUES (1,'AAA','UP','15m','r1',1.0,0.5,1,'Hours')"
    )
    conn.commit()

    now = pd.Timestamp("2024-01-12 21:00:00", tz="America/New_York")
    start_iso = pd.Timestamp("2024-01-08", tz="America/New_York").isoformat()
    cur.execute(
        "INSERT INTO forward_tests(fav_id,ticker,direction,interval,rule,version,entry_price,target_pct,stop_pct,window_minutes,status,roi_forward,hit_forward,dd_forward,created_at,updated_at) "
        "VALUES (1,'AAA','UP','15m','r1',1,100,1,0.5,60,'target',5,60,10,?,?)",
        (start_iso, start_iso),
    )
    cur.execute(
        "INSERT INTO guardrail_skips(ticker, reason, created_at) VALUES ('BBB','earnings',?)",
        (start_iso,),
    )
    conn.commit()

    subject, body = compile_weekly_digest(cur, ts=now)
    assert "[Forward Digest] Week of 2024-01-08" in subject
    assert "1 New" in subject
    assert "earnings: 1" in body
    conn.close()

import sqlite3

import pandas as pd

import db
from routes import _create_forward_test, check_guardrails


def test_guardrail_skip(monkeypatch, tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    # Insert a favorite required by foreign key
    cur.execute(
        "INSERT INTO favorites(id, ticker, direction, interval, "
        "rule, target_pct, stop_pct, window_value, window_unit) "
        "VALUES (1, 'AAA','UP','15m','r1',1.0,0.5,1,'Hours')"
    )
    conn.commit()

    ts = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame({"Close": [100.0], "High": [100.0], "Low": [100.0]}, index=[ts])
    monkeypatch.setattr("routes.get_prices", lambda *a, **k: {"AAA": df})
    monkeypatch.setattr("routes.check_guardrails", lambda ticker: (False, ["earnings"]))
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
    _create_forward_test(cur, fav)
    cur.execute("SELECT count(*) FROM forward_tests")
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT reason FROM guardrail_skips")
    assert cur.fetchone()[0] == "earnings"
    conn.close()


def test_check_guardrails(monkeypatch):
    today = pd.Timestamp("2024-01-01", tz="America/New_York")
    monkeypatch.setattr("routes.now_et", lambda: today)
    allowed, flags = check_guardrails(
        "AAA",
        get_earnings=lambda t: today + pd.Timedelta(days=3),
        get_adv=lambda t: 1_000_000,
    )
    assert not allowed and "earnings" in flags

    allowed, flags = check_guardrails(
        "AAA",
        get_earnings=lambda t: None,
        get_adv=lambda t: 100,
        adv_threshold=1_000,
    )
    assert not allowed and "low_liquidity" in flags

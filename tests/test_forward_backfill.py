import sqlite3

import pytest

import db
from scripts import backfill_forward_runs as backfill
from services.forward_runs import get_forward_history


def _setup(tmp_path):
    db.DB_PATH = str(tmp_path / "backfill.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def test_backfill_forward_runs_populates_history(tmp_path, monkeypatch):
    conn, cur = _setup(tmp_path)
    cur.execute(
        """
        INSERT INTO favorites(
            ticker, direction, interval, rule, target_pct, stop_pct,
            window_value, window_unit, lookback_years
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("AAPL", "UP", "15m", "rule-1", 1.0, 0.5, 4, "Hours", 1.0),
    )
    fav_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO forward_tests(
            fav_id, ticker, direction, interval, rule, version, entry_price,
            target_pct, stop_pct, window_minutes, status, roi_forward,
            dd_forward, time_to_hit, exit_reason, bars_to_exit, max_drawdown_pct,
            created_at, updated_at, last_run_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fav_id,
            "AAPL",
            "UP",
            "15m",
            "rule-1",
            1,
            188.0,
            1.2,
            0.6,
            60,
            "ok",
            5.0,
            1.0,
            90.0,
            "target",
            6,
            1.5,
            "2025-01-05T13:15:00",
            "2025-01-05T15:15:00",
            "2025-01-05T15:15:00",
        ),
    )
    conn.commit()

    events = []
    monkeypatch.setattr(backfill, "log_telemetry", events.append)

    stats = backfill.backfill_forward_runs(db.DB_PATH, dry_run=False)
    assert stats.inserted == 1
    assert stats.updated == 0

    cur.execute(
        "SELECT favorite_id, entry_ts, outcome, roi, tt_bars FROM forward_runs WHERE favorite_id=?",
        (str(fav_id),),
    )
    row = cur.fetchone()
    assert row is not None
    assert row["outcome"] == "hit"
    assert row["tt_bars"] == 6
    assert row["roi"] == pytest.approx(0.05)

    history = get_forward_history(str(fav_id), limit=5)
    assert len(history) == 1

    repeat_stats = backfill.backfill_forward_runs(db.DB_PATH, dry_run=False)
    assert repeat_stats.updated >= 1

    assert any(evt["event"] == "forward_runs_backfill_start" for evt in events)
    assert any(evt["event"] == "forward_runs_backfill_done" for evt in events)

    conn.close()

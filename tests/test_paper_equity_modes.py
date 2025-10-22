import time
from typing import Iterator

import pytest

import db
from db import get_db
from services import paper_trading


@pytest.fixture
def db_cursor(tmp_path) -> Iterator:
    db.DB_PATH = str(tmp_path / "paper_modes.db")
    db.init_db()
    gen = get_db()
    cursor = next(gen)
    try:
        yield cursor
    finally:
        try:
            cursor.connection.close()
        except Exception:
            pass
        try:
            gen.close()
        except Exception:
            pass


def _latest_equity(cursor, mode: str) -> float | None:
    row = cursor.execute(
        "SELECT equity FROM equity_log WHERE mode=? ORDER BY ts DESC LIMIT 1",
        (mode,),
    ).fetchone()
    return float(row[0]) if row else None


def test_summary_uses_mode_equity(db_cursor):
    paper_trading.ensure_settings(db_cursor)
    db_cursor.execute("DELETE FROM equity_log")
    db_cursor.connection.commit()

    paper_trading.upsert_settings(
        db_cursor,
        "hf",
        {"starting_balance": 50_000.0, "pct_equity_per_trade": 5.0},
    )
    paper_trading.upsert_settings(
        db_cursor,
        "lf",
        {"starting_balance": 25_000.0, "pct_equity_per_trade": 2.5},
    )

    summary = paper_trading.get_summary(db_cursor)
    assert summary["hf"]["account_equity"] == pytest.approx(50_000.0)
    assert summary["lf"]["account_equity"] == pytest.approx(25_000.0)

    assert paper_trading.load_settings(db_cursor, "hf").starting_balance == pytest.approx(
        50_000.0
    )
    assert paper_trading.load_settings(db_cursor, "lf").starting_balance == pytest.approx(
        25_000.0
    )

    now_ms = int(time.time() * 1000)
    db_cursor.execute(
        "INSERT INTO equity_log(mode, ts, equity) VALUES(?,?,?)",
        ("hf", now_ms, 60_000.0),
    )
    db_cursor.execute(
        "INSERT INTO equity_log(mode, ts, equity) VALUES(?,?,?)",
        ("lf", now_ms + 1, 20_000.0),
    )
    db_cursor.connection.commit()

    assert _latest_equity(db_cursor, "hf") == pytest.approx(60_000.0)
    assert _latest_equity(db_cursor, "lf") == pytest.approx(20_000.0)

    summary_after_updates = paper_trading.get_summary(db_cursor)
    assert summary_after_updates["hf"]["account_equity"] == pytest.approx(60_000.0)
    assert summary_after_updates["lf"]["account_equity"] == pytest.approx(20_000.0)


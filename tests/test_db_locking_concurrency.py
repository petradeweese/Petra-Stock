import datetime as dt
import sqlite3
import threading
from typing import List

import db
from services import data_provider, paper_trading


def _setup_db(path: str) -> None:
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                ts TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(symbol, interval, ts)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_settings (
                id INTEGER PRIMARY KEY CHECK (id=1),
                starting_balance REAL NOT NULL DEFAULT 10000,
                max_pct REAL NOT NULL DEFAULT 10,
                started_at TEXT,
                status TEXT NOT NULL DEFAULT 'inactive'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_persist_bars_while_settings_loads(tmp_path, monkeypatch):
    db_path = tmp_path / "locktest.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db, "_ENGINE", None, raising=False)
    _setup_db(str(db_path))

    base = dt.datetime.now(dt.timezone.utc)
    bars: List[dict] = []
    for idx in range(1500):
        bars.append(
            {
                "ts": base + dt.timedelta(minutes=idx),
                "open": 100.0 + idx,
                "high": 101.0 + idx,
                "low": 99.0 + idx,
                "close": 100.5 + idx,
                "volume": 1_000 + idx,
            }
        )

    errors: List[Exception] = []

    def writer() -> None:
        try:
            for _ in range(3):
                data_provider._persist_bars("ABC", "15m", bars)
        except Exception as exc:  # pragma: no cover - capture unexpected failures
            errors.append(exc)

    def reader() -> None:
        for _ in range(30):
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                cur = conn.cursor()
                paper_trading.ensure_settings(cur)
            except Exception as exc:  # pragma: no cover - capture unexpected failures
                errors.append(exc)
            finally:
                conn.close()

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    reader_thread.join()

    locked_errors = [
        exc for exc in errors if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()
    ]
    assert not locked_errors, f"Encountered locked errors: {locked_errors}"
    assert not errors, f"Unexpected errors encountered: {errors}"

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles

import db
import routes
from db import get_db
from services import paper_trading


@pytest.fixture
def db_cursor(tmp_path) -> Iterator:
    db.DB_PATH = str(tmp_path / "paper.db")
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


def _restart_active(cursor, *, when: datetime | None = None) -> None:
    paper_trading.restart_engine(cursor, now=when)
    paper_trading.start_engine(cursor, now=when)


def test_load_settings_creates_defaults_when_missing(db_cursor):
    db_cursor.execute("DROP TABLE IF EXISTS paper_settings")
    db_cursor.connection.commit()
    settings = paper_trading.load_settings(db_cursor)
    assert settings == paper_trading.PaperSettings(10000.0, 10.0, "inactive", None)


def test_settings_roundtrip_with_tuple_rows(tmp_path):
    original_path = db.DB_PATH
    try:
        db.DB_PATH = str(tmp_path / "paper_tuple.db")
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS paper_settings")
            conn.commit()
            first = paper_trading.load_settings(cursor)
            raw_row = cursor.execute(
                "SELECT starting_balance, max_pct FROM paper_settings WHERE id=1"
            ).fetchone()
            assert isinstance(raw_row, tuple)
            assert first.starting_balance == 10000.0
            assert first.max_pct == 10.0
            paper_trading.update_settings(cursor, starting_balance=15000, max_pct=12.5)
            second = paper_trading.load_settings(cursor)
            assert second.starting_balance == 15000.0
            assert second.max_pct == 12.5
        finally:
            conn.close()
    finally:
        db.DB_PATH = original_path


def test_calculate_max_contracts_respects_cap():
    qty, cap = paper_trading.calculate_max_contracts(10000, 25, 2.5)
    assert qty == 10
    assert cap == pytest.approx(2500)


def test_calculate_max_contracts_handles_rounding():
    qty, cap = paper_trading.calculate_max_contracts(15000, 12.5, 1.3333)
    assert qty * 1.3333 * 100 <= cap + 1e-6
    assert qty == 14  # floor(1875/133.33) while staying under cap


def test_calculate_roi_basic():
    assert paper_trading.calculate_roi(2.5, 3.0) == pytest.approx(20.0)


def test_resolve_fill_price_fallbacks():
    mid, src = paper_trading.resolve_fill_price(None, 3.5, None, None)
    assert mid == pytest.approx(3.5)
    assert src == "recent_mid"
    synth, src2 = paper_trading.resolve_fill_price(None, None, 2.0, 0.5)
    assert synth == pytest.approx(1.0)
    assert src2 == "synthetic_delta"


def test_open_position_dedupe(db_cursor):
    _restart_active(db_cursor, when=datetime(2024, 5, 1, tzinfo=timezone.utc))
    first = paper_trading.open_position(
        db_cursor,
        ticker="AAPL",
        call_put="CALL",
        strike=180,
        expiry="2024-09-20",
        interval="15m",
        mid_price=2.5,
        recent_mid=None,
        underlying_move=None,
        delta=0.4,
        source_alert_id="alert-1",
    )
    assert first is not None
    duplicate_alert = paper_trading.open_position(
        db_cursor,
        ticker="AAPL",
        call_put="CALL",
        strike=180,
        expiry="2024-09-20",
        interval="15m",
        mid_price=2.5,
        recent_mid=None,
        underlying_move=None,
        delta=0.4,
        source_alert_id="alert-1",
    )
    assert duplicate_alert is None
    duplicate_open = paper_trading.open_position(
        db_cursor,
        ticker="AAPL",
        call_put="CALL",
        strike=180,
        expiry="2024-09-20",
        interval="15m",
        mid_price=2.5,
        recent_mid=None,
        underlying_move=None,
        delta=0.4,
        source_alert_id="alert-2",
    )
    assert duplicate_open is None
    count = db_cursor.execute("SELECT COUNT(1) FROM paper_trades").fetchone()[0]
    assert count == 1


def test_open_position_skips_when_no_price(db_cursor, caplog):
    caplog.set_level(logging.INFO, logger="services.paper_trading")
    _restart_active(db_cursor)
    result = paper_trading.open_position(
        db_cursor,
        ticker="MSFT",
        call_put="CALL",
        strike=300,
        expiry="2024-11-15",
        interval="5m",
        mid_price=None,
        recent_mid=None,
        underlying_move=None,
        delta=None,
    )
    assert result is None
    assert any(
        getattr(record, "reason", "") == "no_price"
        for record in caplog.records
        if record.name == "services.paper_trading"
    )


def test_restart_engine_idempotency(db_cursor):
    paper_trading.update_settings(db_cursor, starting_balance=7500, max_pct=15)
    first = paper_trading.restart_engine(db_cursor, now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    assert first.status == "active"
    paper_trading.restart_engine(db_cursor, now=datetime(2024, 6, 2, tzinfo=timezone.utc))
    equity_points = db_cursor.execute("SELECT COUNT(1) FROM paper_equity").fetchone()[0]
    trades = db_cursor.execute("SELECT COUNT(1) FROM paper_trades").fetchone()[0]
    assert equity_points == 1
    assert trades == 0


def test_export_trades_csv_schema(db_cursor):
    _restart_active(db_cursor)
    trade_id = paper_trading.open_position(
        db_cursor,
        ticker="TSLA",
        call_put="PUT",
        strike=150,
        expiry="2024-09-20",
        interval="30m",
        mid_price=1.5,
        recent_mid=None,
        underlying_move=None,
        delta=0.45,
        source_alert_id="test-tsla",
    )
    assert trade_id is not None
    paper_trading.close_position(
        db_cursor,
        trade_id,
        exit_price=1.8,
        reason="target",
    )
    csv_text, filename = paper_trading.export_trades_csv(db_cursor, None)
    assert filename == "paper_trades.csv"
    rows = list(csv.reader(csv_text.strip().splitlines()))
    assert rows[0] == [
        "Ticker",
        "Call/Put",
        "Strike–Expiry",
        "Quantity",
        "Entry Time & Date",
        "Entry Price",
        "Exit Time & Date",
        "Exit Price",
        "ROI%",
        "Status",
    ]
    assert rows[1][0] == "TSLA"
    assert rows[1][-1] == "closed"


def test_paper_trade_lifecycle(db_cursor, caplog):
    caplog.set_level(logging.INFO, logger="services.paper_trading")
    paper_trading.update_settings(db_cursor, starting_balance=5000, max_pct=20)
    paper_trading.restart_engine(db_cursor, now=datetime(2024, 5, 1, tzinfo=timezone.utc))
    trade_id = paper_trading.open_position(
        db_cursor,
        ticker="TSLA",
        call_put="PUT",
        strike=150,
        expiry="2024-09-20",
        interval="30m",
        mid_price=None,
        recent_mid=1.2,
        underlying_move=None,
        delta=0.5,
        source_alert_id="tsla-hit",
    )
    assert trade_id is not None
    summary_after_entry = paper_trading.get_summary(db_cursor)
    assert summary_after_entry["balance"] < summary_after_entry["starting_balance"]
    trade = paper_trading.close_position(
        db_cursor,
        trade_id,
        exit_price=1.8,
        reason="target",
    )
    assert trade is not None
    assert trade.interval == "30m"
    assert trade.executed_at
    summary_after_exit = paper_trading.get_summary(db_cursor)
    assert summary_after_exit["balance"] > summary_after_entry["balance"]
    points = paper_trading.get_equity_points(db_cursor, "1d")
    assert len(points) >= 2
    trades = paper_trading.list_trades(db_cursor, "all")
    assert trades and trades[0]["status"] == "closed"
    assert any(record.message == "paper_trade_opened" for record in caplog.records)
    assert any(record.message == "paper_trade_closed" for record in caplog.records)


def test_paper_page_roi_chip_class(tmp_path):
    db.DB_PATH = str(tmp_path / "paper_ui.db")
    db.init_db()
    gen = get_db()
    cursor = next(gen)
    try:
        paper_trading.restart_engine(cursor)
        trade_id = paper_trading.open_position(
            cursor,
            ticker="NFLX",
            call_put="CALL",
            strike=400,
            expiry="2024-12-20",
            interval="1h",
            mid_price=2.0,
            recent_mid=None,
            underlying_move=None,
            delta=0.5,
            source_alert_id="nflx-hit",
        )
        assert trade_id is not None
        paper_trading.close_position(
            cursor,
            trade_id,
            exit_price=1.0,
            reason="stop",
        )
    finally:
        try:
            cursor.connection.close()
        except Exception:
            pass
        try:
            gen.close()
        except Exception:
            pass

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    res = client.get("/paper")
    assert res.status_code == 200
    assert "paper-roi-chip chip-neg" in res.text


def test_trades_api_rejects_invalid_status(db_cursor):
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    response = client.get("/paper/trades?status=bogus")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid status filter"


def test_equity_api_requires_valid_range(db_cursor):
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    response = client.get("/paper/equity?range=10y")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid range"

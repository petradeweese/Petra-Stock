from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import sqlite3

import pytest

import db
from db import get_db
from services.scalper import lf_engine


@pytest.fixture
def db_cursor(tmp_path) -> Iterator:
    db.DB_PATH = str(tmp_path / "scalper.db")
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


def _cursor_with_row_factory(row_factory) -> sqlite3.Cursor:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = row_factory
    conn.execute("PRAGMA foreign_keys=ON")
    return conn.cursor()


def _close_cursor(cursor: sqlite3.Cursor) -> None:
    try:
        cursor.connection.close()
    except Exception:
        pass


def test_load_settings_missing_row(monkeypatch):
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        lf_engine._ensure_schema(cursor)
        cursor.execute("DELETE FROM scalper_lf_settings")
        cursor.connection.commit()

        monkeypatch.setattr(lf_engine, "_ensure_schema", lambda db: None)
        settings = lf_engine.load_settings(cursor)
        assert isinstance(settings, lf_engine.LFSettings)
        assert settings.starting_balance == pytest.approx(100000.0)
    finally:
        _close_cursor(cursor)


def test_get_status_missing_row(monkeypatch):
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        lf_engine._ensure_schema(cursor)
        cursor.execute("DELETE FROM scalper_lf_state")
        cursor.connection.commit()

        monkeypatch.setattr(lf_engine, "_ensure_schema", lambda db: None)
        status = lf_engine.get_status(cursor)
        assert status.status == "inactive"
        assert status.started_at is None
    finally:
        _close_cursor(cursor)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_load_settings_tuple_rows():
    cursor = _cursor_with_row_factory(None)
    try:
        settings = lf_engine.load_settings(cursor)
        assert isinstance(settings, lf_engine.LFSettings)
        assert settings.starting_balance == pytest.approx(100000.0)
    finally:
        _close_cursor(cursor)


def test_load_settings_mapping_rows():
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        settings = lf_engine.load_settings(cursor)
        assert isinstance(settings, lf_engine.LFSettings)
        assert settings.tickers
    finally:
        _close_cursor(cursor)


def test_get_status_tuple_rows():
    cursor = _cursor_with_row_factory(None)
    try:
        status = lf_engine.get_status(cursor)
        assert isinstance(status, lf_engine.LFStatus)
        assert status.status in {"inactive", "active"}
    finally:
        _close_cursor(cursor)


def test_get_status_mapping_rows():
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        status = lf_engine.get_status(cursor)
        assert isinstance(status, lf_engine.LFStatus)
        assert status.status in {"inactive", "active"}
    finally:
        _close_cursor(cursor)


def test_position_sizing_formula():
    qty = lf_engine.calculate_position_size(balance=100000, pct_per_trade=3, mid_price=2.5)
    assert qty == 12  # floor((100000 * 0.03) / (2.5 * 100))


def test_fill_and_fees_model(db_cursor):
    lf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc))
    settings = lf_engine.load_settings(db_cursor)
    trade_id = lf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.0,
        entry_time=datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
        settings=settings,
    )
    assert trade_id is not None
    closed = lf_engine.close_trade(
        db_cursor,
        trade_id,
        mid_price=2.5,
        exit_time=datetime(2024, 1, 2, 15, 10, tzinfo=timezone.utc),
        settings=settings,
        reason="target",
    )
    assert closed is not None
    assert closed.entry_price == pytest.approx(2.01)
    assert closed.exit_price == pytest.approx(2.49)
    # qty computed based on 3% sizing default -> floor((100000*0.03)/(2*100)) == 15
    qty = lf_engine.calculate_position_size(100000, settings.pct_per_trade, 2.0)
    expected_fees = round((qty * settings.per_contract_fee + settings.per_order_fee) * 2, 2)
    assert closed.fees == pytest.approx(expected_fees)


def test_exit_conditions_evaluate_target_stop_time():
    entry = 2.0
    bars = [
        {"ts": "2024-01-02T14:31:00Z", "close": 2.02},
        {"ts": "2024-01-02T14:32:00Z", "close": 2.12},
        {"ts": "2024-01-02T14:33:00Z", "close": 1.92},
    ]
    price, reason, ts = lf_engine.evaluate_exit(
        entry_price=entry,
        bars=bars,
        target_pct=4.0,
        stop_pct=-3.0,
        time_cap_minutes=10,
    )
    assert reason == "target"
    assert price == pytest.approx(2.12)
    assert ts is not None

    timeout_price, timeout_reason, _ = lf_engine.evaluate_exit(
        entry_price=entry,
        bars=[{"ts": "2024-01-02T14:31:00Z", "close": 2.01}],
        target_pct=10.0,
        stop_pct=-10.0,
        time_cap_minutes=0,
    )
    assert timeout_reason == "timeout"
    assert timeout_price == pytest.approx(2.01)


def test_daily_trade_cap_and_dedupe(db_cursor, caplog):
    caplog.set_level("INFO")
    settings = lf_engine.update_settings(
        db_cursor,
        starting_balance=100000,
        pct_per_trade=3,
        daily_trade_cap=1,
        tickers=["SPY"],
        profit_target_pct=6,
        max_adverse_pct=-3,
        time_cap_minutes=15,
        session_start="09:30",
        session_end="16:00",
        allow_premarket=False,
        allow_postmarket=False,
        per_contract_fee=0.65,
        per_order_fee=0.0,
        rsi_filter=False,
    )
    lf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc))
    first_id = lf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.0,
        entry_time=datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
        settings=settings,
    )
    assert first_id is not None
    duplicate = lf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.0,
        entry_time=datetime(2024, 1, 2, 15, 5, tzinfo=timezone.utc),
        settings=settings,
    )
    assert duplicate is None
    cap_hit = lf_engine.open_trade(
        db_cursor,
        ticker="QQQ",
        option_type="CALL",
        strike=380,
        expiry="2024-01-19",
        mid_price=1.5,
        entry_time=datetime(2024, 1, 2, 15, 6, tzinfo=timezone.utc),
        settings=settings,
    )
    assert cap_hit is None


def test_status_lifecycle(db_cursor):
    stopped = lf_engine.stop_engine(db_cursor)
    assert stopped.status == "inactive"
    started = lf_engine.start_engine(db_cursor, now=datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert started.status == "active"
    restarted = lf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert restarted.started_at.endswith("03T00:00:00+00:00")


def test_lf_status_case_insensitive(db_cursor):
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    lf_engine.restart_engine(db_cursor, now=start)
    db_cursor.execute("UPDATE scalper_lf_state SET status='ACTIVE' WHERE id=1")
    db_cursor.connection.commit()

    status = lf_engine.get_status(db_cursor)
    assert status.status == "active"

    trade_id = lf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=None,
        expiry=None,
        mid_price=2.0,
        entry_time=start,
    )
    assert trade_id is not None


@pytest.mark.anyio
async def test_adapter_fallback(monkeypatch):
    calls: list[str] = []

    async def fail_quote(symbol: str):
        calls.append("schwab")
        raise RuntimeError("boom")

    async def success_quote(symbol: str):
        calls.append("yfinance")
        return {"symbol": symbol, "price": 1.23, "source": "yfinance"}

    monkeypatch.setattr(lf_engine, "_fetch_schwab_quote", fail_quote)
    monkeypatch.setattr(lf_engine, "_fetch_yfinance_quote", success_quote)

    quote = await lf_engine.fetch_quote("SPY")
    assert quote["source"] == "yfinance"
    assert calls == ["schwab", "yfinance"]


def test_backtest_smoke(db_cursor):
    bars = {
        "SPY": [
            {"ts": "2024-01-02T14:30:00Z", "open": 470.0, "high": 471.0, "low": 469.5, "close": 470.5, "vwap": 470.3},
            {"ts": "2024-01-02T14:31:00Z", "open": 470.5, "high": 471.5, "low": 470.2, "close": 471.2, "vwap": 470.7},
            {"ts": "2024-01-02T14:32:00Z", "open": 471.2, "high": 472.0, "low": 471.0, "close": 471.8, "vwap": 471.0},
            {"ts": "2024-01-02T14:33:00Z", "open": 471.8, "high": 472.4, "low": 471.5, "close": 472.2, "vwap": 471.4},
            {"ts": "2024-01-02T14:34:00Z", "open": 472.2, "high": 472.6, "low": 471.9, "close": 472.5, "vwap": 471.7},
            {"ts": "2024-01-02T14:35:00Z", "open": 472.5, "high": 473.0, "low": 472.2, "close": 472.9, "vwap": 472.0},
            {"ts": "2024-01-02T14:36:00Z", "open": 472.9, "high": 473.4, "low": 472.6, "close": 472.7, "vwap": 472.3},
        ]
    }
    results = lf_engine.run_backtest(db_cursor, bars_by_symbol=bars)
    assert "equity_curve" in results
    assert "summary" in results
    assert isinstance(results["equity_curve"], list)
    assert set(results["summary"].keys()) >= {"starting_balance", "ending_balance", "net_profit", "total_trades", "win_rate"}


def test_activity_csv_schema(db_cursor):
    lf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 2, tzinfo=timezone.utc))
    settings = lf_engine.load_settings(db_cursor)
    trade_id = lf_engine.open_trade(
        db_cursor,
        ticker="TSLA",
        option_type="PUT",
        strike=250,
        expiry="2024-01-19",
        mid_price=1.25,
        entry_time=datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
        settings=settings,
    )
    assert trade_id is not None
    lf_engine.close_trade(
        db_cursor,
        trade_id,
        mid_price=1.5,
        exit_time=datetime(2024, 1, 2, 15, 5, tzinfo=timezone.utc),
        settings=settings,
        reason="target",
    )
    csv_text, filename = lf_engine.export_activity_csv(db_cursor)
    assert filename == "scalper_lf_activity.csv"
    rows = [line.split(",") for line in csv_text.strip().splitlines()]
    assert rows[0] == [
        "Date",
        "Ticker",
        "Call/Put",
        "Strike-Expiry",
        "Qty",
        "Entry Time",
        "Entry Price",
        "Exit Time",
        "Exit Price",
        "ROI%",
        "Fees",
    ]
    assert any("Fees" in header for header in rows[0])

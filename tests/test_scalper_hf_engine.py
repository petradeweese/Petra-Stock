from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import sqlite3

import pytest

import db
from db import get_db
from services.scalper import hf_engine


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


def test_hf_load_settings_tuple_rows():
    cursor = _cursor_with_row_factory(None)
    try:
        settings = hf_engine.load_settings(cursor)
        assert isinstance(settings, hf_engine.HFSettings)
        assert settings.tickers
    finally:
        _close_cursor(cursor)


def test_hf_get_status_tuple_rows():
    cursor = _cursor_with_row_factory(None)
    try:
        status = hf_engine.get_status(cursor)
        assert isinstance(status, hf_engine.HFStatus)
        assert status.status in {"inactive", "active", "halted"}
    finally:
        _close_cursor(cursor)


def test_hf_load_settings_missing_row(monkeypatch):
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        hf_engine._ensure_schema(cursor)
        cursor.execute("DELETE FROM scalper_hf_settings")
        cursor.connection.commit()

        monkeypatch.setattr(hf_engine, "_ensure_schema", lambda db: None)
        settings = hf_engine.load_settings(cursor)
        assert settings.starting_balance == pytest.approx(100000.0)
        assert settings.tickers
    finally:
        _close_cursor(cursor)


def test_hf_get_status_missing_row(monkeypatch):
    cursor = _cursor_with_row_factory(sqlite3.Row)
    try:
        hf_engine._ensure_schema(cursor)
        cursor.execute("DELETE FROM scalper_hf_state")
        cursor.connection.commit()

        monkeypatch.setattr(hf_engine, "_ensure_schema", lambda db: None)
        status = hf_engine.get_status(cursor)
        assert status.status == "inactive"
        assert status.started_at is None
        assert status.account_equity == pytest.approx(100000.0)
    finally:
        _close_cursor(cursor)


def test_hf_update_settings_resets_equity_seed_when_pristine(db_cursor):
    hf_engine.get_status(db_cursor)
    settings = hf_engine.load_settings(db_cursor)
    row = db_cursor.execute(
        "SELECT balance FROM scalper_hf_equity ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(settings.starting_balance)

    updated = hf_engine.update_settings(
        db_cursor,
        starting_balance=50000,
        pct_per_trade=settings.pct_per_trade,
        daily_trade_cap=settings.daily_trade_cap,
        tickers=settings.tickers,
        profit_target_pct=settings.profit_target_pct,
        max_adverse_pct=settings.max_adverse_pct,
        time_cap_minutes=settings.time_cap_minutes,
        cooldown_minutes=settings.cooldown_minutes,
        max_open_positions=settings.max_open_positions,
        daily_max_drawdown_pct=settings.daily_max_drawdown_pct,
        per_contract_fee=settings.per_contract_fee,
        per_order_fee=settings.per_order_fee,
        volatility_gate=settings.volatility_gate,
    )

    assert updated.starting_balance == pytest.approx(50000)
    row_after = db_cursor.execute(
        "SELECT balance FROM scalper_hf_equity ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    assert row_after is not None
    assert row_after[0] == pytest.approx(50000)
    status_after = hf_engine.get_status(db_cursor)
    assert status_after.account_equity == pytest.approx(50000)


@pytest.fixture
def db_cursor(tmp_path) -> Iterator:
    db.DB_PATH = str(tmp_path / "scalper_hf.db")
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


def test_hf_signal_controls(db_cursor):
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    hf_engine.restart_engine(db_cursor, now=start)
    settings = hf_engine.update_settings(
        db_cursor,
        starting_balance=50000,
        pct_per_trade=2,
        daily_trade_cap=5,
        tickers=["SPY", "QQQ"],
        profit_target_pct=4,
        max_adverse_pct=-2,
        time_cap_minutes=5,
        cooldown_minutes=2,
        max_open_positions=2,
        daily_max_drawdown_pct=-3,
        per_contract_fee=0.65,
        per_order_fee=0.0,
        volatility_gate=3.0,
    )
    # Insufficient momentum should gate the trade.
    blocked = hf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.5,
        entry_time=start,
        momentum_score=0.1,
        vwap=2.5,
        ema9=2.5,
        volatility=1.0,
        liquidity=1.0,
        settings=settings,
    )
    assert blocked is None

    entry_one = start + timedelta(minutes=1)
    trade_one = hf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.4,
        entry_time=entry_one,
        momentum_score=1.1,
        vwap=2.4,
        ema9=2.35,
        volatility=1.0,
        liquidity=0.9,
        settings=settings,
    )
    assert trade_one is not None

    entry_two = entry_one + timedelta(minutes=1)
    trade_two = hf_engine.open_trade(
        db_cursor,
        ticker="QQQ",
        option_type="CALL",
        strike=380,
        expiry="2024-01-19",
        mid_price=1.8,
        entry_time=entry_two,
        momentum_score=1.2,
        vwap=1.81,
        ema9=1.79,
        volatility=1.1,
        liquidity=0.8,
        settings=settings,
    )
    assert trade_two is not None

    # Third trade should be blocked by max open positions.
    trade_three = hf_engine.open_trade(
        db_cursor,
        ticker="TSLA",
        option_type="PUT",
        strike=250,
        expiry="2024-01-19",
        mid_price=3.0,
        entry_time=entry_two,
        momentum_score=1.3,
        vwap=3.0,
        ema9=2.95,
        volatility=1.0,
        liquidity=0.7,
        settings=settings,
    )
    assert trade_three is None

    exit_one = entry_one + timedelta(minutes=2)
    closed_one = hf_engine.close_trade(
        db_cursor,
        trade_one,
        mid_price=2.6,
        exit_time=exit_one,
        liquidity=0.8,
        settings=settings,
    )
    assert closed_one is not None

    # Cool-down should block immediate re-entry.
    cooldown_block = hf_engine.open_trade(
        db_cursor,
        ticker="MSFT",
        option_type="CALL",
        strike=330,
        expiry="2024-01-19",
        mid_price=1.5,
        entry_time=exit_one,
        momentum_score=1.1,
        vwap=1.5,
        ema9=1.48,
        volatility=1.0,
        liquidity=0.9,
        settings=settings,
    )
    assert cooldown_block is None

    # Force a drawdown breach and ensure engine halts.
    aggressive_settings = hf_engine.update_settings(
        db_cursor,
        starting_balance=settings.starting_balance,
        pct_per_trade=5,
        daily_trade_cap=settings.daily_trade_cap,
        tickers=settings.tickers.split(","),
        profit_target_pct=settings.profit_target_pct,
        max_adverse_pct=settings.max_adverse_pct,
        time_cap_minutes=settings.time_cap_minutes,
        cooldown_minutes=settings.cooldown_minutes,
        max_open_positions=settings.max_open_positions,
        daily_max_drawdown_pct=-0.5,
        per_contract_fee=settings.per_contract_fee,
        per_order_fee=settings.per_order_fee,
        volatility_gate=settings.volatility_gate,
    )
    exit_two = entry_two + timedelta(minutes=3)
    hf_engine.close_trade(
        db_cursor,
        trade_two,
        mid_price=1.2,
        exit_time=exit_two,
        liquidity=0.4,
        settings=aggressive_settings,
    )
    breach_attempt = hf_engine.open_trade(
        db_cursor,
        ticker="AMD",
        option_type="CALL",
        strike=180,
        expiry="2024-01-19",
        mid_price=1.2,
        entry_time=exit_two + timedelta(minutes=3),
        momentum_score=1.5,
        vwap=1.21,
        ema9=1.19,
        volatility=1.0,
        liquidity=0.6,
        settings=aggressive_settings,
    )
    assert breach_attempt is None
    halted_status = hf_engine.status_payload(db_cursor)
    assert halted_status.get("status") == "halted"
    assert halted_status.get("halted") is True


def test_hf_slippage_widening(db_cursor):
    hf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc))
    settings = hf_engine.update_settings(
        db_cursor,
        starting_balance=100000,
        pct_per_trade=1,
        daily_trade_cap=10,
        tickers=["SPY"],
        profit_target_pct=4,
        max_adverse_pct=-2,
        time_cap_minutes=5,
        cooldown_minutes=0,
        max_open_positions=3,
        daily_max_drawdown_pct=-10,
        per_contract_fee=0.0,
        per_order_fee=0.0,
        volatility_gate=5.0,
    )
    entry_time = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)
    trade_liquid = hf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=1.50,
        entry_time=entry_time,
        momentum_score=1.2,
        vwap=1.50,
        ema9=1.49,
        volatility=1.0,
        liquidity=1.0,
        settings=settings,
    )
    assert trade_liquid is not None
    low_liq_time = entry_time + timedelta(minutes=1)
    trade_thin = hf_engine.open_trade(
        db_cursor,
        ticker="QQQ",
        option_type="CALL",
        strike=380,
        expiry="2024-01-19",
        mid_price=1.50,
        entry_time=low_liq_time,
        momentum_score=1.3,
        vwap=1.50,
        ema9=1.49,
        volatility=1.0,
        liquidity=0.1,
        settings=settings,
    )
    assert trade_thin is not None
    row_liquid = db_cursor.execute(
        "SELECT entry_price FROM scalper_hf_activity WHERE id=?",
        (trade_liquid,),
    ).fetchone()
    row_thin = db_cursor.execute(
        "SELECT entry_price FROM scalper_hf_activity WHERE id=?",
        (trade_thin,),
    ).fetchone()
    assert row_liquid and row_thin
    assert float(row_thin["entry_price"]) > float(row_liquid["entry_price"])


def test_hf_metrics_snapshot(db_cursor):
    hf_engine.restart_engine(db_cursor, now=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc))
    settings = hf_engine.update_settings(
        db_cursor,
        starting_balance=100000,
        pct_per_trade=1,
        daily_trade_cap=10,
        tickers=["SPY", "QQQ"],
        profit_target_pct=4,
        max_adverse_pct=-2,
        time_cap_minutes=5,
        cooldown_minutes=0,
        max_open_positions=3,
        daily_max_drawdown_pct=-10,
        per_contract_fee=0.0,
        per_order_fee=0.0,
        volatility_gate=5.0,
    )
    entry = datetime(2024, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = hf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=450,
        expiry="2024-01-19",
        mid_price=2.0,
        entry_time=entry,
        momentum_score=1.0,
        vwap=2.0,
        ema9=1.99,
        volatility=1.0,
        liquidity=0.8,
        settings=settings,
    )
    hf_engine.close_trade(
        db_cursor,
        t1,
        mid_price=2.4,
        exit_time=entry + timedelta(minutes=2),
        liquidity=0.8,
        settings=settings,
    )
    t2 = hf_engine.open_trade(
        db_cursor,
        ticker="QQQ",
        option_type="CALL",
        strike=380,
        expiry="2024-01-19",
        mid_price=1.5,
        entry_time=entry + timedelta(minutes=3),
        momentum_score=1.1,
        vwap=1.5,
        ema9=1.48,
        volatility=1.0,
        liquidity=0.9,
        settings=settings,
    )
    hf_engine.close_trade(
        db_cursor,
        t2,
        mid_price=1.2,
        exit_time=entry + timedelta(minutes=5),
        liquidity=0.5,
        settings=settings,
    )
    metrics = hf_engine.metrics_snapshot(db_cursor)
    assert metrics["win_rate"] == pytest.approx(50.0)
    assert metrics["avg_win"] > 0
    assert metrics["avg_loss"] < 0
    assert metrics["profit_factor"] > 0
    assert metrics["trades_per_day"] >= 1.0


def test_hf_backtest_smoke(db_cursor):
    bars = {
        "SPY": [
            {"ts": "2024-01-02T14:30:00Z", "open": 470.0, "high": 471.0, "low": 469.5, "close": 470.5, "vwap": 470.3, "volume": 120000},
            {"ts": "2024-01-02T14:31:00Z", "open": 470.5, "high": 471.5, "low": 470.2, "close": 471.2, "vwap": 470.7, "volume": 98000},
            {"ts": "2024-01-02T14:32:00Z", "open": 471.2, "high": 472.0, "low": 471.0, "close": 471.8, "vwap": 471.0, "volume": 86000},
            {"ts": "2024-01-02T14:33:00Z", "open": 471.8, "high": 472.4, "low": 471.5, "close": 472.2, "vwap": 471.4, "volume": 79000},
            {"ts": "2024-01-02T14:34:00Z", "open": 472.2, "high": 472.6, "low": 471.9, "close": 472.5, "vwap": 471.7, "volume": 72000},
        ]
    }
    results = hf_engine.run_backtest(db_cursor, bars_by_symbol=bars)
    assert "equity_curve" in results
    assert "summary" in results
    assert isinstance(results["equity_curve"], list)
    assert set(results["summary"].keys()) >= {"starting_balance", "ending_balance", "net_profit", "total_trades", "win_rate"}


def test_hf_current_equity_updates_after_close(db_cursor):
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    hf_engine.update_settings(
        db_cursor,
        starting_balance=50000,
        pct_per_trade=1.5,
        daily_trade_cap=10,
        tickers=["SPY"],
        profit_target_pct=4,
        max_adverse_pct=-3,
        time_cap_minutes=5,
        cooldown_minutes=0,
        max_open_positions=3,
        daily_max_drawdown_pct=-10,
        per_contract_fee=0.0,
        per_order_fee=0.0,
        volatility_gate=5.0,
    )
    hf_engine.restart_engine(db_cursor, now=start)
    settings = hf_engine.load_settings(db_cursor)
    initial_equity = hf_engine.current_equity(db_cursor)
    assert initial_equity == pytest.approx(settings.starting_balance)

    entry_time = start + timedelta(minutes=1)
    trade_id = hf_engine.open_trade(
        db_cursor,
        ticker="SPY",
        option_type="CALL",
        strike=None,
        expiry=None,
        mid_price=1.0,
        entry_time=entry_time,
        momentum_score=1.0,
        vwap=1.0,
        ema9=1.0,
        volatility=0.5,
        liquidity=1.0,
        settings=settings,
    )
    assert trade_id is not None

    exit_time = entry_time + timedelta(minutes=1)
    closed = hf_engine.close_trade(
        db_cursor,
        trade_id,
        mid_price=1.2,
        exit_time=exit_time,
        reason="target",
        liquidity=1.0,
        settings=settings,
    )
    assert closed is not None

    updated_equity = hf_engine.current_equity(db_cursor)
    assert updated_equity > initial_equity

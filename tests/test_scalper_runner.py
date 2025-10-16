"""Regression and integration tests for the scalper runner."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Dict, Iterator, List

import pytest

import db
import services.scalper.runner as runner
from services.scalper import hf_engine, lf_engine
from services.scalper.runner import _determine_lookback, _run_hf_iteration, _run_lf_iteration


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _bar(
    ts: datetime,
    *,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    volume: float = 10_000,
    vwap: float | None = None,
    liquidity: float = 1.0,
    volatility: float | None = None,
) -> Dict[str, float | str]:
    high_val = high_price if high_price is not None else max(open_price, close_price)
    low_val = low_price if low_price is not None else min(open_price, close_price)
    vwap_val = vwap if vwap is not None else (high_val + low_val + close_price) / 3
    vol_val = volatility if volatility is not None else abs(high_val - low_val)
    return {
        "ts": ts.isoformat(),
        "open": open_price,
        "high": high_val,
        "low": low_val,
        "close": close_price,
        "volume": volume,
        "vwap": vwap_val,
        "volatility": vol_val,
        "liquidity": liquidity,
    }


def test_determine_lookback_default():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    lookback = _determine_lookback(45, now=now, open_rows=[], max_minutes=300)
    assert lookback == 45


def test_determine_lookback_extends_for_open_trade():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    entry = now - timedelta(minutes=120)
    row = {"entry_time": entry.isoformat()}
    lookback = _determine_lookback(45, now=now, open_rows=[row], max_minutes=300)
    assert lookback >= 125
    assert lookback <= 300


def test_determine_lookback_ignores_invalid_entries():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    rows = [
        {"entry_time": None},
        {"entry_time": "invalid"},
    ]
    lookback = _determine_lookback(60, now=now, open_rows=rows, max_minutes=120)
    assert lookback == 60


@contextmanager
def _cursor(path: str) -> Iterator[sqlite3.Cursor]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        try:
            cursor.connection.commit()
        except Exception:
            pass
        conn.close()


@pytest.mark.anyio
async def test_hf_iteration_closes_and_opens(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "hf_runner.db")
    db.init_db()
    monkeypatch.setattr("services.scalper.runner.DB_PATH", db.DB_PATH, raising=False)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    with _cursor(db.DB_PATH) as cursor:
        hf_engine.restart_engine(cursor, now=start)
        settings = hf_engine.update_settings(
            cursor,
            starting_balance=50_000,
            pct_per_trade=2.0,
            daily_trade_cap=10,
            tickers=["SPY", "QQQ"],
            profit_target_pct=4.0,
            max_adverse_pct=-2.0,
            time_cap_minutes=5,
            cooldown_minutes=0,
            max_open_positions=3,
            daily_max_drawdown_pct=-10.0,
            per_contract_fee=0.65,
            per_order_fee=0.0,
            volatility_gate=5.0,
        )
        entry_time = start + timedelta(minutes=1)
        trade_id = hf_engine.open_trade(
            cursor,
            ticker="SPY",
            option_type="CALL",
            strike=None,
            expiry=None,
            mid_price=2.0,
            entry_time=entry_time,
            momentum_score=1.0,
            vwap=2.0,
            ema9=2.0,
            volatility=1.0,
            liquidity=0.9,
            settings=settings,
        )
        assert trade_id is not None

    iteration_now = start + timedelta(minutes=6)
    spy_bars: List[Dict[str, float | str]] = []
    for idx, close in enumerate([2.01, 2.05, 2.09, 2.11]):
        ts = entry_time + timedelta(minutes=idx)
        spy_bars.append(
            _bar(
                ts,
                open_price=close - 0.02,
                close_price=close,
                high_price=close + 0.03,
                low_price=close - 0.03,
                liquidity=0.9,
            )
        )
    qqq_prev = iteration_now - timedelta(minutes=1)
    qqq_now = iteration_now
    qqq_bars = [
        _bar(
            qqq_prev,
            open_price=1.95,
            close_price=2.0,
            high_price=2.05,
            low_price=1.9,
            liquidity=0.9,
        ),
        _bar(
            qqq_now,
            open_price=2.0,
            close_price=2.6,
            high_price=2.62,
            low_price=1.98,
            vwap=2.6,
            liquidity=1.0,
        ),
    ]

    bars_by_symbol = {"SPY": spy_bars, "QQQ": qqq_bars}

    async def fake_fetch(symbols, *, lookback_minutes, now):  # type: ignore[override]
        providers = {sym: "mock" for sym in symbols}
        return {sym: bars_by_symbol.get(sym, []) for sym in symbols}, providers

    orig_close = hf_engine.close_trade
    close_calls: List[int] = []
    eval_calls: List[dict] = []
    status_calls: List[hf_engine.HFStatus] = []

    def capture_close(db, trade_id, **kwargs):
        close_calls.append(trade_id)
        return orig_close(db, trade_id, **kwargs)

    orig_eval = hf_engine.evaluate_exit

    def capture_eval(**kwargs):
        eval_calls.append(kwargs)
        return orig_eval(**kwargs)

    orig_status = hf_engine.get_status

    def capture_status(db, **kwargs):
        st = orig_status(db, **kwargs)
        status_calls.append(st)
        return st

    monkeypatch.setattr("services.scalper.runner._fetch_recent_bars", fake_fetch)
    monkeypatch.setattr(hf_engine, "close_trade", capture_close)
    monkeypatch.setattr(hf_engine, "evaluate_exit", capture_eval)
    monkeypatch.setattr(hf_engine, "get_status", capture_status)
    await _run_hf_iteration(iteration_now, startup_logged=True, iteration=1)

    assert status_calls and all(st.status == "active" for st in status_calls)
    assert eval_calls, "expected evaluate_exit to be invoked"
    assert close_calls, "expected close_trade to be invoked"
    with _cursor(db.DB_PATH) as cursor:
        open_rows = cursor.execute(
            "SELECT ticker FROM scalper_hf_activity WHERE status='open' ORDER BY id"
        ).fetchall()
        assert [row["ticker"] for row in open_rows] == ["QQQ"]
        closed_rows = cursor.execute(
            "SELECT ticker, net_pl FROM scalper_hf_activity WHERE status='closed'"
        ).fetchall()
        assert any(row["ticker"] == "SPY" and float(row["net_pl"]) > 0 for row in closed_rows)
        status = hf_engine.status_payload(cursor)
        assert status["status"] == "active"
        assert status["open_positions"] == 1
        assert status["account_equity"] > 50_000
        equity = cursor.execute(
            "SELECT balance FROM scalper_hf_equity ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        assert equity is not None
        assert equity["balance"] == pytest.approx(status["account_equity"])


@pytest.mark.anyio
async def test_hf_iteration_log_reports_final_open_positions(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "hf_runner_log.db")
    db.init_db()
    monkeypatch.setattr("services.scalper.runner.DB_PATH", db.DB_PATH, raising=False)
    start = datetime(2024, 1, 4, 14, 0, tzinfo=timezone.utc)
    with _cursor(db.DB_PATH) as cursor:
        hf_engine.restart_engine(cursor, now=start)
        settings = hf_engine.update_settings(
            cursor,
            starting_balance=60_000,
            pct_per_trade=2.0,
            daily_trade_cap=5,
            tickers=["SPY"],
            profit_target_pct=4.0,
            max_adverse_pct=-2.0,
            time_cap_minutes=5,
            cooldown_minutes=0,
            max_open_positions=3,
            daily_max_drawdown_pct=-10.0,
            per_contract_fee=0.65,
            per_order_fee=0.0,
            volatility_gate=5.0,
        )
        entry_time = start + timedelta(minutes=1)
        trade_id = hf_engine.open_trade(
            cursor,
            ticker="SPY",
            option_type="CALL",
            strike=None,
            expiry=None,
            mid_price=2.0,
            entry_time=entry_time,
            momentum_score=1.0,
            vwap=2.0,
            ema9=2.0,
            volatility=1.0,
            liquidity=0.9,
            settings=settings,
        )
        assert trade_id is not None

    iteration_now = start + timedelta(minutes=10)
    spy_bars = [
        _bar(
            entry_time,
            open_price=2.0,
            close_price=2.0,
            high_price=2.03,
            low_price=1.97,
            liquidity=0.9,
        ),
        _bar(
            entry_time + timedelta(minutes=1),
            open_price=2.02,
            close_price=2.12,
            high_price=2.15,
            low_price=2.0,
            liquidity=0.9,
        ),
    ]

    async def fake_fetch(symbols, *, lookback_minutes, now):  # type: ignore[override]
        providers = {sym: "mock" for sym in symbols}
        return {sym: spy_bars if sym == "SPY" else [] for sym in symbols}, providers

    info_logs: List[str] = []

    def capture_info(msg, *args, **kwargs):
        info_logs.append(msg % args if args else str(msg))

    monkeypatch.setattr("services.scalper.runner._fetch_recent_bars", fake_fetch)

    def refuse_new_trades(db_conn, **kwargs):
        return None

    monkeypatch.setattr(hf_engine, "open_trade", refuse_new_trades)
    monkeypatch.setattr(runner.logger, "info", capture_info)

    await _run_hf_iteration(iteration_now, startup_logged=True, iteration=2)

    with _cursor(db.DB_PATH) as cursor:
        count_row = cursor.execute(
            "SELECT COUNT(1) FROM scalper_hf_activity WHERE status='open'",
        ).fetchone()
        assert count_row is not None and int(count_row[0]) == 0

    iteration_logs = [msg for msg in info_logs if msg.startswith("hf_loop_iteration")]
    assert iteration_logs, "expected hf_loop_iteration log entry"
    assert "open_positions=0" in iteration_logs[-1]


@pytest.mark.anyio
async def test_lf_iteration_closes_and_opens(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "lf_runner.db")
    db.init_db()
    monkeypatch.setattr("services.scalper.runner.DB_PATH", db.DB_PATH, raising=False)
    start = datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)
    with _cursor(db.DB_PATH) as cursor:
        lf_engine.restart_engine(cursor, now=start)
        settings = lf_engine.update_settings(
            cursor,
            starting_balance=80_000,
            pct_per_trade=3.0,
            daily_trade_cap=10,
            tickers=["SPY", "QQQ"],
            profit_target_pct=6.0,
            max_adverse_pct=-3.0,
            time_cap_minutes=15,
            session_start="09:30",
            session_end="16:00",
            allow_premarket=False,
            allow_postmarket=False,
            per_contract_fee=0.65,
            per_order_fee=0.0,
            rsi_filter=False,
        )
        entry_time = start + timedelta(minutes=1)
        trade_id = lf_engine.open_trade(
            cursor,
            ticker="SPY",
            option_type="CALL",
            strike=None,
            expiry=None,
            mid_price=2.0,
            entry_time=entry_time,
            settings=settings,
        )
        assert trade_id is not None

    iteration_now = start + timedelta(minutes=8)
    spy_bars: List[Dict[str, float | str]] = []
    for idx, close in enumerate([2.01, 2.05, 2.08, 2.12, 2.15, 2.2]):
        ts = entry_time + timedelta(minutes=idx)
        spy_bars.append(
            _bar(
                ts,
                open_price=close - 0.02,
                close_price=close,
                high_price=close + 0.03,
                low_price=close - 0.03,
                liquidity=0.9,
            )
        )
    qqq_bars: List[Dict[str, float | str]] = []
    for idx, close in enumerate([1.8, 1.85, 1.88, 1.9, 1.95, 2.2]):
        ts = iteration_now - timedelta(minutes=5 - idx)
        qqq_bars.append(
            _bar(
                ts,
                open_price=close - 0.02,
                close_price=close,
                high_price=close + 0.03,
                low_price=close - 0.03,
                vwap=close - 0.01,
                liquidity=0.95,
            )
        )

    bars_by_symbol = {"SPY": spy_bars, "QQQ": qqq_bars}

    async def fake_fetch(symbols, *, lookback_minutes, now):  # type: ignore[override]
        providers = {sym: "mock" for sym in symbols}
        return {sym: bars_by_symbol.get(sym, []) for sym in symbols}, providers

    orig_close = lf_engine.close_trade
    close_calls: List[int] = []
    eval_calls: List[dict] = []
    status_calls: List[lf_engine.LFStatus] = []

    def capture_close(db, trade_id, **kwargs):
        close_calls.append(trade_id)
        return orig_close(db, trade_id, **kwargs)

    orig_eval = lf_engine.evaluate_exit

    def capture_eval(**kwargs):
        eval_calls.append(kwargs)
        return orig_eval(**kwargs)

    orig_status = lf_engine.get_status

    def capture_status(db, **kwargs):
        st = orig_status(db, **kwargs)
        status_calls.append(st)
        return st

    monkeypatch.setattr("services.scalper.runner._fetch_recent_bars", fake_fetch)
    monkeypatch.setattr(lf_engine, "close_trade", capture_close)
    monkeypatch.setattr(lf_engine, "evaluate_exit", capture_eval)
    monkeypatch.setattr(lf_engine, "get_status", capture_status)
    await _run_lf_iteration(iteration_now, startup_logged=True, iteration=1)

    assert status_calls and all(st.status == "active" for st in status_calls)
    assert eval_calls, "expected evaluate_exit to be invoked"
    assert close_calls, "expected close_trade to be invoked"
    with _cursor(db.DB_PATH) as cursor:
        open_rows = cursor.execute(
            "SELECT ticker FROM scalper_lf_activity WHERE status='open' ORDER BY id"
        ).fetchall()
        open_symbols = [row["ticker"] for row in open_rows]
        assert "QQQ" in open_symbols
        closed_rows = cursor.execute(
            "SELECT ticker, net_pl FROM scalper_lf_activity WHERE status='closed'"
        ).fetchall()
        assert any(row["ticker"] == "SPY" and float(row["net_pl"]) > 0 for row in closed_rows)
        status = lf_engine.status_payload(cursor)
        assert status["status"] == "active"
        assert status["open_positions"] >= 1
        assert status["account_equity"] > 80_000
        metrics = lf_engine.metrics_snapshot(cursor)
        assert metrics["win_rate"] >= 0.0


@pytest.mark.anyio
async def test_lf_iteration_log_reports_final_open_positions(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "lf_runner_log.db")
    db.init_db()
    monkeypatch.setattr("services.scalper.runner.DB_PATH", db.DB_PATH, raising=False)
    start = datetime(2024, 1, 4, 15, 0, tzinfo=timezone.utc)
    with _cursor(db.DB_PATH) as cursor:
        lf_engine.restart_engine(cursor, now=start)
        settings = lf_engine.update_settings(
            cursor,
            starting_balance=70_000,
            pct_per_trade=3.0,
            daily_trade_cap=5,
            tickers=["SPY"],
            profit_target_pct=4.0,
            max_adverse_pct=-3.0,
            time_cap_minutes=15,
            session_start="09:30",
            session_end="16:00",
            allow_premarket=False,
            allow_postmarket=False,
            per_contract_fee=0.65,
            per_order_fee=0.0,
            rsi_filter=False,
        )
        entry_time = start + timedelta(minutes=1)
        trade_id = lf_engine.open_trade(
            cursor,
            ticker="SPY",
            option_type="CALL",
            strike=None,
            expiry=None,
            mid_price=2.0,
            entry_time=entry_time,
            settings=settings,
        )
        assert trade_id is not None

    iteration_now = entry_time + timedelta(minutes=6)
    closes = [2.0, 2.04, 2.08, 2.12, 2.15, 2.18]
    spy_bars = [
        _bar(
            entry_time + timedelta(minutes=idx),
            open_price=closes[idx] - 0.02,
            close_price=closes[idx],
            high_price=closes[idx] + 0.03,
            low_price=closes[idx] - 0.03,
            liquidity=0.9,
        )
        for idx in range(len(closes))
    ]

    async def fake_fetch(symbols, *, lookback_minutes, now):  # type: ignore[override]
        providers = {sym: "mock" for sym in symbols}
        return {sym: spy_bars if sym == "SPY" else [] for sym in symbols}, providers

    info_logs: List[str] = []

    def capture_info(msg, *args, **kwargs):
        info_logs.append(msg % args if args else str(msg))

    monkeypatch.setattr("services.scalper.runner._fetch_recent_bars", fake_fetch)

    def refuse_new_trades(db_conn, **kwargs):
        return None

    monkeypatch.setattr(lf_engine, "open_trade", refuse_new_trades)
    monkeypatch.setattr(runner.logger, "info", capture_info)

    await _run_lf_iteration(iteration_now, startup_logged=True, iteration=2)

    with _cursor(db.DB_PATH) as cursor:
        count_row = cursor.execute(
            "SELECT COUNT(1) FROM scalper_lf_activity WHERE status='open'",
        ).fetchone()
        assert count_row is not None and int(count_row[0]) == 0

    iteration_logs = [msg for msg in info_logs if msg.startswith("lf_loop_iteration")]
    assert iteration_logs, "expected lf_loop_iteration log entry"
    assert "open_positions=0" in iteration_logs[-1]

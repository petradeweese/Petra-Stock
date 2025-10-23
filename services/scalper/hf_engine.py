"""High-frequency paper scalper engine."""
from __future__ import annotations

import csv
import io
import logging
import sqlite3

import db as db_module
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services import paper_trading

from utils import TZ, now_et

from .lf_engine import evaluate_exit
from .shared import (
    FeeModel,
    apply_slippage,
    calculate_position_size as _shared_position_size,
    compute_trade_metrics,
    summarize_backtest,
)

logger = logging.getLogger(__name__)

_DEFAULT_TICKERS = "SPY,QQQ,NVDA,TSLA,META,AMD"


def _ensure_row_factory(db) -> None:
    conn = getattr(db, "connection", None)
    if conn is None:
        return
    try:
        if getattr(conn, "row_factory", None) is not sqlite3.Row:
            conn.row_factory = sqlite3.Row
    except Exception:
        # Best-effort: fallback to whatever the connection supports.
        pass


@dataclass(slots=True)
class HFSettings:
    starting_balance: float
    pct_per_trade: float
    daily_trade_cap: int
    tickers: str
    profit_target_pct: float
    max_adverse_pct: float
    time_cap_minutes: int
    cooldown_minutes: int
    max_open_positions: int
    daily_max_drawdown_pct: float
    per_contract_fee: float
    per_order_fee: float
    volatility_gate: float


@dataclass(slots=True)
class HFStatus:
    status: str
    started_at: Optional[str]
    account_equity: float
    open_positions: int
    realized_pl_day: float
    unrealized_pl: float
    win_rate_pct: float
    halted: bool


@dataclass(slots=True)
class HFActivity:
    id: int
    trade_date: str
    ticker: str
    option_type: str
    strike: float | None
    expiry: str | None
    qty: int
    entry_time: str
    entry_price: float
    exit_time: str | None
    exit_price: float | None
    roi_pct: float | None
    fees: float
    status: str


@dataclass(slots=True)
class EquityPoint:
    ts: str
    balance: float


def _default_settings() -> HFSettings:
    return HFSettings(
        starting_balance=100000.0,
        pct_per_trade=1.0,
        daily_trade_cap=50,
        tickers=_DEFAULT_TICKERS,
        profit_target_pct=4.0,
        max_adverse_pct=-2.0,
        time_cap_minutes=5,
        cooldown_minutes=2,
        max_open_positions=2,
        daily_max_drawdown_pct=-6.0,
        per_contract_fee=0.65,
        per_order_fee=0.0,
        volatility_gate=3.0,
    )


def _coerce_float(value: Any, *, default: float, key: str) -> float:
    try:
        if value is None:
            raise ValueError("missing")
        return float(value)
    except (TypeError, ValueError):
        logger.warning("hf_settings_invalid_float key=%s value=%r", key, value)
        return float(default)


def _coerce_int(value: Any, *, default: int, key: str) -> int:
    try:
        if value is None:
            raise ValueError("missing")
        return int(value)
    except (TypeError, ValueError):
        logger.warning("hf_settings_invalid_int key=%s value=%r", key, value)
        return int(default)


def _row_value(
    row: Any, key: str, index: int | None = None, default: Any = None
) -> Any:
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(key, default)
    if key:
        try:
            return row[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            pass
    if index is not None:
        try:
            return row[index]  # type: ignore[index]
        except (IndexError, TypeError):
            pass
    return default


def _create_hf_schema(db) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_hf_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            starting_balance REAL NOT NULL,
            pct_per_trade REAL NOT NULL,
            daily_trade_cap INTEGER NOT NULL,
            tickers TEXT NOT NULL,
            profit_target_pct REAL NOT NULL,
            max_adverse_pct REAL NOT NULL,
            time_cap_minutes INTEGER NOT NULL,
            cooldown_minutes INTEGER NOT NULL,
            max_open_positions INTEGER NOT NULL,
            daily_max_drawdown_pct REAL NOT NULL,
            per_contract_fee REAL NOT NULL,
            per_order_fee REAL NOT NULL,
            volatility_gate REAL NOT NULL
        )
        """
    )
    db.execute(
        """
        INSERT OR IGNORE INTO scalper_hf_settings (
            id, starting_balance, pct_per_trade, daily_trade_cap, tickers,
            profit_target_pct, max_adverse_pct, time_cap_minutes,
            cooldown_minutes, max_open_positions, daily_max_drawdown_pct,
            per_contract_fee, per_order_fee, volatility_gate
        ) VALUES (
            1, 100000.0, 1.0, 50, ?,
            4.0, -2.0, 5,
            2, 2, -6.0,
            0.65, 0.0, 3.0
        )
        """,
        (_DEFAULT_TICKERS,),
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_hf_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            status TEXT NOT NULL,
            started_at TEXT,
            halted_at TEXT,
            halt_reason TEXT
        )
        """
    )
    db.execute(
        """
        INSERT OR IGNORE INTO scalper_hf_state(id, status, started_at, halted_at, halt_reason)
        VALUES(1, 'inactive', NULL, NULL, NULL)
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_hf_equity (
            ts TEXT PRIMARY KEY,
            balance REAL NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_hf_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL,
            expiry TEXT,
            qty INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_time TEXT,
            exit_price REAL,
            roi_pct REAL,
            fees REAL NOT NULL,
            status TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            reason TEXT,
            realized_pl REAL,
            net_pl REAL
        )
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scalper_hf_activity_date
            ON scalper_hf_activity(trade_date)
        """
    )


def _ensure_schema(db) -> None:
    _ensure_row_factory(db)
    conn = getattr(db, "connection", None)
    if conn is None:
        raise RuntimeError("hf_engine._ensure_schema requires a DB connection")

    if getattr(conn, "in_transaction", False):
        _create_hf_schema(db)
        return

    def _apply() -> None:
        db.execute("BEGIN IMMEDIATE")
        try:
            _create_hf_schema(db)
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    db_module.retry_locked(_apply)


def load_settings(db) -> HFSettings:
    _ensure_schema(db)
    row = db.execute(
        """
        SELECT starting_balance, pct_per_trade, daily_trade_cap, tickers,
               profit_target_pct, max_adverse_pct, time_cap_minutes,
               cooldown_minutes, max_open_positions, daily_max_drawdown_pct,
               per_contract_fee, per_order_fee, volatility_gate
          FROM scalper_hf_settings WHERE id = 1
        """
    ).fetchone()
    defaults = _default_settings()
    if row is None:
        logger.warning("hf_settings_missing_row using defaults")
        return defaults
    tickers_value = _row_value(row, "tickers", 3)
    tickers = str(tickers_value).strip() if tickers_value is not None else ""
    if not tickers:
        tickers = defaults.tickers
    return HFSettings(
        starting_balance=_coerce_float(
            _row_value(row, "starting_balance", 0),
            default=defaults.starting_balance,
            key="starting_balance",
        ),
        pct_per_trade=_coerce_float(
            _row_value(row, "pct_per_trade", 1),
            default=defaults.pct_per_trade,
            key="pct_per_trade",
        ),
        daily_trade_cap=_coerce_int(
            _row_value(row, "daily_trade_cap", 2),
            default=defaults.daily_trade_cap,
            key="daily_trade_cap",
        ),
        tickers=tickers or defaults.tickers,
        profit_target_pct=_coerce_float(
            _row_value(row, "profit_target_pct", 4),
            default=defaults.profit_target_pct,
            key="profit_target_pct",
        ),
        max_adverse_pct=_coerce_float(
            _row_value(row, "max_adverse_pct", 5),
            default=defaults.max_adverse_pct,
            key="max_adverse_pct",
        ),
        time_cap_minutes=_coerce_int(
            _row_value(row, "time_cap_minutes", 6),
            default=defaults.time_cap_minutes,
            key="time_cap_minutes",
        ),
        cooldown_minutes=_coerce_int(
            _row_value(row, "cooldown_minutes", 7),
            default=defaults.cooldown_minutes,
            key="cooldown_minutes",
        ),
        max_open_positions=_coerce_int(
            _row_value(row, "max_open_positions", 8),
            default=defaults.max_open_positions,
            key="max_open_positions",
        ),
        daily_max_drawdown_pct=_coerce_float(
            _row_value(row, "daily_max_drawdown_pct", 9),
            default=defaults.daily_max_drawdown_pct,
            key="daily_max_drawdown_pct",
        ),
        per_contract_fee=_coerce_float(
            _row_value(row, "per_contract_fee", 10),
            default=defaults.per_contract_fee,
            key="per_contract_fee",
        ),
        per_order_fee=_coerce_float(
            _row_value(row, "per_order_fee", 11),
            default=defaults.per_order_fee,
            key="per_order_fee",
        ),
        volatility_gate=_coerce_float(
            _row_value(row, "volatility_gate", 12),
            default=defaults.volatility_gate,
            key="volatility_gate",
        ),
    )


def update_settings(
    db,
    *,
    starting_balance: float,
    pct_per_trade: float,
    daily_trade_cap: int,
    tickers: Sequence[str] | str,
    profit_target_pct: float,
    max_adverse_pct: float,
    time_cap_minutes: int,
    cooldown_minutes: int,
    max_open_positions: int,
    daily_max_drawdown_pct: float,
    per_contract_fee: float,
    per_order_fee: float,
    volatility_gate: float,
) -> HFSettings:
    _ensure_schema(db)
    current_settings = load_settings(db)
    tickers_text = ",".join(tickers) if isinstance(tickers, (list, tuple, set)) else str(tickers)
    db.execute(
        """
        UPDATE scalper_hf_settings
           SET starting_balance=?, pct_per_trade=?, daily_trade_cap=?, tickers=?,
               profit_target_pct=?, max_adverse_pct=?, time_cap_minutes=?,
               cooldown_minutes=?, max_open_positions=?, daily_max_drawdown_pct=?,
               per_contract_fee=?, per_order_fee=?, volatility_gate=?
         WHERE id=1
        """,
        (
            float(starting_balance),
            max(0.1, min(float(pct_per_trade), 3.0)),
            int(max(0, daily_trade_cap)),
            tickers_text,
            float(profit_target_pct),
            float(max_adverse_pct),
            int(max(1, time_cap_minutes)),
            int(max(0, cooldown_minutes)),
            int(max(1, max_open_positions)),
            float(daily_max_drawdown_pct),
            max(0.0, float(per_contract_fee)),
            max(0.0, float(per_order_fee)),
            max(0.0, float(volatility_gate)),
        ),
    )
    _maybe_reset_equity_seed(
        db,
        previous_balance=current_settings.starting_balance,
        new_balance=float(starting_balance),
    )
    db.connection.commit()
    return load_settings(db)


def _now_utc(ts: datetime | None = None) -> datetime:
    if ts:
        return ts.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _ensure_equity_seed(db, settings: HFSettings, *, now: datetime | None = None) -> None:
    row = db.execute("SELECT COUNT(1) FROM scalper_hf_equity").fetchone()
    count = int(row[0]) if row else 0
    if count == 0:
        current_dt = _now_utc(now)
        current = current_dt.isoformat()
        db.execute(
            "INSERT OR REPLACE INTO scalper_hf_equity(ts, balance) VALUES(?, ?)",
            (current, float(settings.starting_balance)),
        )
        paper_trading.append_equity_point(
            db, "hf", float(settings.starting_balance), ts=int(current_dt.timestamp() * 1000)
        )
        db.connection.commit()
        return
    if now is None:
        return
    activity_row = db.execute("SELECT COUNT(1) FROM scalper_hf_activity").fetchone()
    activity_count = int(activity_row[0]) if activity_row else 0
    if activity_count > 0:
        return
    current_dt = _now_utc(now)
    current = current_dt.isoformat()
    db.execute(
        "UPDATE scalper_hf_equity SET ts=?, balance=? WHERE ROWID = (SELECT ROWID FROM scalper_hf_equity LIMIT 1)",
        (current, float(settings.starting_balance)),
    )
    paper_trading.append_equity_point(
        db, "hf", float(settings.starting_balance), ts=int(current_dt.timestamp() * 1000)
    )
    db.connection.commit()


def _maybe_reset_equity_seed(
    db,
    *,
    previous_balance: float,
    new_balance: float,
    now: datetime | None = None,
) -> None:
    if abs(new_balance - previous_balance) < 1e-6:
        return
    equity_row = db.execute("SELECT COUNT(1) FROM scalper_hf_equity").fetchone()
    equity_count = int(equity_row[0]) if equity_row else 0
    if equity_count > 1:
        logger.info(
            "hf_equity_seed_reset_skipped reason=history previous=%.2f new=%.2f",
            previous_balance,
            new_balance,
        )
        return
    activity_row = db.execute("SELECT COUNT(1) FROM scalper_hf_activity").fetchone()
    activity_count = int(activity_row[0]) if activity_row else 0
    if activity_count > 0:
        logger.info(
            "hf_equity_seed_reset_skipped reason=activity previous=%.2f new=%.2f",
            previous_balance,
            new_balance,
        )
        return
    seed_ts: str | None = None
    if equity_count == 1:
        row = db.execute(
            "SELECT ts FROM scalper_hf_equity ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            seed_ts = str(row[0])
    if not seed_ts:
        seed_ts = _now_utc(now).isoformat()
    db.execute("DELETE FROM scalper_hf_equity")
    db.execute(
        "INSERT INTO scalper_hf_equity(ts, balance) VALUES(?, ?)",
        (seed_ts, float(new_balance)),
    )
    seed_dt: datetime | None = None
    try:
        seed_dt = datetime.fromisoformat(seed_ts)
    except ValueError:
        seed_dt = None
    ts_ms = int(seed_dt.timestamp() * 1000) if seed_dt else None
    paper_trading.append_equity_point(db, "hf", float(new_balance), ts=ts_ms)
    logger.info(
        "hf_equity_seed_reset previous=%.2f new=%.2f", previous_balance, new_balance
    )


def start_engine(db, *, now: datetime | None = None) -> HFStatus:
    _ensure_schema(db)
    settings = load_settings(db)
    started = _now_utc(now).isoformat()
    db.execute(
        "UPDATE scalper_hf_state SET status='active', started_at=?, halted_at=NULL, halt_reason=NULL WHERE id=1",
        (started,),
    )
    db.connection.commit()
    _ensure_equity_seed(db, settings, now=now)
    paper_trading.seed_equity_if_empty(db, "hf")
    return get_status(db, settings=settings)


def stop_engine(db, *, now: datetime | None = None) -> HFStatus:
    _ensure_schema(db)
    db.execute("UPDATE scalper_hf_state SET status='inactive' WHERE id=1")
    db.connection.commit()
    return get_status(db)


def restart_engine(db, *, now: datetime | None = None) -> HFStatus:
    _ensure_schema(db)
    settings = load_settings(db)
    started = _now_utc(now).isoformat()
    db.execute(
        "UPDATE scalper_hf_state SET status='active', started_at=?, halted_at=NULL, halt_reason=NULL WHERE id=1",
        (started,),
    )
    db.connection.commit()
    _ensure_equity_seed(db, settings, now=now)
    paper_trading.seed_equity_if_empty(db, "hf")
    return get_status(db, settings=settings)


def get_status(db, *, settings: HFSettings | None = None) -> HFStatus:
    _ensure_schema(db)
    settings = settings or load_settings(db)
    _ensure_equity_seed(db, settings)
    state = db.execute(
        "SELECT status, started_at, halted_at FROM scalper_hf_state WHERE id=1"
    ).fetchone()
    if state is None:
        logger.warning("hf_state_missing defaulting to inactive")
    status_value = _row_value(state, "status", 0, "inactive")
    status = str(status_value or "inactive")
    started_raw = _row_value(state, "started_at", 1)
    started_at = str(started_raw) if started_raw else None
    halted = bool(_row_value(state, "halted_at", 2))
    equity = float(_latest_equity(db) or settings.starting_balance)
    open_positions = _count_open_positions(db)
    realized_today = _realized_today(db)
    win_rate = _win_rate(db)
    return HFStatus(
        status=status,
        started_at=started_at,
        account_equity=equity,
        open_positions=open_positions,
        realized_pl_day=realized_today,
        unrealized_pl=0.0,
        win_rate_pct=win_rate,
        halted=halted,
    )


def current_equity(db) -> float:
    _ensure_schema(db)
    return _latest_equity(db)


def status_payload(db) -> Dict[str, Any]:
    settings = load_settings(db)
    st = get_status(db, settings=settings)
    return {
        "status": st.status,
        "started_at": st.started_at,
        "account_equity": float(st.account_equity or 0.0),
        "open_positions": st.open_positions,
        "realized_pl_day": st.realized_pl_day,
        "unrealized_pl": st.unrealized_pl,
        "win_rate_pct": st.win_rate_pct,
        "halted": st.halted,
        "config": {
            "pct_per_trade": settings.pct_per_trade,
            "daily_trade_cap": settings.daily_trade_cap,
            "cooldown_minutes": settings.cooldown_minutes,
            "max_open_positions": settings.max_open_positions,
            "daily_max_drawdown_pct": settings.daily_max_drawdown_pct,
        },
    }


def calculate_position_size(balance: float, pct_per_trade: float, mid_price: float) -> int:
    return _shared_position_size(balance, pct_per_trade, mid_price)


def _latest_equity(db) -> float:
    result = db.execute("SELECT balance FROM scalper_hf_equity ORDER BY ts DESC LIMIT 1")
    scalar = getattr(result, "scalar_one_or_none", None)
    val: Any = None
    if callable(scalar):
        val = scalar()
    else:
        row = getattr(result, "fetchone", lambda: None)()
        if isinstance(row, Mapping):
            val = row.get("balance")
        elif row is not None:
            try:
                val = row[0]
            except (IndexError, TypeError):
                val = None
        else:
            val = None
    return float(val) if val is not None else 0.0


def _count_open_positions(db) -> int:
    row = db.execute(
        "SELECT COUNT(1) FROM scalper_hf_activity WHERE status='open'"
    ).fetchone()
    return int(row[0]) if row else 0


def _realized_today(db) -> float:
    today = now_et().date()
    start = datetime.combine(today, datetime.min.time()).replace(tzinfo=TZ)
    end = start + timedelta(days=1)
    row = db.execute(
        """
        SELECT COALESCE(SUM(net_pl), 0) FROM scalper_hf_activity
         WHERE exit_time >= ? AND exit_time < ?
        """,
        (start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()),
    ).fetchone()
    return float(row[0]) if row else 0.0


def _win_rate(db) -> float:
    rows = db.execute(
        "SELECT net_pl FROM scalper_hf_activity WHERE status='closed' AND net_pl IS NOT NULL"
    ).fetchall()
    if not rows:
        return 0.0
    wins = 0
    for row in rows:
        net_pl = _row_value(row, "net_pl", 0, 0.0)
        try:
            value = float(net_pl or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            wins += 1
    return round(wins / len(rows) * 100.0, 2)


def _dedupe_key(ticker: str, option_type: str, strike: float | None, expiry: str | None) -> str:
    parts = [ticker.upper(), option_type.upper()]
    if strike is not None:
        parts.append(f"{float(strike):.2f}")
    if expiry:
        parts.append(str(expiry))
    return "|".join(parts)


def _slippage_ticks(liquidity_score: float | None) -> int:
    if liquidity_score is None:
        return 1
    score = max(0.0, float(liquidity_score))
    if score >= 0.75:
        return 1
    if score >= 0.4:
        return 2
    return 3


def _cooldown_active(db, now_ts: datetime, *, cooldown_minutes: int) -> bool:
    row = db.execute(
        "SELECT exit_time FROM scalper_hf_activity WHERE exit_time IS NOT NULL ORDER BY exit_time DESC LIMIT 1"
    ).fetchone()
    if not row or not row["exit_time"]:
        return False
    try:
        exit_ts = datetime.fromisoformat(str(row["exit_time"]))
    except ValueError:
        return False
    cooldown = exit_ts + timedelta(minutes=cooldown_minutes)
    return now_ts < cooldown


def _daily_trade_count(db, session_date: date) -> int:
    start = datetime.combine(session_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    row = db.execute(
        """
        SELECT COUNT(1) FROM scalper_hf_activity
         WHERE entry_time >= ? AND entry_time < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    return int(row[0]) if row else 0


def _session_start_equity(db, session_date: date, *, default: float) -> float:
    start = datetime.combine(session_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    row = db.execute(
        """
        SELECT balance FROM scalper_hf_equity
         WHERE ts >= ? AND ts < ?
         ORDER BY ts ASC LIMIT 1
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    if not row:
        return default
    return float(row["balance"])


def _halt_engine(db, *, reason: str, now: datetime) -> None:
    db.execute(
        "UPDATE scalper_hf_state SET status='halted', halted_at=?, halt_reason=? WHERE id=1",
        (now.isoformat(), reason),
    )
    db.connection.commit()


def _drawdown_breached(db, now_ts: datetime, settings: HFSettings) -> bool:
    if settings.daily_max_drawdown_pct >= 0:
        return False
    current = _latest_equity(db)
    session_start = _session_start_equity(db, now_ts.date(), default=settings.starting_balance)
    baseline = float(settings.starting_balance) if settings.starting_balance > 0 else session_start
    if session_start <= 0 or baseline <= 0:
        return False
    loss = session_start - current
    if loss <= 0:
        return False
    threshold = abs(float(settings.daily_max_drawdown_pct)) / 100.0 * baseline
    return loss >= threshold


def open_trade(
    db,
    *,
    ticker: str,
    option_type: str,
    strike: float | None,
    expiry: str | None,
    mid_price: float,
    entry_time: datetime | None = None,
    momentum_score: float | None = None,
    vwap: float | None = None,
    ema9: float | None = None,
    volatility: float | None = None,
    liquidity: float | None = None,
    settings: HFSettings | None = None,
) -> Optional[int]:
    _ensure_schema(db)
    settings = settings or load_settings(db)
    state = get_status(db)
    if state.status != "active":
        logger.info("hf_trade_skipped status=%s", state.status)
        return None

    entry_dt = _now_utc(entry_time)
    if _drawdown_breached(db, entry_dt, settings):
        _halt_engine(db, reason="drawdown", now=entry_dt)
        logger.warning("hf_drawdown_halt equity=%s", _latest_equity(db))
        return None

    if _cooldown_active(db, entry_dt, cooldown_minutes=settings.cooldown_minutes):
        logger.info("hf_trade_cooldown")
        return None

    if _count_open_positions(db) >= settings.max_open_positions:
        logger.info("hf_trade_max_positions")
        return None

    if _daily_trade_count(db, entry_dt.date()) >= settings.daily_trade_cap:
        logger.info("hf_trade_cap_reached date=%s cap=%s", entry_dt.date(), settings.daily_trade_cap)
        return None

    dedupe_key = _dedupe_key(ticker, option_type, strike, expiry)
    existing = db.execute(
        "SELECT id FROM scalper_hf_activity WHERE status='open' AND dedupe_key=?",
        (dedupe_key,),
    ).fetchone()
    if existing:
        logger.info("hf_trade_dedupe ticker=%s option=%s", ticker, option_type)
        return None

    price = float(mid_price)
    vwap_val = float(vwap) if vwap is not None else price
    ema_val = float(ema9) if ema9 is not None else price
    momentum_val = float(momentum_score) if momentum_score is not None else 0.0
    vol_val = float(volatility) if volatility is not None else 0.0

    if momentum_val < 0.5:
        logger.info("hf_trade_signal_momentum_insufficient score=%.3f", momentum_val)
        return None
    if price > vwap_val * 1.01:
        logger.info("hf_trade_signal_no_pullback price=%.2f vwap=%.2f", price, vwap_val)
        return None
    if price < ema_val * 0.995:
        logger.info("hf_trade_signal_below_ema price=%.2f ema=%.2f", price, ema_val)
        return None
    if settings.volatility_gate > 0 and vol_val > settings.volatility_gate:
        logger.info("hf_trade_volatility_block vol=%.2f gate=%.2f", vol_val, settings.volatility_gate)
        return None

    equity = _latest_equity(db)
    qty = calculate_position_size(equity, settings.pct_per_trade, price)
    if qty <= 0:
        logger.info("hf_trade_qty_zero ticker=%s", ticker)
        return None

    ticks = _slippage_ticks(liquidity)
    entry_price = apply_slippage(price, side="buy", ticks=ticks)
    fee_model = FeeModel(per_contract=settings.per_contract_fee, per_order=settings.per_order_fee)
    fees = fee_model.order_fees(qty)
    trade_date = entry_dt.date().isoformat()
    db.execute(
        """
        INSERT INTO scalper_hf_activity(
            trade_date, ticker, option_type, strike, expiry, qty,
            entry_time, entry_price, fees, status, dedupe_key
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trade_date,
            ticker.upper(),
            option_type.upper(),
            strike,
            expiry,
            qty,
            entry_dt.isoformat(),
            entry_price,
            fees,
            "open",
            dedupe_key,
        ),
    )
    db.connection.commit()
    row = db.execute("SELECT last_insert_rowid() AS id").fetchone()
    return int(row["id"]) if row else None


def close_trade(
    db,
    trade_id: int,
    *,
    mid_price: float,
    exit_time: datetime | None = None,
    reason: str = "exit",
    liquidity: float | None = None,
    settings: HFSettings | None = None,
) -> Optional[HFActivity]:
    _ensure_schema(db)
    settings = settings or load_settings(db)
    row = db.execute(
        "SELECT id, qty, entry_price, fees FROM scalper_hf_activity WHERE id=? AND status='open'",
        (trade_id,),
    ).fetchone()
    if not row:
        return None
    exit_dt = _now_utc(exit_time)
    qty = int(row["qty"])
    entry_price = float(row["entry_price"])
    entry_fees = float(row["fees"])
    ticks = _slippage_ticks(liquidity)
    exit_price = apply_slippage(mid_price, side="sell", ticks=ticks)
    fee_model = FeeModel(per_contract=settings.per_contract_fee, per_order=settings.per_order_fee)
    exit_fees = fee_model.order_fees(qty)
    gross = (exit_price - entry_price) * qty * 100.0
    total_fees = entry_fees + exit_fees
    net = gross - total_fees
    roi = 0.0 if entry_price <= 0 else round((exit_price - entry_price) / entry_price * 100.0, 2)

    db.execute(
        """
        UPDATE scalper_hf_activity
           SET exit_time=?, exit_price=?, roi_pct=?, fees=?, status='closed',
               reason=?, realized_pl=?, net_pl=?
         WHERE id=?
        """,
        (
            exit_dt.isoformat(),
            exit_price,
            roi,
            total_fees,
            reason,
            gross,
            net,
            trade_id,
        ),
    )
    new_equity = _latest_equity(db) + net
    db.execute(
        "INSERT INTO scalper_hf_equity(ts, balance) VALUES(?, ?)",
        (exit_dt.isoformat(), new_equity),
    )
    paper_trading.append_equity_point(
        db, "hf", float(new_equity), ts=int(exit_dt.timestamp() * 1000)
    )
    db.connection.commit()
    updated = db.execute(
        """
        SELECT id, trade_date, ticker, option_type, strike, expiry, qty,
               entry_time, entry_price, exit_time, exit_price, roi_pct,
               fees, status
          FROM scalper_hf_activity WHERE id=?
        """,
        (trade_id,),
    ).fetchone()
    if not updated:
        return None
    return HFActivity(
        id=int(updated["id"]),
        trade_date=str(updated["trade_date"]),
        ticker=str(updated["ticker"]),
        option_type=str(updated["option_type"]),
        strike=updated["strike"],
        expiry=updated["expiry"],
        qty=int(updated["qty"]),
        entry_time=str(updated["entry_time"]),
        entry_price=float(updated["entry_price"]),
        exit_time=updated["exit_time"],
        exit_price=updated["exit_price"],
        roi_pct=updated["roi_pct"],
        fees=float(updated["fees"]),
        status=str(updated["status"]),
    )


def list_activity(db, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    _ensure_schema(db)
    query = (
        "SELECT trade_date, ticker, option_type, strike, expiry, qty, entry_time,"
        " entry_price, exit_time, exit_price, roi_pct, fees, status FROM"
        " scalper_hf_activity ORDER BY entry_time DESC"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = db.execute(query).fetchall()
    data: List[Dict[str, Any]] = []
    for row in rows:
        data.append(
            {
                "date": row["trade_date"],
                "ticker": row["ticker"],
                "call_put": row["option_type"],
                "strike": row["strike"],
                "expiry": row["expiry"],
                "qty": row["qty"],
                "entry_time": row["entry_time"],
                "entry_price": row["entry_price"],
                "exit_time": row["exit_time"],
                "exit_price": row["exit_price"],
                "roi_pct": row["roi_pct"],
                "fees": row["fees"],
                "status": row["status"],
            }
        )
    return data


def export_activity_csv(db) -> tuple[str, str]:
    rows = list_activity(db)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
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
    )
    for row in rows:
        strike = row["strike"]
        expiry = row["expiry"]
        strike_label = "—"
        if strike is not None:
            strike_label = f"${float(strike):.2f}".rstrip("0").rstrip(".")
        expiry_label = expiry or "—"
        writer.writerow(
            [
                row["date"],
                row["ticker"],
                row["call_put"],
                f"{strike_label}-{expiry_label}",
                row["qty"],
                row["entry_time"],
                row["entry_price"],
                row["exit_time"],
                row["exit_price"],
                row["roi_pct"],
                row["fees"],
            ]
        )
    return output.getvalue(), "scalper_hf_activity.csv"


def get_equity_points(db, range_key: str) -> List[EquityPoint]:
    _ensure_schema(db)
    limit = {
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "1y": timedelta(days=365),
    }.get(range_key, timedelta(days=30))
    cutoff = _now_utc() - limit
    rows = db.execute(
        "SELECT ts, balance FROM scalper_hf_equity WHERE ts >= ? ORDER BY ts ASC",
        (cutoff.isoformat(),),
    ).fetchall()
    return [EquityPoint(ts=row["ts"], balance=float(row["balance"])) for row in rows]


def metrics_snapshot(db) -> Dict[str, float]:
    _ensure_schema(db)
    settings = load_settings(db)
    rows = db.execute(
        """
        SELECT entry_time, exit_time, roi_pct, realized_pl, net_pl
          FROM scalper_hf_activity
         WHERE status='closed'
         ORDER BY exit_time ASC
        """
    ).fetchall()
    trade_rows = [
        {
            "entry_time": row["entry_time"],
            "exit_time": row["exit_time"],
            "roi_pct": row["roi_pct"],
            "realized_pl": row["realized_pl"],
            "net_pl": row["net_pl"],
        }
        for row in rows
    ]
    return dict(
        compute_trade_metrics(trade_rows, starting_balance=settings.starting_balance)
    )


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    ema_values: List[float] = []
    ema = values[0]
    for price in values:
        ema = price * k + ema * (1 - k)
        ema_values.append(ema)
    return ema_values


def _normalize_liquidity(bar: Mapping[str, Any]) -> float:
    volume = float(bar.get("volume") or bar.get("oi") or 0.0)
    if volume <= 0:
        return 0.0
    scale = 100000.0
    return max(0.0, min(volume / scale, 1.0))


def _estimate_volatility(bar: Mapping[str, Any]) -> float:
    if "atr" in bar:
        return float(bar["atr"])
    if "volatility" in bar:
        return float(bar["volatility"])
    high = float(bar.get("high", 0.0))
    low = float(bar.get("low", 0.0))
    return abs(high - low)


def run_backtest(
    db,
    *,
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: HFSettings | None = None,
) -> Dict[str, Any]:
    settings = settings or load_settings(db)
    balance = settings.starting_balance
    equity_curve: List[Dict[str, Any]] = []
    cooldown_until: datetime | None = None
    fee_model = FeeModel(per_contract=settings.per_contract_fee, per_order=settings.per_order_fee)

    for symbol, bars in bars_by_symbol.items():
        closes = [float(bar.get("close", 0.0)) for bar in bars]
        ema9 = _ema(closes, 9)
        open_trade_price: float | None = None
        entry_idx: int | None = None
        entry_ts: datetime | None = None
        for idx, bar in enumerate(bars):
            ts_raw = bar.get("ts")
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                ts = None
            price = float(bar.get("close", 0.0))
            if ts and cooldown_until and ts < cooldown_until:
                continue
            momentum = 0.0
            if idx > 0:
                momentum = price - float(bars[idx - 1].get("close", price))
            vwap = float(bar.get("vwap", price))
            ema_val = ema9[idx] if idx < len(ema9) else price
            vol = _estimate_volatility(bar)
            liquidity = _normalize_liquidity(bar)

            if open_trade_price is None:
                if (
                    momentum >= 0.4
                    and price <= vwap * 1.01
                    and price >= ema_val * 0.995
                    and (settings.volatility_gate <= 0 or vol <= settings.volatility_gate)
                ):
                    open_trade_price = apply_slippage(price, side="buy", ticks=_slippage_ticks(liquidity))
                    entry_idx = idx
                    entry_ts = ts
                continue

            if open_trade_price is not None and entry_idx is not None:
                segment = bars[entry_idx : idx + 1]
                exit_price, reason, exit_ts_raw = evaluate_exit(
                    entry_price=open_trade_price,
                    bars=segment,
                    target_pct=settings.profit_target_pct,
                    stop_pct=settings.max_adverse_pct,
                    time_cap_minutes=settings.time_cap_minutes,
                )
                exit_ts = None
                if exit_ts_raw:
                    try:
                        exit_ts = datetime.fromisoformat(exit_ts_raw)
                    except ValueError:
                        exit_ts = ts
                qty = max(1, calculate_position_size(balance, settings.pct_per_trade, open_trade_price))
                entry_fees = fee_model.order_fees(qty)
                exit_fees = fee_model.order_fees(qty)
                gross = (exit_price - open_trade_price) * qty * 100.0
                net = gross - (entry_fees + exit_fees)
                balance += net
                point_ts = exit_ts or ts or entry_ts
                equity_curve.append(
                    {
                        "ts": point_ts.isoformat() if point_ts else None,
                        "balance": balance,
                        "symbol": symbol,
                        "reason": reason,
                        "net": net,
                    }
                )
                cooldown_until = (exit_ts or ts or entry_ts or datetime.now(timezone.utc)) + timedelta(
                    minutes=settings.cooldown_minutes
                )
                open_trade_price = None
                entry_idx = None
                entry_ts = None

    summary = summarize_backtest(equity_curve, starting_balance=settings.starting_balance)
    return {"equity_curve": equity_curve, "summary": summary}

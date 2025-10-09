"""Low-frequency paper scalper engine.

The implementation is intentionally self-contained and focuses on deterministic
behaviour so it can be exercised in unit tests without external market data.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils import TZ, now_et

from .shared import (
    FeeModel,
    apply_slippage,
    calculate_position_size as _shared_position_size,
    compute_trade_metrics,
    summarize_backtest,
)

logger = logging.getLogger(__name__)

_DEFAULT_TICKERS = "SPY,QQQ,TSLA,NVDA"


def _ensure_row_factory(db) -> None:
    conn = getattr(db, "connection", None)
    if conn is None:
        return
    try:
        if getattr(conn, "row_factory", None) is not sqlite3.Row:
            conn.row_factory = sqlite3.Row
    except Exception:
        # Some third-party cursors may not allow reassignment; ignore quietly.
        pass


def _row_value(row: Any, key: str, index: int | None = None, default: Any = None) -> Any:
    """Return a column value from either mapping-style or tuple rows."""

    if row is None:
        return default
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


def _coerce_float(value: Any, *, default: float, key: str) -> float:
    try:
        if value is None:
            raise ValueError("missing")
        return float(value)
    except (TypeError, ValueError):
        logger.warning("lf_settings_invalid_float key=%s value=%r", key, value)
        return float(default)


def _coerce_int(value: Any, *, default: int, key: str) -> int:
    try:
        if value is None:
            raise ValueError("missing")
        return int(value)
    except (TypeError, ValueError):
        logger.warning("lf_settings_invalid_int key=%s value=%r", key, value)
        return int(default)


def _coerce_str(value: Any, *, default: str, key: str) -> str:
    if value is None:
        logger.warning("lf_settings_missing_str key=%s", key)
        return default
    return str(value)


def _coerce_bool(value: Any, *, default: bool, key: str) -> bool:
    if value is None:
        logger.warning("lf_settings_missing_bool key=%s", key)
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    logger.warning("lf_settings_invalid_bool key=%s value=%r", key, value)
    return default


def _default_settings() -> LFSettings:
    return LFSettings(
        starting_balance=100000.0,
        pct_per_trade=3.0,
        daily_trade_cap=20,
        tickers=_DEFAULT_TICKERS,
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


@dataclass(slots=True)
class LFSettings:
    starting_balance: float
    pct_per_trade: float
    daily_trade_cap: int
    tickers: str
    profit_target_pct: float
    max_adverse_pct: float
    time_cap_minutes: int
    session_start: str
    session_end: str
    allow_premarket: bool
    allow_postmarket: bool
    per_contract_fee: float
    per_order_fee: float
    rsi_filter: bool


@dataclass(slots=True)
class LFStatus:
    status: str
    started_at: Optional[str]
    account_equity: float
    open_positions: int
    realized_pl_day: float
    unrealized_pl: float
    win_rate_pct: float


@dataclass(slots=True)
class LFActivity:
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


def _ensure_schema(db) -> None:
    _ensure_row_factory(db)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_lf_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            starting_balance REAL NOT NULL,
            pct_per_trade REAL NOT NULL,
            daily_trade_cap INTEGER NOT NULL,
            tickers TEXT NOT NULL,
            profit_target_pct REAL NOT NULL,
            max_adverse_pct REAL NOT NULL,
            time_cap_minutes INTEGER NOT NULL,
            session_start TEXT NOT NULL,
            session_end TEXT NOT NULL,
            allow_premarket INTEGER NOT NULL,
            allow_postmarket INTEGER NOT NULL,
            per_contract_fee REAL NOT NULL,
            per_order_fee REAL NOT NULL,
            rsi_filter INTEGER NOT NULL
        )
        """
    )
    db.execute(
        """
        INSERT OR IGNORE INTO scalper_lf_settings(
            id, starting_balance, pct_per_trade, daily_trade_cap, tickers,
            profit_target_pct, max_adverse_pct, time_cap_minutes,
            session_start, session_end, allow_premarket, allow_postmarket,
            per_contract_fee, per_order_fee, rsi_filter
        ) VALUES(
            1, 100000.0, 3.0, 20, ?,
            6.0, -3.0, 15,
            '09:30', '16:00', 0, 0,
            0.65, 0.0, 0
        )
        """,
        (_DEFAULT_TICKERS,),
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_lf_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            status TEXT NOT NULL,
            started_at TEXT
        )
        """
    )
    db.execute(
        """INSERT OR IGNORE INTO scalper_lf_state(id, status, started_at)
               VALUES(1, 'inactive', NULL)"""
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_lf_equity (
            ts TEXT PRIMARY KEY,
            balance REAL NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scalper_lf_activity (
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
            net_pl REAL,
            session_id TEXT
        )
        """
    )
    cols = {
        (col["name"] if isinstance(col, Mapping) else col[1])
        for col in db.execute("PRAGMA table_info(scalper_lf_activity)").fetchall()
    }
    if "mark_price" not in cols:
        db.execute("ALTER TABLE scalper_lf_activity ADD COLUMN mark_price REAL")
    db.connection.commit()


def load_settings(db) -> LFSettings:
    _ensure_schema(db)
    row = db.execute(
        """
        SELECT starting_balance, pct_per_trade, daily_trade_cap, tickers,
               profit_target_pct, max_adverse_pct, time_cap_minutes,
               session_start, session_end, allow_premarket, allow_postmarket,
               per_contract_fee, per_order_fee, rsi_filter
          FROM scalper_lf_settings WHERE id=1
        """
    ).fetchone()
    defaults = _default_settings()
    if row is None:
        logger.warning("lf_settings_missing_row using defaults")
        return defaults
    tickers_value = _row_value(row, "tickers", 3)
    tickers = str(tickers_value).strip() if tickers_value is not None else ""
    if not tickers:
        tickers = defaults.tickers
    return LFSettings(
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
        session_start=_coerce_str(
            _row_value(row, "session_start", 7),
            default=defaults.session_start,
            key="session_start",
        ),
        session_end=_coerce_str(
            _row_value(row, "session_end", 8),
            default=defaults.session_end,
            key="session_end",
        ),
        allow_premarket=_coerce_bool(
            _row_value(row, "allow_premarket", 9),
            default=defaults.allow_premarket,
            key="allow_premarket",
        ),
        allow_postmarket=_coerce_bool(
            _row_value(row, "allow_postmarket", 10),
            default=defaults.allow_postmarket,
            key="allow_postmarket",
        ),
        per_contract_fee=_coerce_float(
            _row_value(row, "per_contract_fee", 11),
            default=defaults.per_contract_fee,
            key="per_contract_fee",
        ),
        per_order_fee=_coerce_float(
            _row_value(row, "per_order_fee", 12),
            default=defaults.per_order_fee,
            key="per_order_fee",
        ),
        rsi_filter=_coerce_bool(
            _row_value(row, "rsi_filter", 13),
            default=defaults.rsi_filter,
            key="rsi_filter",
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
    session_start: str,
    session_end: str,
    allow_premarket: bool,
    allow_postmarket: bool,
    per_contract_fee: float,
    per_order_fee: float,
    rsi_filter: bool,
) -> LFSettings:
    _ensure_schema(db)
    pct_value = max(0.5, min(float(pct_per_trade), 10.0))
    balance_value = max(0.0, float(starting_balance))
    cap_value = max(0, int(daily_trade_cap))
    tickers_value = ",".join(tickers) if isinstance(tickers, Sequence) and not isinstance(tickers, str) else str(tickers)
    tickers_value = ",".join(filter(None, [t.strip().upper() for t in tickers_value.split(",")])) or _DEFAULT_TICKERS
    target_pct = float(profit_target_pct)
    stop_pct = float(max_adverse_pct)
    time_cap = max(1, int(time_cap_minutes))
    per_contract = max(0.0, float(per_contract_fee))
    per_order = max(0.0, float(per_order_fee))
    db.execute(
        """
        UPDATE scalper_lf_settings
           SET starting_balance=?, pct_per_trade=?, daily_trade_cap=?, tickers=?,
               profit_target_pct=?, max_adverse_pct=?, time_cap_minutes=?,
               session_start=?, session_end=?, allow_premarket=?, allow_postmarket=?,
               per_contract_fee=?, per_order_fee=?, rsi_filter=?
         WHERE id=1
        """,
        (
            balance_value,
            pct_value,
            cap_value,
            tickers_value,
            target_pct,
            stop_pct,
            time_cap,
            session_start,
            session_end,
            1 if allow_premarket else 0,
            1 if allow_postmarket else 0,
            per_contract,
            per_order,
            1 if rsi_filter else 0,
        ),
    )
    db.connection.commit()
    return load_settings(db)


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _ensure_equity_seed(db, settings: LFSettings, *, now: datetime | None = None) -> None:
    row = db.execute("SELECT COUNT(1) FROM scalper_lf_equity").fetchone()
    count = int(row[0]) if row else 0
    if count:
        return
    current = _now_utc(now).isoformat()
    db.execute(
        "INSERT OR REPLACE INTO scalper_lf_equity(ts, balance) VALUES(?, ?)",
        (current, float(settings.starting_balance)),
    )
    db.connection.commit()


def start_engine(db, *, now: datetime | None = None) -> LFStatus:
    _ensure_schema(db)
    settings = load_settings(db)
    started = _now_utc(now).isoformat()
    db.execute(
        "UPDATE scalper_lf_state SET status='active', started_at=? WHERE id=1",
        (started,),
    )
    db.connection.commit()
    _ensure_equity_seed(db, settings, now=now)
    return get_status(db)


def stop_engine(db, *, now: datetime | None = None) -> LFStatus:
    _ensure_schema(db)
    db.execute("UPDATE scalper_lf_state SET status='inactive' WHERE id=1")
    db.connection.commit()
    return get_status(db)


def restart_engine(db, *, now: datetime | None = None) -> LFStatus:
    _ensure_schema(db)
    settings = load_settings(db)
    started = _now_utc(now).isoformat()
    db.execute(
        "UPDATE scalper_lf_state SET status='active', started_at=? WHERE id=1",
        (started,),
    )
    db.connection.commit()
    _ensure_equity_seed(db, settings, now=now)
    return get_status(db)


def get_status(db) -> LFStatus:
    _ensure_schema(db)
    state = db.execute(
        "SELECT status, started_at FROM scalper_lf_state WHERE id=1"
    ).fetchone()
    if state is None:
        logger.warning("lf_state_missing defaulting to inactive")
    status_value = _row_value(state, "status", 0, "inactive")
    status = str(status_value or "inactive")
    started_raw = _row_value(state, "started_at", 1)
    started_at = str(started_raw) if started_raw else None
    equity = float(_latest_equity(db) or 0.0)
    open_positions = _count_open_positions(db)
    realized_today = _realized_today(db)
    unrealized = _unrealized_pl(db)
    win_rate = _win_rate(db)
    return LFStatus(
        status=status,
        started_at=started_at,
        account_equity=equity,
        open_positions=open_positions,
        realized_pl_day=realized_today,
        unrealized_pl=unrealized,
        win_rate_pct=win_rate,
    )


def current_equity(db) -> float:
    _ensure_schema(db)
    return _latest_equity(db)


def status_payload(db) -> Dict[str, Any]:
    st = get_status(db)
    settings = load_settings(db)
    return {
        "status": st.status,
        "started_at": st.started_at,
        "account_equity": float(st.account_equity or 0.0),
        "open_positions": st.open_positions,
        "realized_pl_day": st.realized_pl_day,
        "unrealized_pl": st.unrealized_pl,
        "win_rate_pct": st.win_rate_pct,
        "settings": {
            "starting_balance": settings.starting_balance,
            "pct_per_trade": settings.pct_per_trade,
            "daily_trade_cap": settings.daily_trade_cap,
            "tickers": settings.tickers,
            "profit_target_pct": settings.profit_target_pct,
            "max_adverse_pct": settings.max_adverse_pct,
            "time_cap_minutes": settings.time_cap_minutes,
            "session_start": settings.session_start,
            "session_end": settings.session_end,
            "allow_premarket": settings.allow_premarket,
            "allow_postmarket": settings.allow_postmarket,
            "per_contract_fee": settings.per_contract_fee,
            "per_order_fee": settings.per_order_fee,
            "rsi_filter": settings.rsi_filter,
        },
    }




def metrics_snapshot(db) -> Dict[str, float]:
    _ensure_schema(db)
    settings = load_settings(db)
    rows = db.execute(
        """
        SELECT entry_time, exit_time, roi_pct, realized_pl, net_pl
          FROM scalper_lf_activity
         WHERE status='closed'
         ORDER BY exit_time ASC
        """
    ).fetchall()
    return dict(compute_trade_metrics(rows, starting_balance=settings.starting_balance))


def _latest_equity(db) -> float:
    result = db.execute("SELECT balance FROM scalper_lf_equity ORDER BY ts DESC LIMIT 1")
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
    if val is not None:
        return float(val)
    settings = load_settings(db)
    return float(settings.starting_balance)


def _count_open_positions(db) -> int:
    row = db.execute(
        "SELECT COUNT(1) FROM scalper_lf_activity WHERE status='open'"
    ).fetchone()
    return int(row[0]) if row else 0


def _realized_today(db) -> float:
    today = now_et().date()
    start = datetime.combine(today, datetime.min.time(), tzinfo=TZ).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    row = db.execute(
        """
        SELECT COALESCE(SUM(net_pl), 0) FROM scalper_lf_activity
         WHERE exit_time IS NOT NULL AND status='closed'
           AND exit_time >= ? AND exit_time < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    return float(row[0]) if row else 0.0


def _unrealized_pl(db) -> float:
    row = db.execute(
        """
        SELECT COALESCE(SUM((COALESCE(mark_price, entry_price) - entry_price) * qty * 100), 0)
          FROM scalper_lf_activity
         WHERE status='open'
        """
    ).fetchone()
    return float(row[0]) if row else 0.0


def _win_rate(db, lookback: int = 20) -> float:
    rows = db.execute(
        """
        SELECT net_pl FROM scalper_lf_activity
         WHERE status='closed'
         ORDER BY exit_time DESC
         LIMIT ?
        """,
        (lookback,),
    ).fetchall()
    if not rows:
        return 0.0
    wins = sum(
        1
        for row in rows
        if float(_row_value(row, "net_pl", 0, 0.0) or 0.0) > 0.0
    )
    return round(wins / len(rows) * 100.0, 2)


def calculate_position_size(balance: float, pct_per_trade: float, mid_price: float) -> int:
    return _shared_position_size(balance, pct_per_trade, mid_price)


def _apply_tick(mid_price: float, *, is_buy: bool, ticks: int = 1) -> float:
    side = "buy" if is_buy else "sell"
    return apply_slippage(mid_price, side=side, ticks=ticks)


def _order_fees(qty: int, *, per_contract: float, per_order: float) -> float:
    model = FeeModel(per_contract=float(per_contract), per_order=float(per_order))
    return model.order_fees(qty)


def open_trade(
    db,
    *,
    ticker: str,
    option_type: str,
    strike: float | None,
    expiry: str | None,
    mid_price: float,
    entry_time: datetime | None = None,
    settings: LFSettings | None = None,
) -> Optional[int]:
    _ensure_schema(db)
    settings = settings or load_settings(db)
    state = get_status(db)
    if state.status != "active":
        logger.info("lf_trade_skipped status=%s", state.status)
        return None

    entry_dt = _now_utc(entry_time)
    dedupe_key = _dedupe_key(ticker, option_type, strike, expiry)
    existing = db.execute(
        "SELECT id FROM scalper_lf_activity WHERE status='open' AND dedupe_key=?",
        (dedupe_key,),
    ).fetchone()
    if existing:
        logger.info("lf_trade_dedupe ticker=%s option=%s", ticker, option_type)
        return None

    if _daily_trade_count(db, entry_dt.date()) >= settings.daily_trade_cap:
        logger.info("lf_trade_cap_reached date=%s cap=%s", entry_dt.date(), settings.daily_trade_cap)
        return None

    equity = _latest_equity(db)
    qty = calculate_position_size(equity, settings.pct_per_trade, mid_price)
    if qty <= 0:
        logger.info("lf_trade_qty_zero ticker=%s", ticker)
        return None

    entry_price = _apply_tick(mid_price, is_buy=True)
    fees = _order_fees(qty, per_contract=settings.per_contract_fee, per_order=settings.per_order_fee)
    trade_date = entry_dt.date().isoformat()
    db.execute(
        """
        INSERT INTO scalper_lf_activity(
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
    if not row:
        return None
    identifier = _row_value(row, "id", 0)
    return int(identifier) if identifier is not None else None


def _daily_trade_count(db, session_date: date) -> int:
    start = datetime.combine(session_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    row = db.execute(
        """
        SELECT COUNT(1) FROM scalper_lf_activity
         WHERE entry_time >= ? AND entry_time < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    return int(row[0]) if row else 0


def _dedupe_key(ticker: str, option_type: str, strike: float | None, expiry: str | None) -> str:
    parts = [ticker.upper(), option_type.upper()]
    if strike is not None:
        parts.append(f"{float(strike):.2f}")
    if expiry:
        parts.append(str(expiry))
    return "|".join(parts)


def close_trade(
    db,
    trade_id: int,
    *,
    mid_price: float,
    exit_time: datetime | None = None,
    reason: str = "exit",
    settings: LFSettings | None = None,
) -> Optional[LFActivity]:
    _ensure_schema(db)
    settings = settings or load_settings(db)
    row = db.execute(
        "SELECT id, qty, entry_price, fees FROM scalper_lf_activity WHERE id=? AND status='open'",
        (trade_id,),
    ).fetchone()
    if not row:
        return None
    exit_dt = _now_utc(exit_time)
    qty = int(_row_value(row, "qty", 1, 0))
    entry_price = float(_row_value(row, "entry_price", 2, 0.0))
    entry_fees = float(_row_value(row, "fees", 3, 0.0))
    exit_price = _apply_tick(mid_price, is_buy=False)
    exit_fees = _order_fees(qty, per_contract=settings.per_contract_fee, per_order=settings.per_order_fee)
    gross = (exit_price - entry_price) * qty * 100.0
    total_fees = entry_fees + exit_fees
    net = gross - total_fees
    roi = 0.0 if entry_price <= 0 else round((exit_price - entry_price) / entry_price * 100.0, 2)

    db.execute(
        """
        UPDATE scalper_lf_activity
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
        "INSERT INTO scalper_lf_equity(ts, balance) VALUES(?, ?)",
        (exit_dt.isoformat(), new_equity),
    )
    db.connection.commit()
    updated = db.execute(
        """
        SELECT id, trade_date, ticker, option_type, strike, expiry, qty,
               entry_time, entry_price, exit_time, exit_price, roi_pct,
               fees, status
          FROM scalper_lf_activity WHERE id=?
        """,
        (trade_id,),
    ).fetchone()
    if not updated:
        return None
    updated_id = _row_value(updated, "id", 0, trade_id)
    trade_date = _row_value(updated, "trade_date", 1, "")
    ticker = _row_value(updated, "ticker", 2, "")
    option_type = _row_value(updated, "option_type", 3, "")
    strike = _row_value(updated, "strike", 4)
    expiry = _row_value(updated, "expiry", 5)
    qty_value = _row_value(updated, "qty", 6, 0)
    entry_time = _row_value(updated, "entry_time", 7, "")
    entry_price_val = _row_value(updated, "entry_price", 8, 0.0)
    exit_time = _row_value(updated, "exit_time", 9)
    exit_price = _row_value(updated, "exit_price", 10)
    roi_pct = _row_value(updated, "roi_pct", 11)
    fees_value = _row_value(updated, "fees", 12, 0.0)
    status_value = _row_value(updated, "status", 13, "")
    return LFActivity(
        id=int(updated_id) if updated_id is not None else trade_id,
        trade_date=str(trade_date),
        ticker=str(ticker),
        option_type=str(option_type),
        strike=strike,
        expiry=expiry,
        qty=int(qty_value),
        entry_time=str(entry_time),
        entry_price=float(entry_price_val),
        exit_time=None if exit_time is None else str(exit_time),
        exit_price=None if exit_price is None else float(exit_price),
        roi_pct=roi_pct,
        fees=float(fees_value),
        status=str(status_value),
    )


def list_activity(db, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    _ensure_schema(db)
    query = (
        "SELECT trade_date, ticker, option_type, strike, expiry, qty, entry_time,"
        " entry_price, exit_time, exit_price, roi_pct, fees, status FROM"
        " scalper_lf_activity ORDER BY entry_time DESC"
    )
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = db.execute(query).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        trade_date = _row_value(row, "trade_date", 0, "")
        ticker = _row_value(row, "ticker", 1, "")
        option_type = _row_value(row, "option_type", 2, "")
        strike = _row_value(row, "strike", 3)
        expiry = _row_value(row, "expiry", 4)
        qty_value = _row_value(row, "qty", 5, 0)
        entry_time = _row_value(row, "entry_time", 6, "")
        entry_price_val = _row_value(row, "entry_price", 7)
        exit_time = _row_value(row, "exit_time", 8)
        exit_price = _row_value(row, "exit_price", 9)
        roi_pct = _row_value(row, "roi_pct", 10)
        fees_value = _row_value(row, "fees", 11, 0.0)
        status_value = _row_value(row, "status", 12, "")
        results.append(
            {
                "date": str(trade_date),
                "ticker": str(ticker),
                "call_put": str(option_type),
                "strike": strike,
                "expiry": expiry,
                "qty": int(qty_value),
                "entry_time": str(entry_time),
                "entry_price": None if entry_price_val is None else float(entry_price_val),
                "exit_time": None if exit_time is None else str(exit_time),
                "exit_price": None if exit_price is None else float(exit_price),
                "roi_pct": roi_pct,
                "fees": float(fees_value),
                "status": str(status_value),
            }
        )
    return results


def export_activity_csv(db) -> Tuple[str, str]:
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
        if strike is None:
            strike_label = "—"
        else:
            strike_label = f"${float(strike):.2f}".rstrip("0").rstrip(".")
        expiry = row["expiry"] or "—"
        combo = f"{strike_label}-{expiry}"
        roi = row["roi_pct"]
        roi_label = "" if roi is None else f"{float(roi):.2f}"
        writer.writerow(
            [
                row["date"],
                row["ticker"],
                row["call_put"],
                combo,
                row["qty"],
                row["entry_time"],
                row["entry_price"],
                row["exit_time"],
                row["exit_price"],
                roi_label,
                f"{float(row['fees']):.2f}",
            ]
        )
    return output.getvalue(), "scalper_lf_activity.csv"


def get_equity_points(db, range_key: str) -> List[EquityPoint]:
    _ensure_schema(db)
    now = _now_utc()
    if range_key == "1d":
        start = now - timedelta(days=1)
    elif range_key == "1w":
        start = now - timedelta(weeks=1)
    elif range_key == "1m":
        start = now - timedelta(days=30)
    elif range_key == "1y":
        start = now - timedelta(days=365)
    else:
        raise ValueError("invalid range")
    rows = db.execute(
        """
        SELECT ts, balance FROM scalper_lf_equity
         WHERE ts >= ?
         ORDER BY ts ASC
        """,
        (start.isoformat(),),
    ).fetchall()
    points: List[EquityPoint] = []
    for row in rows:
        ts_value = _row_value(row, "ts", 0, "")
        balance = _row_value(row, "balance", 1, 0.0)
        points.append(
            EquityPoint(
                ts=str(ts_value),
                balance=float(balance) if balance is not None else 0.0,
            )
        )
    return points


async def _fetch_schwab_quote(symbol: str) -> Dict[str, Any]:  # pragma: no cover - network
    from services import schwab_client

    return await schwab_client.get_quote(symbol)


async def _fetch_yfinance_quote(symbol: str) -> Dict[str, Any]:  # pragma: no cover - network
    from services import data_provider

    return await data_provider._fetch_yfinance_quote(symbol)


async def fetch_quote(symbol: str) -> Dict[str, Any]:
    try:
        quote = await _fetch_schwab_quote(symbol)
    except Exception:
        quote = {}
    if quote:
        return quote
    return await _fetch_yfinance_quote(symbol)


def fetch_quote_sync(symbol: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(fetch_quote(symbol), loop).result()
    return asyncio.run(fetch_quote(symbol))


def _rsi(values: Sequence[float], period: int = 14) -> List[float]:
    result: List[float] = []
    gains: List[float] = []
    losses: List[float] = []
    for i, value in enumerate(values):
        if i == 0:
            gains.append(0.0)
            losses.append(0.0)
            result.append(50.0)
            continue
        delta = value - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
        if i < period:
            result.append(50.0)
            continue
        avg_gain = sum(gains[i - period + 1 : i + 1]) / period
        avg_loss = sum(losses[i - period + 1 : i + 1]) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - 100 / (1 + rs))
    return result


def evaluate_exit(
    *,
    entry_price: float,
    bars: Sequence[Mapping[str, Any]],
    target_pct: float,
    stop_pct: float,
    time_cap_minutes: int,
) -> Tuple[float, str, Optional[str]]:
    if not bars:
        return entry_price, "timeout", None
    start_time: Optional[datetime] = None
    for bar in bars:
        ts_raw = bar.get("ts")
        price = float(bar.get("close", entry_price))
        if isinstance(ts_raw, str):
            try:
                ts_val = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                ts_val = None
        else:
            ts_val = None
        if start_time is None and ts_val is not None:
            start_time = ts_val
        move_pct = 0.0 if entry_price == 0 else (price - entry_price) / entry_price * 100.0
        if move_pct >= target_pct:
            return price, "target", ts_val.isoformat() if ts_val else None
        if move_pct <= stop_pct:
            return price, "stop", ts_val.isoformat() if ts_val else None
        if (
            start_time is not None
            and ts_val is not None
            and (ts_val - start_time) >= timedelta(minutes=time_cap_minutes)
        ):
            return price, "timeout", ts_val.isoformat()
    return price, "timeout", None


def run_backtest(
    db,
    *,
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: LFSettings | None = None,
) -> Dict[str, Any]:
    settings = settings or load_settings(db)
    balance = settings.starting_balance
    equity_curve: List[Dict[str, Any]] = []
    for symbol, bars in bars_by_symbol.items():
        closes = [float(bar.get("close", 0.0)) for bar in bars]
        if not closes:
            continue
        rsi_values = _rsi(closes)
        window_high = None
        window_low = None
        window_end = None
        open_trade_price = None
        entry_index = None
        for idx, bar in enumerate(bars):
            ts_raw = bar.get("ts")
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                ts = None
            price = float(bar.get("close", 0.0))
            if window_end is None:
                window_end = idx + 5
            if idx < (window_end or 0):
                high = float(bar.get("high", price))
                low = float(bar.get("low", price))
                window_high = high if window_high is None else max(window_high, high)
                window_low = low if window_low is None else min(window_low, low)
                continue
            vwap = float(bar.get("vwap", price))
            rsi_ok = True
            if settings.rsi_filter and idx < len(rsi_values):
                rsi_ok = rsi_values[idx] >= 50.0
            if open_trade_price is None and price > (window_high or price) and price >= vwap and rsi_ok:
                open_trade_price = price
                entry_index = idx
                continue
            if open_trade_price is not None and entry_index is not None:
                segment = bars[entry_index : idx + 1]
                exit_price, reason, _ = evaluate_exit(
                    entry_price=open_trade_price,
                    bars=segment,
                    target_pct=settings.profit_target_pct,
                    stop_pct=settings.max_adverse_pct,
                    time_cap_minutes=settings.time_cap_minutes,
                )
                pnl = (exit_price - open_trade_price) * 100.0
                balance += pnl
                equity_curve.append(
                    {
                        "ts": ts.isoformat() if ts else None,
                        "balance": balance,
                        "symbol": symbol,
                        "reason": reason,
                        "net": pnl,
                    }
                )
                open_trade_price = None
                entry_index = None
    summary = summarize_backtest(equity_curve, starting_balance=settings.starting_balance)
    return {"equity_curve": equity_curve, "summary": summary}

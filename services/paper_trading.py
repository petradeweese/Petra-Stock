"""Paper trading engine, persistence helpers, and API utilities."""
from __future__ import annotations

import csv
import io
import logging
import math
import os
import sqlite3
import time

import db as db_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Mapping

from utils import now_et

logger = logging.getLogger(__name__)

# Feature flag allowing deployments to disable real fills while testing.
PAPER_DRY_RUN = os.getenv("PAPER_TRADING_DRY_RUN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DEFAULT_STARTING_BALANCE = 100_000.0

LEGACY_MODE = "paper"

_SETTINGS_COLUMNS = {
    "starting_balance",
    "max_pct",
    "pct_equity_per_trade",
    "daily_trade_cap",
    "time_cap_minutes",
    "cooldown_minutes",
    "profit_target_pct",
    "mae_pct",
    "max_open_positions",
    "daily_max_drawdown_pct",
    "volatility_gate",
    "per_order_fee",
    "per_contract_fee",
    "tickers_csv",
    "rsi_filter_enabled",
    "status",
    "started_at",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_settings (
  mode TEXT PRIMARY KEY,
  starting_balance REAL NOT NULL,
  max_pct REAL,
  pct_equity_per_trade REAL,
  daily_trade_cap INTEGER,
  time_cap_minutes INTEGER,
  cooldown_minutes INTEGER,
  profit_target_pct REAL,
  mae_pct REAL,
  max_open_positions INTEGER,
  daily_max_drawdown_pct REAL,
  volatility_gate REAL,
  per_order_fee REAL,
  per_contract_fee REAL,
  tickers_csv TEXT,
  rsi_filter_enabled INTEGER DEFAULT 0,
  status TEXT DEFAULT 'inactive',
  started_at TEXT
);

CREATE TABLE IF NOT EXISTS equity_log (
  id INTEGER PRIMARY KEY,
  mode TEXT NOT NULL,
  ts INTEGER NOT NULL,
  equity REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_log_mode_ts ON equity_log(mode, ts DESC);
"""


@dataclass(slots=True)
class PaperSettings:
    mode: str
    starting_balance: float
    max_pct: float
    status: str
    started_at: str | None
    pct_equity_per_trade: float | None = None
    daily_trade_cap: int | None = None
    time_cap_minutes: int | None = None
    cooldown_minutes: int | None = None
    profit_target_pct: float | None = None
    mae_pct: float | None = None
    max_open_positions: int | None = None
    daily_max_drawdown_pct: float | None = None
    volatility_gate: float | None = None
    per_order_fee: float | None = None
    per_contract_fee: float | None = None
    tickers_csv: str | None = None
    rsi_filter_enabled: bool = False


@dataclass(slots=True)
class PaperTrade:
    id: int
    ticker: str
    call_put: str
    strike: float | None
    expiry: str | None
    qty: int
    interval: str | None
    entry_time: str
    executed_at: str
    entry_price: float
    exit_time: str | None
    exit_price: float | None
    roi_pct: float | None
    status: str
    source_alert_id: str | None
    price_source: str | None


@dataclass(slots=True)
class EquityPoint:
    ts: str
    balance: float


def _row_get(row, key: int | str, idx: int, default=None):
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        pass
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        try:
            return row[idx]
        except Exception:
            return default
    return default


def _now_utc_iso(dt: datetime | None = None) -> str:
    current = dt or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _table_columns(db, table: str) -> list[str]:
    try:
        rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return []
    names: list[str] = []
    for row in rows or []:
        value = _row_get(row, "name", 1, None)
        if value is None:
            continue
        names.append(str(value))
    return names


def _migrate_legacy_paper_settings(db) -> None:
    columns = _table_columns(db, "paper_settings")
    if not columns:
        return
    if "mode" in columns:
        return
    db.executescript(
        """
        ALTER TABLE paper_settings RENAME TO paper_settings_old;
        CREATE TABLE paper_settings (
          mode TEXT PRIMARY KEY,
          starting_balance REAL NOT NULL,
          max_pct REAL,
          pct_equity_per_trade REAL,
          daily_trade_cap INTEGER,
          time_cap_minutes INTEGER,
          cooldown_minutes INTEGER,
          profit_target_pct REAL,
          mae_pct REAL,
          max_open_positions INTEGER,
          daily_max_drawdown_pct REAL,
          volatility_gate REAL,
          per_order_fee REAL,
          per_contract_fee REAL,
          tickers_csv TEXT,
          rsi_filter_enabled INTEGER DEFAULT 0,
          status TEXT DEFAULT 'inactive',
          started_at TEXT
        );
        INSERT INTO paper_settings(mode, starting_balance, max_pct, status, started_at)
            SELECT 'paper', starting_balance, max_pct, status, started_at FROM paper_settings_old WHERE id = 1;
        DROP TABLE paper_settings_old;
        """
    )


def _has_equity_log_mode(db) -> bool:
    columns = _table_columns(db, "equity_log")
    return "mode" in columns if columns else False


def migrate_equity_log_mode(db) -> None:
    if _has_equity_log_mode(db):
        return
    columns = _table_columns(db, "equity_log")
    if not columns:
        return
    db.executescript(
        """
        ALTER TABLE equity_log RENAME TO equity_log_old;
        CREATE TABLE equity_log (
          id INTEGER PRIMARY KEY,
          mode TEXT NOT NULL,
          ts INTEGER NOT NULL,
          equity REAL NOT NULL
        );
        INSERT INTO equity_log(mode, ts, equity)
          SELECT 'lf' AS mode, ts, equity FROM equity_log_old;
        DROP TABLE equity_log_old;
        CREATE INDEX IF NOT EXISTS idx_equity_log_mode_ts ON equity_log(mode, ts DESC);
        """
    )


def _ensure_mode_rows(db) -> None:
    defaults = {
        LEGACY_MODE: {"starting_balance": 10_000.0, "max_pct": 10.0, "status": "inactive"},
        "hf": {"starting_balance": DEFAULT_STARTING_BALANCE, "status": "inactive"},
        "lf": {"starting_balance": DEFAULT_STARTING_BALANCE, "status": "inactive"},
    }
    for mode, payload in defaults.items():
        row = db.execute("SELECT 1 FROM paper_settings WHERE mode=?", (mode,)).fetchone()
        if row:
            continue
        db.execute(
            "INSERT INTO paper_settings(mode, starting_balance, max_pct, status) VALUES(?,?,?,?)",
            (
                mode,
                float(payload.get("starting_balance", DEFAULT_STARTING_BALANCE)),
                float(payload.get("max_pct", payload.get("pct_equity_per_trade", 0.0) or 0.0)),
                str(payload.get("status", "inactive")),
            ),
        )


def ensure_settings(db) -> None:
    conn = getattr(db, "connection", None)
    if conn is None:
        raise RuntimeError("paper_trading.ensure_settings requires a DB connection")

    if getattr(conn, "in_transaction", False):
        _migrate_legacy_paper_settings(db)
        migrate_equity_log_mode(db)
        db.executescript(SCHEMA)
        _ensure_mode_rows(db)
        return

    def _apply() -> None:
        db.execute("BEGIN IMMEDIATE")
        try:
            _migrate_legacy_paper_settings(db)
            migrate_equity_log_mode(db)
            db.executescript(SCHEMA)
            _ensure_mode_rows(db)
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    db_module.retry_locked(_apply)


def _settings_defaults_for_mode(mode: str) -> dict[str, object]:
    mode_key = (mode or "").lower()
    if mode_key == LEGACY_MODE:
        return {"starting_balance": 10_000.0, "max_pct": 10.0, "status": "inactive"}
    return {"starting_balance": DEFAULT_STARTING_BALANCE, "status": "inactive"}


def _row_to_settings(mode: str, row) -> PaperSettings:
    defaults = _settings_defaults_for_mode(mode)
    start_default = float(defaults.get("starting_balance", DEFAULT_STARTING_BALANCE))
    max_pct_default = float(defaults.get("max_pct", defaults.get("pct_equity_per_trade", 0.0) or 0.0))
    pct_trade = _row_get(row, "pct_equity_per_trade", 3, None)
    max_pct_value = _row_get(row, "max_pct", 2, None)
    max_pct_final = max_pct_value if max_pct_value is not None else pct_trade
    def _as_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _as_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return PaperSettings(
        mode=mode,
        starting_balance=float(_row_get(row, "starting_balance", 1, start_default) or start_default),
        max_pct=float(max_pct_final or max_pct_default or 0.0),
        status=str(_row_get(row, "status", 16, defaults.get("status", "inactive")) or defaults.get("status", "inactive")),
        started_at=_row_get(row, "started_at", 17),
        pct_equity_per_trade=_as_float(pct_trade),
        daily_trade_cap=_as_int(_row_get(row, "daily_trade_cap", 4)),
        time_cap_minutes=_as_int(_row_get(row, "time_cap_minutes", 5)),
        cooldown_minutes=_as_int(_row_get(row, "cooldown_minutes", 6)),
        profit_target_pct=_as_float(_row_get(row, "profit_target_pct", 7)),
        mae_pct=_as_float(_row_get(row, "mae_pct", 8)),
        max_open_positions=_as_int(_row_get(row, "max_open_positions", 9)),
        daily_max_drawdown_pct=_as_float(_row_get(row, "daily_max_drawdown_pct", 10)),
        volatility_gate=_as_float(_row_get(row, "volatility_gate", 11)),
        per_order_fee=_as_float(_row_get(row, "per_order_fee", 12)),
        per_contract_fee=_as_float(_row_get(row, "per_contract_fee", 13)),
        tickers_csv=_row_get(row, "tickers_csv", 14),
        rsi_filter_enabled=bool(int(_row_get(row, "rsi_filter_enabled", 15, 0) or 0)),
    )


def load_settings(db, mode: str) -> PaperSettings:
    ensure_settings(db)
    mode_key = (mode or LEGACY_MODE).lower()
    row = db.execute("SELECT * FROM paper_settings WHERE mode=?", (mode_key,)).fetchone()
    if not row:
        defaults = _settings_defaults_for_mode(mode_key)
        db.execute(
            "INSERT OR REPLACE INTO paper_settings(mode, starting_balance, max_pct, status) VALUES(?,?,?,?)",
            (
                mode_key,
                float(defaults.get("starting_balance", DEFAULT_STARTING_BALANCE)),
                float(defaults.get("max_pct", defaults.get("pct_equity_per_trade", 0.0) or 0.0)),
                str(defaults.get("status", "inactive")),
            ),
        )
        row = db.execute("SELECT * FROM paper_settings WHERE mode=?", (mode_key,)).fetchone()
    return _row_to_settings(mode_key, row)


def upsert_settings(db, mode: str, payload: dict[str, object]) -> None:
    ensure_settings(db)
    mode_key = (mode or LEGACY_MODE).lower()
    filtered = {k: v for k, v in payload.items() if k in _SETTINGS_COLUMNS}
    if not filtered:
        return
    columns: list[str] = []
    values: list[object] = []
    for key, value in filtered.items():
        columns.append(key)
        values.append(value)
    placeholders = ", ".join(["?"] * len(columns))
    assignments = ", ".join([f"{col}=excluded.{col}" for col in columns])
    sql = (
        f"INSERT INTO paper_settings(mode, {', '.join(columns)}) VALUES(?, {placeholders}) "
        f"ON CONFLICT(mode) DO UPDATE SET {assignments}"
    )
    db.execute(sql, [mode_key, *values])


def get_latest_equity(db, mode: str) -> float | None:
    ensure_settings(db)
    row = db.execute(
        "SELECT equity FROM equity_log WHERE mode=? ORDER BY ts DESC LIMIT 1",
        ((mode or LEGACY_MODE).lower(),),
    ).fetchone()
    return float(_row_get(row, "equity", 0)) if row else None


def seed_equity_if_empty(db, mode: str) -> None:
    ensure_settings(db)
    mode_key = (mode or LEGACY_MODE).lower()
    latest = get_latest_equity(db, mode_key)
    if latest is not None:
        return
    settings = load_settings(db, mode_key)
    now_ms = int(time.time() * 1000)
    db.execute(
        "INSERT INTO equity_log(mode, ts, equity) VALUES(?,?,?)",
        (mode_key, now_ms, float(settings.starting_balance)),
    )


def append_equity_point(db, mode: str, equity: float, *, ts: int | None = None) -> None:
    ensure_settings(db)
    mode_key = (mode or LEGACY_MODE).lower()
    timestamp = int(ts if ts is not None else time.time() * 1000)
    db.execute(
        "INSERT INTO equity_log(mode, ts, equity) VALUES(?,?,?)",
        (mode_key, timestamp, float(equity)),
    )


def update_settings(db, mode: str, *, starting_balance: float, max_pct: float) -> None:
    ensure_settings(db)
    mode_key = (mode or LEGACY_MODE).lower()
    start_value = max(0.0, float(starting_balance))
    max_pct_value = max(0.0, min(float(max_pct), 100.0))
    db.execute(
        """
        INSERT INTO paper_settings(mode, starting_balance, max_pct)
        VALUES(?, ?, ?)
        ON CONFLICT(mode) DO UPDATE SET starting_balance=excluded.starting_balance, max_pct=excluded.max_pct
        """,
        (mode_key, start_value, max_pct_value),
    )
    conn = getattr(db, "connection", None)
    if conn is not None:
        conn.commit()


def _seed_equity(db, settings: PaperSettings, *, now: datetime | None = None) -> None:
    row = db.execute("SELECT COUNT(1) FROM paper_equity").fetchone()
    count = int(_row_get(row, 0, 0, 0) or 0) if row else 0
    if count == 0:
        ts = _now_utc_iso(now)
        db.execute(
            "INSERT OR REPLACE INTO paper_equity(ts, balance) VALUES(?, ?)",
            (ts, float(settings.starting_balance)),
        )
        db.connection.commit()


def _latest_balance(db, settings: PaperSettings | None = None) -> float:
    row = db.execute(
        "SELECT balance FROM paper_equity ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if row:
        return float(_row_get(row, "balance", 0, 0.0) or 0.0)
    if settings is None:
        settings = load_settings(db, LEGACY_MODE)
    return float(settings.starting_balance)


_CAP_EPSILON = 1e-6


def calculate_max_contracts(balance: float, max_pct: float, option_price: float) -> tuple[int, float]:
    pct = max(0.0, float(max_pct)) / 100.0
    cap = max(0.0, float(balance)) * pct
    cost_per_contract = max(0.0, float(option_price)) * 100.0
    if cost_per_contract <= 0 or cap <= 0:
        return 0, cap
    raw_qty = cap / cost_per_contract
    qty = int(math.floor(raw_qty + 1e-9))
    while qty > 0 and (qty * cost_per_contract) - cap > _CAP_EPSILON:
        qty -= 1
    if qty < 0:
        qty = 0
    return qty, cap


def calculate_roi(entry_price: float, exit_price: float) -> float:
    entry = float(entry_price)
    exit_val = float(exit_price)
    if entry == 0:
        return 0.0
    return (exit_val - entry) / entry * 100.0


def resolve_fill_price(
    mid_price: float | None,
    recent_mid: float | None,
    underlying_move: float | None,
    delta: float | None,
) -> tuple[float, str]:
    if mid_price is not None and mid_price > 0:
        return float(mid_price), "mid"
    if recent_mid is not None and recent_mid > 0:
        return float(recent_mid), "recent_mid"
    if underlying_move is not None and delta is not None:
        magnitude = abs(float(underlying_move)) * max(abs(float(delta)), 0.01)
        return max(0.01, round(magnitude, 2)), "synthetic_delta"
    raise ValueError("Unable to determine option price for paper trade")


def dedupe_key(ticker: str, call_put: str, interval: str | None) -> str:
    parts = [str(ticker or "").upper(), str(call_put or "").upper()]
    if interval:
        parts.append(str(interval).lower())
    return "|".join(parts)


def _format_strike_exp_csv(strike: float | None, expiry: str | None) -> str:
    if strike is None and not expiry:
        return "—"
    strike_text = "—"
    if strike is not None:
        strike_text = f"${strike:,.2f}".rstrip("0").rstrip(".")
    expiry_text = "—"
    if expiry:
        try:
            parsed = datetime.fromisoformat(str(expiry))
        except ValueError:
            expiry_text = str(expiry)
        else:
            expiry_text = parsed.strftime("%Y-%m-%d")
    return f"{strike_text} — {expiry_text}"


def _has_open_position(db, ticker: str, call_put: str, interval: str | None) -> bool:
    ticker_key = ticker.upper()
    direction = call_put.upper()
    if interval:
        row = db.execute(
            """
            SELECT id FROM paper_trades
             WHERE status='open' AND ticker=? AND call_put=? AND interval=?
             LIMIT 1
            """,
            (ticker_key, direction, interval.lower()),
        ).fetchone()
    else:
        row = db.execute(
            """
            SELECT id FROM paper_trades
             WHERE status='open' AND ticker=? AND call_put=? AND interval IS NULL
             LIMIT 1
            """,
            (ticker_key, direction),
        ).fetchone()
    return bool(row)


def _has_seen_alert(db, key: str | None) -> bool:
    if not key:
        return False
    row = db.execute(
        "SELECT id FROM paper_trades WHERE source_alert_id=? LIMIT 1",
        (key,),
    ).fetchone()
    return bool(row)


def _append_equity(db, balance: float, *, ts: datetime | None = None) -> None:
    stamp = _now_utc_iso(ts)
    db.execute(
        "INSERT OR REPLACE INTO paper_equity(ts, balance) VALUES(?, ?)",
        (stamp, float(balance)),
    )
    db.connection.commit()
    logger.info(
        "paper_equity_point",
        extra={"balance": float(balance), "ts": stamp},
    )


def open_position(
    db,
    *,
    ticker: str,
    call_put: str,
    strike: float | None,
    expiry: str | None,
    interval: str | None,
    mid_price: float | None,
    recent_mid: float | None,
    underlying_move: float | None,
    delta: float | None,
    source_alert_id: str | None = None,
    now: datetime | None = None,
) -> Optional[int]:
    settings = load_settings(db, LEGACY_MODE)
    _seed_equity(db, settings, now=now)
    if settings.status.lower() != "active":
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "inactive", "key": source_alert_id},
        )
        return None
    interval_norm = interval.lower() if interval else None
    alert_key = (source_alert_id or "").strip() or None
    dedupe_token = dedupe_key(ticker, call_put, interval)
    if _has_seen_alert(db, alert_key):
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "duplicate_alert", "key": alert_key},
        )
        return None
    if _has_open_position(db, ticker, call_put, interval_norm):
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "duplicate_open", "key": dedupe_token},
        )
        return None
    if PAPER_DRY_RUN:
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "dry_run", "key": dedupe_token},
        )
        return None
    try:
        price, price_source = resolve_fill_price(mid_price, recent_mid, underlying_move, delta)
    except ValueError:
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "no_price", "key": dedupe_token},
        )
        return None
    balance = _latest_balance(db, settings)
    qty, cap = calculate_max_contracts(balance, settings.max_pct, price)
    if qty <= 0:
        logger.info(
            "paper_trade_skipped",
            extra={
                "symbol": ticker,
                "reason": "insufficient_capital",
                "cap": cap,
                "price": price,
            },
        )
        return None
    cost = price * qty * 100.0
    if cost - cap > _CAP_EPSILON:
        max_affordable = int(math.floor((cap + _CAP_EPSILON) / (price * 100.0)))
        while max_affordable > 0 and (price * max_affordable * 100.0) - cap > _CAP_EPSILON:
            max_affordable -= 1
        if max_affordable <= 0:
            logger.info(
                "paper_trade_skipped",
                extra={
                    "symbol": ticker,
                    "reason": "insufficient_capital",
                    "cap": cap,
                    "price": price,
                },
            )
            return None
        qty = max_affordable
        cost = price * qty * 100.0
    entry_ts = _now_utc_iso(now or now_et())
    executed_at = entry_ts
    try:
        db.execute(
            """
            INSERT INTO paper_trades(
                ticker, call_put, strike, expiry, qty, interval, entry_time, executed_at, entry_price, status, source_alert_id, price_source
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticker.upper(),
                call_put.upper(),
                strike,
                expiry,
                qty,
                interval_norm,
                entry_ts,
                executed_at,
                price,
                "open",
                alert_key,
                price_source,
            ),
        )
    except sqlite3.IntegrityError:
        logger.info(
            "paper_trade_skipped",
            extra={"symbol": ticker, "reason": "duplicate_alert", "key": alert_key or dedupe_token},
        )
        return None
    except Exception as exc:
        if "unique" in str(exc).lower():
            logger.info(
                "paper_trade_skipped",
                extra={"symbol": ticker, "reason": "duplicate_alert", "key": alert_key or dedupe_token},
            )
            return None
        raise
    trade_id = int(db.lastrowid)
    new_balance = balance - cost
    _append_equity(db, new_balance, ts=now)
    logger.info(
        "paper_trade_opened",
        extra={
            "symbol": ticker.upper(),
            "qty": qty,
            "price": price,
            "strike": strike,
            "expiry": expiry,
            "call_put": call_put.upper(),
            "trade_id": trade_id,
            "price_source": price_source,
            "interval": interval_norm,
        },
    )
    return trade_id


def close_position(
    db,
    trade_id: int,
    *,
    exit_price: float,
    reason: str,
    now: datetime | None = None,
) -> Optional[PaperTrade]:
    row = db.execute(
        "SELECT * FROM paper_trades WHERE id=? LIMIT 1",
        (int(trade_id),),
    ).fetchone()
    status = str(_row_get(row, "status", 13, "")).lower() if row else ""
    if not row or status != "open":
        return None
    settings = load_settings(db, LEGACY_MODE)
    balance = _latest_balance(db, settings)
    qty = int(_row_get(row, "qty", 5, 0))
    entry_price = float(_row_get(row, "entry_price", 9, 0.0))
    roi = calculate_roi(entry_price, exit_price)
    exit_ts = _now_utc_iso(now or now_et())
    db.execute(
        """
        UPDATE paper_trades
           SET status='closed', exit_time=?, exit_price=?, roi_pct=?
         WHERE id=?
        """,
        (exit_ts, float(exit_price), roi, int(trade_id)),
    )
    proceeds = float(exit_price) * qty * 100.0
    new_balance = balance + proceeds
    _append_equity(db, new_balance, ts=now)
    logger.info(
        "paper_trade_closed",
        extra={
            "symbol": _row_get(row, "ticker", 1),
            "qty": qty,
            "price": exit_price,
            "roi": roi,
            "reason": reason,
            "trade_id": trade_id,
        },
    )
    return PaperTrade(
        id=int(trade_id),
        ticker=_row_get(row, "ticker", 1),
        call_put=_row_get(row, "call_put", 2),
        strike=_row_get(row, "strike", 3),
        expiry=_row_get(row, "expiry", 4),
        qty=qty,
        interval=_row_get(row, "interval", 6),
        entry_time=_row_get(row, "entry_time", 7),
        executed_at=_row_get(row, "executed_at", 8),
        entry_price=entry_price,
        exit_time=exit_ts,
        exit_price=float(exit_price),
        roi_pct=roi,
        status="closed",
        source_alert_id=_row_get(row, "source_alert_id", 14),
        price_source=_row_get(row, "price_source", 15),
    )


def start_engine(db, *, now: datetime | None = None) -> PaperSettings:
    settings = load_settings(db, LEGACY_MODE)
    if settings.status.lower() == "active":
        _seed_equity(db, settings, now=now)
        return settings
    started_at = _now_utc_iso(now or now_et())
    db.execute(
        "UPDATE paper_settings SET status='active', started_at=? WHERE mode=?",
        (started_at, LEGACY_MODE),
    )
    db.connection.commit()
    _seed_equity(db, settings, now=now)
    return load_settings(db, LEGACY_MODE)


def stop_engine(db) -> PaperSettings:
    db.execute(
        "UPDATE paper_settings SET status='inactive' WHERE mode=?",
        (LEGACY_MODE,),
    )
    db.connection.commit()
    return load_settings(db, LEGACY_MODE)


def restart_engine(db, *, now: datetime | None = None) -> PaperSettings:
    settings = load_settings(db, LEGACY_MODE)
    db.execute("DELETE FROM paper_trades")
    db.execute("DELETE FROM paper_equity")
    db.connection.commit()
    _seed_equity(db, settings, now=now)
    return start_engine(db, now=now)


def get_summary(db) -> dict:
    ensure_settings(db)
    legacy_settings = load_settings(db, LEGACY_MODE)
    _seed_equity(db, legacy_settings)
    balance = _latest_balance(db, legacy_settings)
    starting = float(legacy_settings.starting_balance) or 0.0
    roi = calculate_roi(starting or 1.0, balance) if starting else 0.0
    open_row = db.execute(
        "SELECT COUNT(1) FROM paper_trades WHERE status='open'"
    ).fetchone()
    open_count = int(_row_get(open_row, 0, 0, 0) or 0) if open_row else 0
    summary: dict[str, dict[str, float | int | str | None]] = {
        "legacy": {
            "balance": balance,
            "starting_balance": starting,
            "roi_pct": roi,
            "status": legacy_settings.status,
            "started_at": legacy_settings.started_at,
            "open_trades": int(open_count),
        }
    }
    for mode in ("hf", "lf"):
        seed_equity_if_empty(db, mode)
        latest = get_latest_equity(db, mode)
        settings = load_settings(db, mode)
        starting_balance = float(settings.starting_balance or 0.0)
        equity_value = float(latest if latest is not None else starting_balance)
        roi_value = calculate_roi(starting_balance or 1.0, equity_value) if starting_balance else 0.0
        summary[mode] = {
            "starting_balance": starting_balance,
            "account_equity": equity_value,
            "roi_pct": roi_value,
        }
    summary.update(
        {
            "balance": balance,
            "starting_balance": starting,
            "roi_pct": roi,
            "status": legacy_settings.status,
            "started_at": legacy_settings.started_at,
            "open_trades": int(open_count),
        }
    )
    return summary


_RANGE_LOOKUPS = {
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
    "1m": timedelta(days=30),
    "1y": timedelta(days=365),
}


def get_equity_points(db, range_key: str) -> List[EquityPoint]:
    settings = load_settings(db, LEGACY_MODE)
    _seed_equity(db, settings)
    range_key = (range_key or "1m").lower()
    delta = _RANGE_LOOKUPS.get(range_key, _RANGE_LOOKUPS["1m"])
    start_ts = datetime.now(timezone.utc) - delta
    rows = db.execute(
        "SELECT ts, balance FROM paper_equity WHERE ts >= ? ORDER BY ts",
        (_now_utc_iso(start_ts),),
    ).fetchall()
    return [
        EquityPoint(
            ts=_row_get(row, "ts", 0),
            balance=float(_row_get(row, "balance", 1, 0.0)),
        )
        for row in rows
    ]


def _query_trades(db, status: str | None) -> Sequence[PaperTrade]:
    params: list = []
    sql = "SELECT * FROM paper_trades"
    if status and status.lower() in {"open", "closed"}:
        sql += " WHERE status=?"
        params.append(status.lower())
    sql += " ORDER BY entry_time DESC"
    rows = db.execute(sql, tuple(params)).fetchall()
    return [
        PaperTrade(
            id=int(_row_get(row, "id", 0, 0)),
            ticker=_row_get(row, "ticker", 1),
            call_put=_row_get(row, "call_put", 2),
            strike=_row_get(row, "strike", 3),
            expiry=_row_get(row, "expiry", 4),
            qty=int(_row_get(row, "qty", 5, 0)),
            interval=_row_get(row, "interval", 6),
            entry_time=_row_get(row, "entry_time", 7),
            executed_at=_row_get(row, "executed_at", 8),
            entry_price=float(_row_get(row, "entry_price", 9, 0.0)),
            exit_time=_row_get(row, "exit_time", 10),
            exit_price=float(_row_get(row, "exit_price", 11))
            if _row_get(row, "exit_price", 11) is not None
            else None,
            roi_pct=float(_row_get(row, "roi_pct", 12))
            if _row_get(row, "roi_pct", 12) is not None
            else None,
            status=_row_get(row, "status", 13),
            source_alert_id=_row_get(row, "source_alert_id", 14),
            price_source=_row_get(row, "price_source", 15),
        )
        for row in rows
    ]


def list_trades(db, status: str | None) -> List[dict]:
    trades = _query_trades(db, status)
    results: list[dict] = []
    for trade in trades:
        results.append(
            {
                "id": trade.id,
                "ticker": trade.ticker,
                "call_put": trade.call_put,
                "strike": trade.strike,
                "expiry": trade.expiry,
                "qty": trade.qty,
                "interval": trade.interval,
                "entry_time": trade.entry_time,
                "executed_at": trade.executed_at,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "roi_pct": trade.roi_pct,
                "status": trade.status,
                "price_source": trade.price_source,
            }
        )
    return results


def export_trades_csv(db, status: str | None) -> tuple[str, str]:
    trades = _query_trades(db, status)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
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
    )
    for trade in trades:
        writer.writerow(
            [
                trade.ticker,
                trade.call_put,
                _format_strike_exp_csv(trade.strike, trade.expiry),
                trade.qty,
                trade.entry_time,
                f"{trade.entry_price:.2f}",
                trade.exit_time or "",
                f"{trade.exit_price:.2f}" if trade.exit_price is not None else "",
                f"{trade.roi_pct:.2f}" if trade.roi_pct is not None else "",
                trade.status,
            ]
        )
    return buffer.getvalue(), "paper_trades.csv"


__all__ = [
    "PaperSettings",
    "PaperTrade",
    "EquityPoint",
    "DEFAULT_STARTING_BALANCE",
    "LEGACY_MODE",
    "calculate_max_contracts",
    "calculate_roi",
    "resolve_fill_price",
    "dedupe_key",
    "open_position",
    "close_position",
    "start_engine",
    "stop_engine",
    "restart_engine",
    "get_summary",
    "get_equity_points",
    "list_trades",
    "export_trades_csv",
    "update_settings",
    "load_settings",
    "upsert_settings",
    "get_latest_equity",
    "seed_equity_if_empty",
    "append_equity_point",
]

"""Minimal persistence layer for the Favorites paper simulator.

This module does not implement the full trading engine but provides
infrastructure for storing simulator configuration and surfacing
status/activity data through the API. It mirrors the structure used by the
scalper paper trading services so that a future engine can plug in without
modifying the routes layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FavoritesSimSettings:
    starting_balance: float = 100000.0
    allocation_mode: str = "percent"  # percent or fixed
    allocation_value: float = 10.0
    per_contract_fee: float = 0.0
    per_order_fee: float = 0.0
    slippage_bps: float = 0.0
    daily_trade_cap: int = 20
    allow_premarket: bool = False
    allow_postmarket: bool = False
    entry_rule: str = "next_open"
    exit_time_cap_minutes: int = 30
    exit_profit_target_pct: float | None = None
    exit_max_adverse_pct: float | None = None


@dataclass(slots=True)
class FavoritesSimStatus:
    status: str
    started_at: str | None


_RANGE_TO_DAYS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _ensure_tables(db) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_favorites_settings (
            id INTEGER PRIMARY KEY CHECK (id=1),
            starting_balance REAL NOT NULL DEFAULT 100000,
            allocation_mode TEXT NOT NULL DEFAULT 'percent',
            allocation_value REAL NOT NULL DEFAULT 10,
            per_contract_fee REAL NOT NULL DEFAULT 0,
            per_order_fee REAL NOT NULL DEFAULT 0,
            slippage_bps REAL NOT NULL DEFAULT 0,
            daily_trade_cap INTEGER NOT NULL DEFAULT 20,
            allow_premarket INTEGER NOT NULL DEFAULT 0,
            allow_postmarket INTEGER NOT NULL DEFAULT 0,
            entry_rule TEXT NOT NULL DEFAULT 'next_open',
            exit_time_cap_minutes INTEGER NOT NULL DEFAULT 30,
            exit_profit_target_pct REAL,
            exit_max_adverse_pct REAL
        )
        """
    )
    db.execute(
        """
        INSERT OR IGNORE INTO paper_favorites_settings(
            id, starting_balance, allocation_mode, allocation_value,
            per_contract_fee, per_order_fee, slippage_bps, daily_trade_cap,
            allow_premarket, allow_postmarket, entry_rule, exit_time_cap_minutes,
            exit_profit_target_pct, exit_max_adverse_pct
        ) VALUES(1, 100000, 'percent', 10, 0, 0, 0, 20, 0, 0, 'next_open', 30, NULL, NULL)
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_favorites_status (
            id INTEGER PRIMARY KEY CHECK (id=1),
            status TEXT NOT NULL DEFAULT 'inactive',
            started_at TEXT
        )
        """
    )
    db.execute(
        """
        INSERT OR IGNORE INTO paper_favorites_status(id, status, started_at)
        VALUES(1, 'inactive', NULL)
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_favorites_equity (
            ts TEXT PRIMARY KEY,
            balance REAL NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_favorites_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            pnl REAL,
            fees REAL,
            metadata TEXT
        )
        """
    )
    db.connection.commit()


def _seed_equity(db, settings: FavoritesSimSettings) -> None:
    row = db.execute("SELECT COUNT(1) FROM paper_favorites_equity").fetchone()
    count = int(row[0]) if row else 0
    if count == 0:
        db.execute(
            "INSERT INTO paper_favorites_equity(ts, balance) VALUES(?, ?)",
            (_now_iso(), float(settings.starting_balance)),
        )
        db.connection.commit()


def load_settings(db) -> FavoritesSimSettings:
    _ensure_tables(db)
    row = db.execute(
        """
        SELECT starting_balance, allocation_mode, allocation_value, per_contract_fee,
               per_order_fee, slippage_bps, daily_trade_cap, allow_premarket,
               allow_postmarket, entry_rule, exit_time_cap_minutes,
               exit_profit_target_pct, exit_max_adverse_pct
        FROM paper_favorites_settings WHERE id=1
        """
    ).fetchone()
    if not row:
        return FavoritesSimSettings()
    return FavoritesSimSettings(
        starting_balance=float(row[0] or 0.0),
        allocation_mode=str(row[1] or "percent"),
        allocation_value=float(row[2] or 0.0),
        per_contract_fee=float(row[3] or 0.0),
        per_order_fee=float(row[4] or 0.0),
        slippage_bps=float(row[5] or 0.0),
        daily_trade_cap=int(row[6] or 0),
        allow_premarket=bool(row[7]),
        allow_postmarket=bool(row[8]),
        entry_rule=str(row[9] or "next_open"),
        exit_time_cap_minutes=int(row[10] or 0),
        exit_profit_target_pct=float(row[11]) if row[11] is not None else None,
        exit_max_adverse_pct=float(row[12]) if row[12] is not None else None,
    )


def update_settings(
    db,
    *,
    starting_balance: float,
    allocation_mode: str,
    allocation_value: float,
    per_contract_fee: float,
    per_order_fee: float,
    slippage_bps: float,
    daily_trade_cap: int,
    allow_premarket: bool,
    allow_postmarket: bool,
    entry_rule: str,
    exit_time_cap_minutes: int,
    exit_profit_target_pct: float | None,
    exit_max_adverse_pct: float | None,
) -> None:
    _ensure_tables(db)
    mode = allocation_mode if allocation_mode in {"percent", "fixed"} else "percent"
    db.execute(
        """
        UPDATE paper_favorites_settings
        SET starting_balance=?, allocation_mode=?, allocation_value=?,
            per_contract_fee=?, per_order_fee=?, slippage_bps=?, daily_trade_cap=?,
            allow_premarket=?, allow_postmarket=?, entry_rule=?, exit_time_cap_minutes=?,
            exit_profit_target_pct=?, exit_max_adverse_pct=?
        WHERE id=1
        """,
        (
            float(max(0.0, starting_balance)),
            mode,
            float(max(0.0, allocation_value)),
            float(max(0.0, per_contract_fee)),
            float(max(0.0, per_order_fee)),
            float(slippage_bps),
            int(max(0, daily_trade_cap)),
            1 if allow_premarket else 0,
            1 if allow_postmarket else 0,
            entry_rule if entry_rule in {"next_open", "signal_close"} else "next_open",
            int(max(0, exit_time_cap_minutes)),
            float(exit_profit_target_pct) if exit_profit_target_pct is not None else None,
            float(exit_max_adverse_pct) if exit_max_adverse_pct is not None else None,
        ),
    )
    db.connection.commit()
    settings = load_settings(db)
    _seed_equity(db, settings)


def _status_row(db) -> FavoritesSimStatus:
    _ensure_tables(db)
    row = db.execute(
        "SELECT status, started_at FROM paper_favorites_status WHERE id=1"
    ).fetchone()
    if not row:
        return FavoritesSimStatus("inactive", None)
    status = str(row[0] or "inactive")
    started = row[1] if row[1] else None
    return FavoritesSimStatus(status, started)


def status_payload(db) -> dict[str, Any]:
    status = _status_row(db)
    return {"status": status.status, "started_at": status.started_at}


def start(db) -> dict[str, Any]:
    _ensure_tables(db)
    now = _now_iso()
    db.execute(
        "UPDATE paper_favorites_status SET status='active', started_at=? WHERE id=1",
        (now,),
    )
    db.connection.commit()
    logger.info("favorites_sim_started", extra={"started_at": now})
    return status_payload(db)


def stop(db) -> dict[str, Any]:
    _ensure_tables(db)
    db.execute(
        "UPDATE paper_favorites_status SET status='inactive', started_at=NULL WHERE id=1"
    )
    db.connection.commit()
    logger.info("favorites_sim_stopped")
    return status_payload(db)


def restart(db) -> dict[str, Any]:
    stop(db)
    return start(db)


def favorites_count(db) -> int:
    try:
        row = db.execute("SELECT COUNT(1) FROM favorites").fetchone()
    except Exception:
        logger.exception("favorites_count_failed")
        return 0
    if not row:
        return 0
    return int(row[0] or 0)


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _equity_rows(db) -> list[tuple[str, float]]:
    _ensure_tables(db)
    rows = db.execute(
        "SELECT ts, balance FROM paper_favorites_equity ORDER BY ts ASC"
    ).fetchall()
    settings = load_settings(db)
    if not rows:
        _seed_equity(db, settings)
        rows = db.execute(
            "SELECT ts, balance FROM paper_favorites_equity ORDER BY ts ASC"
        ).fetchall()
    return [(str(ts), float(balance)) for ts, balance in rows]


def get_equity_points(db, range_key: str) -> list[dict[str, Any]]:
    rows = _equity_rows(db)
    if not rows:
        return []
    if range_key and range_key != "all":
        days = _RANGE_TO_DAYS.get(range_key)
        if days:
            cutoff = _now_utc() - timedelta(days=days)
            filtered: list[tuple[str, float]] = []
            for ts, balance in rows:
                parsed = _parse_iso(ts)
                if parsed is None or parsed >= cutoff:
                    filtered.append((ts, balance))
            if filtered:
                rows = filtered
    return [{"ts": ts, "balance": bal} for ts, bal in rows]


def _current_balance(rows: Iterable[tuple[str, float]]) -> float:
    last = None
    for last in rows:
        pass
    if last is None:
        return 0.0
    return float(last[1])


def summary_payload(db) -> dict[str, Any]:
    settings = load_settings(db)
    rows = _equity_rows(db)
    balance = _current_balance(rows)
    status = _status_row(db)
    pnl = balance - settings.starting_balance
    pnl_pct = 0.0
    if settings.starting_balance:
        pnl_pct = (pnl / settings.starting_balance) * 100.0
    wins = 0
    losses = 0
    total_trades = 0
    avg_win = 0.0
    avg_loss = 0.0
    max_drawdown = 0.0
    try:
        trades = list_activity(db)
    except Exception:
        trades = []
    for trade in trades:
        roi = trade.get("roi_pct")
        if roi is None:
            continue
        total_trades += 1
        if roi > 0:
            wins += 1
        elif roi < 0:
            losses += 1
    if wins:
        win_pnls = [t.get("pnl") for t in trades if (t.get("roi_pct") or 0) > 0]
        win_pnls = [float(v) for v in win_pnls if v is not None]
        if win_pnls:
            avg_win = sum(win_pnls) / len(win_pnls)
    if losses:
        loss_pnls = [t.get("pnl") for t in trades if (t.get("roi_pct") or 0) < 0]
        loss_pnls = [float(v) for v in loss_pnls if v is not None]
        if loss_pnls:
            avg_loss = sum(loss_pnls) / len(loss_pnls)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    universe = favorites_count(db)
    return {
        "status": status.status,
        "started_at": status.started_at,
        "balance": balance,
        "starting_balance": settings.starting_balance,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "universe": universe,
    }


def list_activity(db, limit: int | None = None) -> list[dict[str, Any]]:
    _ensure_tables(db)
    query = "SELECT ts, symbol, side, quantity, price, pnl, fees, metadata FROM paper_favorites_activity ORDER BY ts DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = db.execute(query).fetchall()
    payload: list[dict[str, Any]] = []
    for ts, symbol, side, quantity, price, pnl, fees, metadata in rows:
        meta: dict[str, Any] = {}
        if metadata:
            try:
                meta = json.loads(metadata)
            except (TypeError, json.JSONDecodeError):
                meta = {"raw": metadata}
        payload.append(
            {
                "ts": ts,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "fees": fees,
                "metadata": meta,
                "entry_time": meta.get("entry_time") if isinstance(meta, dict) else None,
                "exit_time": meta.get("exit_time") if isinstance(meta, dict) else None,
                "entry_price": meta.get("entry_price") if isinstance(meta, dict) else None,
                "exit_price": meta.get("exit_price") if isinstance(meta, dict) else None,
                "roi_pct": meta.get("roi_pct") if isinstance(meta, dict) else None,
                "status": meta.get("status") if isinstance(meta, dict) else None,
            }
        )
    return payload


def activity_payload(db) -> dict[str, Any]:
    return {"rows": list_activity(db)}

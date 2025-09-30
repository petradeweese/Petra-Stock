"""Low-frequency paper scalper engine.

The implementation is intentionally self-contained and focuses on deterministic
behaviour so it can be exercised in unit tests without external market data.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
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
    assert row is not None
    return LFSettings(
        starting_balance=float(row["starting_balance"]),
        pct_per_trade=float(row["pct_per_trade"]),
        daily_trade_cap=int(row["daily_trade_cap"]),
        tickers=str(row["tickers"] or _DEFAULT_TICKERS),
        profit_target_pct=float(row["profit_target_pct"]),
        max_adverse_pct=float(row["max_adverse_pct"]),
        time_cap_minutes=int(row["time_cap_minutes"]),
        session_start=str(row["session_start"]),
        session_end=str(row["session_end"]),
        allow_premarket=bool(row["allow_premarket"]),
        allow_postmarket=bool(row["allow_postmarket"]),
        per_contract_fee=float(row["per_contract_fee"]),
        per_order_fee=float(row["per_order_fee"]),
        rsi_filter=bool(row["rsi_filter"]),
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
    status = str(state["status"]) if state else "inactive"
    started_at = state["started_at"] if state else None
    equity = _latest_equity(db)
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


def status_payload(db) -> Dict[str, Any]:
    st = get_status(db)
    settings = load_settings(db)
    return {
        "status": st.status,
        "started_at": st.started_at,
        "account_equity": st.account_equity,
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
    row = db.execute(
        "SELECT balance FROM scalper_lf_equity ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if row:
        return float(row["balance"])
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
    wins = sum(1 for row in rows if float(row["net_pl"] or 0.0) > 0.0)
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
    return int(row["id"]) if row else None


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
    qty = int(row["qty"])
    entry_price = float(row["entry_price"])
    entry_fees = float(row["fees"])
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
    return LFActivity(
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
        " scalper_lf_activity ORDER BY entry_time DESC"
    )
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = db.execute(query).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
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
    return [EquityPoint(ts=row["ts"], balance=float(row["balance"])) for row in rows]


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

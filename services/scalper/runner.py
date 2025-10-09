"""Background loops for paper scalper engines."""
from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from db import DB_PATH
from services.data_provider import fetch_bars_async
from services.scalper import hf_engine, lf_engine
from utils import TZ

logger = logging.getLogger(__name__)

HF_LOOP_INTERVAL = 60.0  # seconds
LF_LOOP_INTERVAL = 60.0  # seconds
HF_LOOKBACK_MINUTES = 45
LF_LOOKBACK_MINUTES = 120


@contextmanager
def _db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        try:
            cursor.connection.close()
        except Exception:
            conn.close()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return float(result)


def _parse_tickers(raw: str) -> List[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def _normalize_liquidity(volume: float) -> float:
    if volume <= 0:
        return 0.0
    return max(0.0, min(volume / 100000.0, 1.0))


def _parse_ts(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _df_to_bars(df) -> List[Dict[str, float | str]]:
    bars: List[Dict[str, float | str]] = []
    if df is None or getattr(df, "empty", True):
        return bars
    for ts, row in df.sort_index().iterrows():
        open_val = _safe_float(row.get("Open"))
        high_val = _safe_float(row.get("High"), open_val)
        low_val = _safe_float(row.get("Low"), open_val)
        close_val = _safe_float(row.get("Close"), open_val)
        volume_val = _safe_float(row.get("Volume"), 0.0)
        vwap = close_val
        if high_val or low_val:
            vwap = (high_val + low_val + close_val) / 3.0
        volatility = abs(high_val - low_val)
        bars.append(
            {
                "ts": ts.isoformat(),
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
                "vwap": vwap,
                "volatility": volatility,
                "liquidity": _normalize_liquidity(volume_val),
            }
        )
    return bars


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    ema_values: List[float] = []
    ema = float(values[0])
    for price in values:
        ema = float(price) * k + ema * (1 - k)
        ema_values.append(ema)
    return ema_values


def _rsi(values: Sequence[float], period: int = 14) -> List[float]:
    if not values:
        return []
    result: List[float] = []
    gains: List[float] = []
    losses: List[float] = []
    for idx, value in enumerate(values):
        val = float(value)
        if idx == 0:
            gains.append(0.0)
            losses.append(0.0)
            result.append(50.0)
            continue
        delta = val - float(values[idx - 1])
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
        if idx < period:
            result.append(50.0)
            continue
        avg_gain = sum(gains[idx - period + 1 : idx + 1]) / period
        avg_loss = sum(losses[idx - period + 1 : idx + 1]) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - 100 / (1 + rs))
    return result


async def _fetch_recent_bars(
    tickers: Sequence[str],
    *,
    lookback_minutes: int,
    now: datetime,
) -> Tuple[Dict[str, List[Dict[str, float | str]]], Dict[str, str]]:
    if not tickers:
        return {}, {}
    end = now.astimezone(timezone.utc)
    start = end - timedelta(minutes=max(1, lookback_minutes))
    try:
        result = await fetch_bars_async(list(tickers), "1m", start, end)
    except Exception:
        logger.exception("scalper_fetch_failed symbols=%s", list(tickers))
        return {}, {}
    bars_by_symbol: Dict[str, List[Dict[str, float | str]]] = {}
    providers: Dict[str, str] = {}
    for symbol in tickers:
        df = result.get(symbol)
        bars = _df_to_bars(df)
        bars_by_symbol[symbol] = bars
        provider = ""
        if df is not None:
            provider = getattr(df, "attrs", {}).get("provider", "")
        providers[symbol] = provider or "db"
        if not bars:
            logger.info("scalper_no_bars symbol=%s provider=%s", symbol, providers[symbol])
    if result.errors:
        for sym, exc in result.errors.items():
            logger.error("scalper_fetch_error symbol=%s err=%s", sym, exc)
    return bars_by_symbol, providers


def _should_run_lf_session(now: datetime, settings: lf_engine.LFSettings) -> bool:
    if settings.allow_premarket and settings.allow_postmarket:
        return True
    now_et = now.astimezone(TZ)
    try:
        start_parts = [int(part) for part in settings.session_start.split(":", 1)]
        end_parts = [int(part) for part in settings.session_end.split(":", 1)]
        start_dt = datetime.combine(now_et.date(), dt_time(start_parts[0], start_parts[1]), tzinfo=TZ)
        end_dt = datetime.combine(now_et.date(), dt_time(end_parts[0], end_parts[1]), tzinfo=TZ)
    except Exception:
        return True
    if not settings.allow_premarket and now_et < start_dt:
        return False
    if not settings.allow_postmarket and now_et >= end_dt:
        return False
    return True


async def hf_loop(
    market_is_open: Callable[[datetime], bool],
    now_et: Callable[[], datetime],
) -> None:
    logger.info("hf_paper_loop_starting")
    startup_logged = False
    iteration = 0
    while True:
        iteration += 1
        loop_started = asyncio.get_event_loop().time()
        try:
            ts = now_et()
            if not market_is_open(ts):
                logger.info("hf_loop_skip reason=market_closed now=%s", ts.isoformat())
            else:
                await _run_hf_iteration(ts, startup_logged, iteration)
                startup_logged = True
        except Exception:
            logger.exception("hf_loop_error iteration=%d", iteration)
        elapsed = asyncio.get_event_loop().time() - loop_started
        await asyncio.sleep(max(5.0, HF_LOOP_INTERVAL - elapsed))


async def _run_hf_iteration(now: datetime, startup_logged: bool, iteration: int) -> None:
    with _db_cursor() as db:
        settings = hf_engine.load_settings(db)
        status = hf_engine.get_status(db)
        tickers = _parse_tickers(settings.tickers)
        if not startup_logged:
            logger.info(
                "hf_engine_started tickers=%s daily_cap=%d pct_per_trade=%.2f",
                ",".join(tickers) or "<none>",
                settings.daily_trade_cap,
                settings.pct_per_trade,
            )
        if status.status != "active":
            logger.info("hf_loop_skip reason=status status=%s", status.status)
            return
        if status.halted:
            logger.warning("hf_loop_halted reason=drawdown")
            return
        if not tickers:
            logger.warning("hf_loop_skip reason=no_tickers")
            return
        bars_by_symbol, providers = await _fetch_recent_bars(
            tickers, lookback_minutes=HF_LOOKBACK_MINUTES, now=now
        )
        closed = 0
        opened = 0
        for row in db.execute(
            "SELECT id, ticker, entry_time, entry_price FROM scalper_hf_activity WHERE status='open'"
        ).fetchall():
            ticker = str(row["ticker"]).upper()
            bars = bars_by_symbol.get(ticker, [])
            entry_time = _parse_ts(row["entry_time"])
            if not bars or entry_time is None:
                continue
            recent = [bar for bar in bars if _parse_ts(bar["ts"]) and _parse_ts(bar["ts"]) >= entry_time]
            if not recent:
                continue
            exit_price, reason, exit_ts_raw = hf_engine.evaluate_exit(
                entry_price=float(row["entry_price"]),
                bars=recent,
                target_pct=settings.profit_target_pct,
                stop_pct=settings.max_adverse_pct,
                time_cap_minutes=settings.time_cap_minutes,
            )
            if reason == "timeout" and not exit_ts_raw:
                continue
            exit_dt = _parse_ts(exit_ts_raw) or _parse_ts(recent[-1]["ts"]) or now.astimezone(timezone.utc)
            liquidity = float(recent[-1].get("liquidity", 0.0))
            closed_trade = hf_engine.close_trade(
                db,
                int(row["id"]),
                mid_price=float(exit_price),
                exit_time=exit_dt,
                reason=reason,
                liquidity=liquidity,
                settings=settings,
            )
            if closed_trade:
                closed += 1
                logger.info(
                    "hf_loop_closed id=%s ticker=%s reason=%s exit=%.2f",
                    closed_trade.id,
                    ticker,
                    reason,
                    float(exit_price),
                )
        for ticker in tickers:
            bars = bars_by_symbol.get(ticker, [])
            if len(bars) < 2:
                continue
            closes = [float(bar["close"]) for bar in bars]
            ema9 = _ema(closes, 9)
            last_bar = bars[-1]
            prev_bar = bars[-2]
            price = float(last_bar["close"])
            if price <= 0:
                continue
            momentum = price - float(prev_bar["close"])
            vwap = float(last_bar.get("vwap", price))
            ema_val = ema9[-1] if ema9 else price
            volatility = float(last_bar.get("volatility", 0.0))
            liquidity = float(last_bar.get("liquidity", 0.0))
            entry_dt = _parse_ts(last_bar["ts"]) or now.astimezone(timezone.utc)
            trade_id = hf_engine.open_trade(
                db,
                ticker=ticker,
                option_type="CALL" if momentum >= 0 else "PUT",
                strike=None,
                expiry=None,
                mid_price=price,
                entry_time=entry_dt,
                momentum_score=momentum,
                vwap=vwap,
                ema9=ema_val,
                volatility=volatility,
                liquidity=liquidity,
                settings=settings,
            )
            if trade_id:
                opened += 1
                logger.info(
                    "hf_loop_opened id=%s ticker=%s price=%.2f momentum=%.4f provider=%s",
                    trade_id,
                    ticker,
                    price,
                    momentum,
                    providers.get(ticker, "db"),
                )
        logger.info(
            "hf_loop_iteration iteration=%d opened=%d closed=%d provider_map=%s",
            iteration,
            opened,
            closed,
            providers,
        )


async def lf_loop(
    market_is_open: Callable[[datetime], bool],
    now_et: Callable[[], datetime],
) -> None:
    logger.info("lf_paper_loop_starting")
    startup_logged = False
    iteration = 0
    while True:
        iteration += 1
        loop_started = asyncio.get_event_loop().time()
        try:
            ts = now_et()
            if not market_is_open(ts):
                logger.info("lf_loop_skip reason=market_closed now=%s", ts.isoformat())
            else:
                await _run_lf_iteration(ts, startup_logged, iteration)
                startup_logged = True
        except Exception:
            logger.exception("lf_loop_error iteration=%d", iteration)
        elapsed = asyncio.get_event_loop().time() - loop_started
        await asyncio.sleep(max(5.0, LF_LOOP_INTERVAL - elapsed))


async def _run_lf_iteration(now: datetime, startup_logged: bool, iteration: int) -> None:
    with _db_cursor() as db:
        settings = lf_engine.load_settings(db)
        status = lf_engine.get_status(db)
        tickers = _parse_tickers(settings.tickers)
        if not startup_logged:
            logger.info(
                "lf_engine_started tickers=%s daily_cap=%d pct_per_trade=%.2f",
                ",".join(tickers) or "<none>",
                settings.daily_trade_cap,
                settings.pct_per_trade,
            )
        if status.status != "active":
            logger.info("lf_loop_skip reason=status status=%s", status.status)
            return
        if not _should_run_lf_session(now, settings):
            logger.info("lf_loop_skip reason=session_window")
            return
        if not tickers:
            logger.warning("lf_loop_skip reason=no_tickers")
            return
        bars_by_symbol, providers = await _fetch_recent_bars(
            tickers, lookback_minutes=LF_LOOKBACK_MINUTES, now=now
        )
        closed = 0
        mark_updates: List[Tuple[float, int]] = []
        for row in db.execute(
            "SELECT id, ticker, entry_time, entry_price FROM scalper_lf_activity WHERE status='open'"
        ).fetchall():
            ticker = str(row["ticker"]).upper()
            bars = bars_by_symbol.get(ticker, [])
            entry_time = _parse_ts(row["entry_time"])
            if not bars or entry_time is None:
                continue
            recent = [bar for bar in bars if _parse_ts(bar["ts"]) and _parse_ts(bar["ts"]) >= entry_time]
            if not recent:
                continue
            exit_price, reason, exit_ts_raw = lf_engine.evaluate_exit(
                entry_price=float(row["entry_price"]),
                bars=recent,
                target_pct=settings.profit_target_pct,
                stop_pct=settings.max_adverse_pct,
                time_cap_minutes=settings.time_cap_minutes,
            )
            exit_dt = _parse_ts(exit_ts_raw)
            if reason == "timeout" and exit_dt is None:
                last_price = float(recent[-1]["close"])
                mark_updates.append((last_price, int(row["id"])))
                continue
            if exit_dt is None:
                exit_dt = _parse_ts(recent[-1]["ts"]) or now.astimezone(timezone.utc)
            closed_trade = lf_engine.close_trade(
                db,
                int(row["id"]),
                mid_price=float(exit_price),
                exit_time=exit_dt,
                reason=reason,
                settings=settings,
            )
            if closed_trade:
                closed += 1
                logger.info(
                    "lf_loop_closed id=%s ticker=%s reason=%s exit=%.2f",
                    closed_trade.id,
                    ticker,
                    reason,
                    float(exit_price),
                )
        if mark_updates:
            db.executemany(
                "UPDATE scalper_lf_activity SET mark_price=? WHERE id=?",
                [(price, trade_id) for price, trade_id in mark_updates],
            )
            db.connection.commit()
        opened = 0
        for ticker in tickers:
            bars = bars_by_symbol.get(ticker, [])
            if len(bars) < 6:
                continue
            closes = [float(bar["close"]) for bar in bars]
            highs = [float(bar["high"]) for bar in bars]
            idx = len(bars) - 1
            price = closes[idx]
            if price <= 0:
                continue
            window_high = max(highs[max(0, idx - 5) : idx]) if idx >= 1 else price
            vwap = float(bars[idx].get("vwap", price))
            rsi_values = _rsi(closes)
            rsi_ok = True
            if settings.rsi_filter and idx < len(rsi_values):
                rsi_ok = rsi_values[idx] >= 50.0
            if not rsi_ok:
                continue
            if not (price > window_high and price >= vwap):
                continue
            entry_dt = _parse_ts(bars[idx]["ts"]) or now.astimezone(timezone.utc)
            trade_id = lf_engine.open_trade(
                db,
                ticker=ticker,
                option_type="CALL",
                strike=None,
                expiry=None,
                mid_price=price,
                entry_time=entry_dt,
                settings=settings,
            )
            if trade_id:
                opened += 1
                logger.info(
                    "lf_loop_opened id=%s ticker=%s price=%.2f provider=%s",
                    trade_id,
                    ticker,
                    price,
                    providers.get(ticker, "db"),
                )
        logger.info(
            "lf_loop_iteration iteration=%d opened=%d closed=%d marked=%d providers=%s",
            iteration,
            opened,
            closed,
            len(mark_updates),
            providers,
        )


__all__ = ["hf_loop", "lf_loop"]

# ruff: noqa: E501
import asyncio
import json
import logging
import random
import sqlite3
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict

from config import settings
from db import DB_PATH, get_settings, set_last_run
from prometheus_client import Counter
from scanner import preload_prices
from services.alerts import alert_due, in_earnings_blackout
from services.data_fetcher import fetch_prices as yahoo_fetch
from services.emailer import send_email
from services.forward import create_forward_test, update_forward_tests
from services.market_data import fetch_prices as md_fetch_prices
from services.market_data import get_prices
from services.polygon_client import fetch_polygon_prices
from services.price_store import upsert_bars

logger = logging.getLogger(__name__)

job_queued = Counter("scheduler_jobs_queued_total", "Jobs queued")
job_success = Counter("scheduler_jobs_success_total", "Jobs succeeded")
job_failure = Counter("scheduler_jobs_failure_total", "Jobs failed")


class WorkQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, Callable[[], Awaitable[None]]]] = (
            asyncio.Queue()
        )
        self.keys: set[str] = set()

    def enqueue(self, key: str, coro_fn: Callable[[], Awaitable[None]]) -> None:
        if key in self.keys:
            return
        self.queue.put_nowait((key, coro_fn))
        self.keys.add(key)
        job_queued.inc()

    async def worker(self) -> None:
        while True:
            key, fn = await self.queue.get()
            try:
                try:
                    await asyncio.wait_for(fn(), timeout=settings.job_timeout)
                    job_success.inc()
                except Exception:
                    logger.exception("job failed key=%s", key)
                    job_failure.inc()
            finally:
                self.keys.discard(key)
                self.queue.task_done()


work_queue = WorkQueue()


def queue_gap_fill(symbol: str, start, end, interval: str) -> None:
    async def _job() -> None:
        df_y = yahoo_fetch([symbol], interval, (end - start).days / 365.0).get(symbol)
        if df_y is not None and not df_y.empty:
            upsert_bars(symbol, df_y)
        df_p = fetch_polygon_prices([symbol], interval, start, end).get(symbol)
        if df_p is not None and not df_p.empty:
            upsert_bars(symbol, df_p)

    key = f"gap:{symbol}:{start.isoformat()}:{end.isoformat()}"
    work_queue.enqueue(key, _job)


def _avg_daily_volume(ticker: str) -> float:
    df = md_fetch_prices([ticker], "1d", 0.5).get(ticker)
    if df is None or getattr(df, "empty", True):
        return 0.0
    for col in ["volume", "Volume", "VOL", "vol"]:
        if col in df.columns:
            return float(df[col].tail(90).mean())
    return 0.0


def _liquidity_ok(ticker: str, min_adv: int = 200_000) -> bool:
    return _avg_daily_volume(ticker) >= min_adv


def _fetch_earnings(ticker: str) -> list[datetime]:  # pragma: no cover - stub
    return []


def _near_earnings(ticker: str, now: datetime) -> bool:
    try:
        dates = _fetch_earnings(ticker)
    except Exception:
        return False
    return in_earnings_blackout(dates, now)


def _fav_to_params(f: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "interval": f.get("interval", "15m"),
        "direction": f.get("direction", "BOTH"),
        "target_pct": f.get("target_pct", 1.0),
        "stop_pct": f.get("stop_pct", 0.5),
        "window_value": f.get("window_value", 4.0),
        "window_unit": f.get("window_unit", "Hours"),
        "lookback_years": f.get("lookback_years", 0.2),
        "max_tt_bars": f.get("max_tt_bars", 12),
        "min_support": f.get("min_support", 20),
        "delta_assumed": f.get("delta", 0.4),
        "theta_per_day_pct": f.get("theta_day", 0.2),
        "atrz_gate": f.get("atrz", 0.10),
        "slope_gate_pct": f.get("slope", 0.02),
        "use_regime": f.get("use_regime", 0),
        "regime_trend_only": f.get("trend_only", 0),
        "vix_z_max": f.get("vix_z_max", 3.0),
        "slippage_bps": f.get("slippage_bps", 7.0),
        "vega_scale": f.get("vega_scale", 0.03),
        "scan_min_hit": 50.0,
        "scan_max_dd": 50.0,
    }


async def favorites_loop(
    market_is_open: Callable[[datetime], bool],
    now_et: Callable[[], datetime],
    compute_scan_for_ticker: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> None:
    logger.info("scheduler started")
    freq = settings.fav_scan_freq_min
    while True:
        start_time = asyncio.get_event_loop().time()
        jitter = random.uniform(0, 5)
        await asyncio.sleep(jitter)
        try:
            ts = now_et()
            if market_is_open(ts):
                boundary = ts.replace(second=0, microsecond=0)
                boundary = boundary.replace(
                    minute=(boundary.minute - boundary.minute % freq)
                )
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    db = conn.cursor()
                    st = get_settings(db)
                    throttle = int(st.get("throttle_minutes") or 60)
                    last_boundary = st.get("last_boundary") or ""
                    last_run_at = st.get("last_run_at") or ""

                    should_run = boundary.isoformat() != last_boundary
                    if last_run_at:
                        last_dt = datetime.fromisoformat(last_run_at)
                        if (ts - last_dt).total_seconds() < throttle * 60:
                            should_run = False

                    if should_run:
                        db.execute(
                            "SELECT * FROM favorites WHERE alerts_enabled=1 ORDER BY id DESC",
                        )
                        favs = [dict(r) for r in db.fetchall()]
                        if not favs:
                            set_last_run(boundary.isoformat(), db)
                            continue
                        preload_prices(
                            [f["ticker"] for f in favs],
                            "15m",
                            max(f.get("lookback_years", 0.2) for f in favs),
                        )
                        hits = []
                        for f in favs:
                            f.setdefault(
                                "cooldown_minutes",
                                int(st.get("fav_cooldown_minutes") or 30),
                            )
                            ticker = f.get("ticker", "?")
                            try:
                                if await asyncio.to_thread(_near_earnings, ticker, ts):
                                    continue
                                if not await asyncio.to_thread(_liquidity_ok, ticker):
                                    continue
                                params = _fav_to_params(f)
                                row = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        compute_scan_for_ticker, ticker, params
                                    ),
                                    timeout=settings.job_timeout,
                                )
                                if (
                                    row
                                    and row.get("hit_pct", 0) >= 50
                                    and row.get("avg_roi_pct", 0) > 0
                                ):
                                    if alert_due(f, boundary, ts):
                                        subject = f"[Pattern Alert] {ticker} {f.get('direction')} — {f.get('rule')}"
                                        body = (
                                            f"Ticker: {ticker}\n"
                                            f"Direction: {f.get('direction')}\n"
                                            f"Pattern: {f.get('rule')}\n"
                                            f"Bar Time: {boundary.isoformat()}\n"
                                        )
                                        send_email(st, subject, body)
                                        db.execute(
                                            "UPDATE favorites SET last_notified_ts=?, last_signal_bar=? WHERE id=?",
                                            (
                                                ts.isoformat(),
                                                boundary.isoformat(),
                                                f["id"],
                                            ),
                                        )
                                        db.connection.commit()
                                        hits.append(row)
                            except KeyError as e:
                                logger.warning(
                                    "favorite scan missing data ticker=%s date=%s",
                                    ticker,
                                    e.args[0],
                                )
                                continue
                            except Exception:
                                logger.exception(
                                    "favorite scan failed ticker=%s", ticker
                                )
                        if hits:
                            started = ts.isoformat()
                            universe = ",".join({r.get("ticker", "") for r in hits})
                            settings_json = json.dumps({"source": "favorites"})
                            db.execute(
                                """INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count, settings_json)
                                    VALUES(?,?,?,?,?,?,?)""",
                                (
                                    started,
                                    "favorites",
                                    settings_json,
                                    universe,
                                    started,
                                    len(hits),
                                    settings_json,
                                ),
                            )
                            run_id = db.lastrowid
                            for r in hits:
                                db.execute(
                                    """INSERT INTO run_results(run_id, ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule)
                                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                                    (
                                        run_id,
                                        r.get("ticker"),
                                        r.get("direction"),
                                        float(r.get("avg_roi_pct", 0.0)),
                                        float(r.get("hit_pct", 0.0)),
                                        int(r.get("support", 0)),
                                        float(r.get("avg_tt", 0.0)),
                                        float(r.get("avg_dd_pct", 0.0)),
                                        float(r.get("stability", 0.0)),
                                        r.get("rule", ""),
                                    ),
                                )
                            db.connection.commit()
                        set_last_run(boundary.isoformat(), db)
        except Exception as e:
            logger.error("scheduler error: %r", e)
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, 60 - elapsed))


async def forward_tests_loop(
    market_is_open: Callable[[datetime], bool],
    now_et: Callable[[], datetime],
    get_prices_fn=get_prices,
) -> None:
    """Background loop advancing forward tests on a cadence."""
    logger.info("forward test scheduler started")
    last_boundary = None
    while True:
        start_time = asyncio.get_event_loop().time()
        try:
            ts = now_et()
            boundary = ts.replace(second=0, microsecond=0)
            run_intraday = market_is_open(ts) and boundary.minute % 15 == 0
            run_after_hours = not market_is_open(ts) and boundary.minute == 0
            if boundary != last_boundary and (run_intraday or run_after_hours):
                last_boundary = boundary
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    db = conn.cursor()
                    db.execute("SELECT * FROM favorites ORDER BY id DESC")
                    favs = [dict(r) for r in db.fetchall()]
                    for f in favs:
                        db.execute(
                            "SELECT status FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1",
                            (f["id"],),
                        )
                        row = db.fetchone()
                        if row is None or row["status"] in ("ok", "error"):
                            create_forward_test(db, f, get_prices_fn)
                    update_forward_tests(db, get_prices_fn)
        except Exception as e:
            logger.error("forward test scheduler error: %r", e)
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, 60 - elapsed))


def setup_scheduler(
    app,
    market_is_open,
    now_et,
    compute_scan_for_ticker,
    get_prices_fn=get_prices,
):
    @app.on_event("startup")
    async def on_startup():
        asyncio.create_task(work_queue.worker())
        asyncio.create_task(
            favorites_loop(market_is_open, now_et, compute_scan_for_ticker)
        )
        asyncio.create_task(forward_tests_loop(market_is_open, now_et, get_prices_fn))

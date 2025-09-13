# ruff: noqa: E501
import asyncio
import logging
import random
import sqlite3
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict

from config import settings
from db import DB_PATH, get_settings, set_last_run
from prometheus_client import Counter
from scanner import preload_prices
from routes import _update_forward_tests
from services.data_fetcher import fetch_prices as yahoo_fetch
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


async def favorites_loop(
    market_is_open: Callable[[datetime], bool],
    now_et: Callable[[], datetime],
    compute_scan_for_ticker: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> None:
    logger.info("scheduler started")
    while True:
        start_time = asyncio.get_event_loop().time()
        jitter = random.uniform(0, 5)
        await asyncio.sleep(jitter)
        try:
            ts = now_et()
            if market_is_open(ts):
                boundary = ts.replace(second=0, microsecond=0)
                boundary = boundary.replace(
                    minute=(boundary.minute - boundary.minute % 15)
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
                            "SELECT ticker, direction, interval, rule FROM favorites ORDER BY id DESC"
                        )
                        favs = [dict(r) for r in db.fetchall()]
                        params = dict(
                            interval="15m",
                            direction="BOTH",
                            scan_min_hit=50.0,
                            atrz_gate=0.10,
                            slope_gate_pct=0.02,
                        )
                        preload_prices(
                            [f["ticker"] for f in favs],
                            params.get("interval", "15m"),
                            params.get("lookback_years", 2.0),
                        )
                        hits = []
                        for f in favs:
                            ticker = f.get("ticker", "?")
                            try:
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
                        # TODO: email YES hits in a readable format
                        # TODO: archive favorites 15m scan results only if there are YES hits
                        set_last_run(boundary.isoformat(), db)
        except Exception as e:
            logger.error("scheduler error: %r", e)
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, 60 - elapsed))


async def forward_tests_loop(
    market_is_open: Callable[[datetime], bool], now_et: Callable[[], datetime]
) -> None:
    logger.info("forward tests scheduler started")
    while True:
        start_time = asyncio.get_event_loop().time()
        jitter = random.uniform(0, 5)
        await asyncio.sleep(jitter)
        try:
            ts = now_et()
            run = False
            if market_is_open(ts):
                if ts.minute % 15 == 0:
                    run = True
            elif ts.minute == 0:
                run = True
            if run:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    db = conn.cursor()
                    _update_forward_tests(db)
        except Exception as e:
            logger.error("forward tests loop error: %r", e)
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, 60 - elapsed))


def setup_scheduler(app, market_is_open, now_et, compute_scan_for_ticker):
    @app.on_event("startup")
    async def on_startup():
        asyncio.create_task(work_queue.worker())
        asyncio.create_task(
            favorites_loop(market_is_open, now_et, compute_scan_for_ticker)
        )
        asyncio.create_task(forward_tests_loop(market_is_open, now_et))

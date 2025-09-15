# ruff: noqa: E501
import asyncio
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict

from config import settings
from db import DB_PATH, get_settings, set_last_run
from prometheus_client import Counter  # type: ignore
from routes import _update_forward_tests  # type: ignore
from scanner import preload_prices  # type: ignore
from services.polygon_client import fetch_polygon_prices_async
from services.price_store import covers, get_coverage, missing_ranges, upsert_bars
from utils import clamp_market_closed

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
                except Exception as e:
                    import traceback as _tb

                    logger.error(
                        "job failed key=%s err=%s msg=%s\n%s",
                        key,
                        type(e).__name__,
                        e,
                        _tb.format_exc(),
                    )
                    job_failure.inc()
            finally:
                self.keys.discard(key)
                self.queue.task_done()


work_queue = WorkQueue()


def queue_gap_fill(symbol: str, start, end, interval: str) -> None:
    async def _job() -> None:
        if settings.clamp_market_closed:
            new_end, clamped = clamp_market_closed(start, end)
            if clamped:
                logger.debug(
                    "clamp reason=market_closed end=%s requested_end=%s",
                    new_end.isoformat(),
                    end.isoformat(),
                )
                if new_end <= start:
                    return
                end_local = new_end
            else:
                end_local = end
        else:
            end_local = end
        cov_min, cov_max = get_coverage(symbol, interval)
        if covers(start, end_local, cov_min, cov_max):
            logger.debug(
                "db_coverage_ok symbol=%s interval=%s %s..%s",
                symbol,
                interval,
                start,
                end_local,
            )
            return
        to_fetch = missing_ranges(start, end_local, cov_min, cov_max)
        logger.info(
            "db_coverage_gap symbol=%s interval=%s missing=%s",
            symbol,
            interval,
            to_fetch,
        )
        chunk = timedelta(days=settings.backfill_chunk_days)
        for a, b in to_fetch:
            cur = a
            while cur < b:
                nxt = min(cur + chunk, b)
                logger.info(
                    "fetch_call symbol=%s requested_start=%s requested_end=%s",
                    symbol,
                    cur.isoformat(),
                    nxt.isoformat(),
                )
                try:
                    df_map = await fetch_polygon_prices_async(
                        [symbol], interval, cur, nxt
                    )
                    df_p = df_map.get(symbol)
                    if df_p is not None and not df_p.empty:
                        upsert_bars(symbol, df_p, interval)
                        logger.info(
                            "fetch_ok symbol=%s rows=%d",
                            symbol,
                            len(df_p),
                        )
                    else:
                        logger.info("fetch_empty symbol=%s", symbol)
                except Exception as e:
                    import traceback as _tb

                    logger.error(
                        "fetch_error symbol=%s err=%s msg=%s\n%s",
                        symbol,
                        type(e).__name__,
                        e,
                        _tb.format_exc(),
                    )
                cur = nxt

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
                        params: Dict[str, Any] = dict(
                            interval="15m",
                            direction="BOTH",
                            scan_min_hit=50.0,
                            atrz_gate=0.10,
                            slope_gate_pct=0.02,
                        )
                        preload_prices(
                            [f["ticker"] for f in favs],
                            params.get("interval", "15m"),
                            float(params.get("lookback_years", 2.0)),
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
                            except Exception as e:
                                import traceback as _tb

                                logger.error(
                                    "favorite scan failed ticker=%s err=%s msg=%s\n%s",
                                    ticker,
                                    type(e).__name__,
                                    e,
                                    _tb.format_exc(),
                                )
                        # TODO: email YES hits in a readable format
                        # TODO: archive favorites 15m scan results only if there are YES hits
                        set_last_run(boundary.isoformat(), db)
        except Exception as e:
            import traceback as _tb

            logger.error(
                "scheduler error err=%s msg=%s\n%s",
                type(e).__name__,
                e,
                _tb.format_exc(),
            )
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
            import traceback as _tb

            logger.error(
                "forward tests loop error err=%s msg=%s\n%s",
                type(e).__name__,
                e,
                _tb.format_exc(),
            )
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

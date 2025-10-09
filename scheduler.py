# ruff: noqa: E501
import asyncio
import json
import logging
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict

from config import settings
from db import DB_PATH, get_settings, row_to_dict, set_last_run
from prometheus_client import Counter  # type: ignore
from routes import _update_forward_tests  # type: ignore
from scanner import preload_prices  # type: ignore
from services.http_client import RateLimitTimeoutSoon
from services.data_provider import compute_request_window, fetch_bars_async
from services.price_store import covers, get_coverage, missing_ranges, upsert_bars
from services import favorites_alerts
from services.scalper import runner as scalper_runner
from utils import clamp_market_closed, market_is_open as util_market_is_open

logger = logging.getLogger(__name__)

job_queued = Counter("scheduler_jobs_queued_total", "Jobs queued")
job_success = Counter("scheduler_jobs_success_total", "Jobs succeeded")
job_failure = Counter("scheduler_jobs_failure_total", "Jobs failed")

RETRY_SHIFT = timedelta(minutes=15)
FAIL_FAST_NETWORK_BUFFER = float(os.getenv("SCAN_FAIL_FAST_NETWORK_BUFFER", "1.5"))
FAIL_FAST_SAFETY_MARGIN = float(os.getenv("SCAN_FAIL_FAST_SAFETY_MARGIN", "0.25"))
REQUEUE_MIN_DELAY = float(os.getenv("SCAN_FAIL_FAST_REQUEUE_MIN", "0.75"))
REQUEUE_MAX_DELAY = float(os.getenv("SCAN_FAIL_FAST_REQUEUE_MAX", "2.0"))


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
        start_time = time.monotonic()
        rows_total = 0
        deadline = start_time + float(os.getenv("SCAN_SOFT_DEADLINE", "120"))
        job_deadline = start_time + float(settings.job_timeout)
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
        last_bar = cov_max
        for a, b in to_fetch:
            cur = a
            while cur < b:
                now_mono = time.monotonic()
                if now_mono > deadline:
                    logger.info("fetch_deadline symbol=%s", symbol)
                    break
                if now_mono >= job_deadline:
                    logger.info("gap_job_timeout symbol=%s interval=%s", symbol, interval)
                    return
                nxt = min(cur + chunk, b)
                request_start = cur
                request_end = nxt
                now_utc = datetime.now(timezone.utc)
                mode = "range"
                if interval == "15m":
                    request_start, request_end, mode = compute_request_window(
                        symbol,
                        interval,
                        request_start,
                        request_end,
                        last_bar=last_bar,
                        now=now_utc,
                    )
                logger.info(
                    json.dumps(
                        {
                            "type": "gap_slice",
                            "symbol": symbol,
                            "interval": interval,
                            "mode": mode,
                            "start": request_start.isoformat(),
                            "end": request_end.isoformat(),
                        }
                    )
                )

                attempt = 1
                retry_done = False
                window_start = request_start
                window_end = request_end

                while True:
                    remaining_job = job_deadline - time.monotonic()
                    if remaining_job <= 0:
                        logger.info(
                            json.dumps(
                                {
                                    "type": "gap_fail_fast",
                                    "symbol": symbol,
                                    "interval": interval,
                                    "predicted_wait_s": 0.0,
                                    "remaining_s": 0.0,
                                    "reason": "deadline_elapsed",
                                }
                            )
                        )
                        return

                    rate_state: Dict[str, float] = {}
                    timeout_ctx = {
                        "deadline": job_deadline,
                        "remaining": max(0.0, remaining_job),
                        "network_buffer": FAIL_FAST_NETWORK_BUFFER,
                        "safety_margin": FAIL_FAST_SAFETY_MARGIN,
                        "rate_state": rate_state,
                    }

                    call_start = time.monotonic()
                    try:
                        df_map = await fetch_bars_async(
                            [symbol],
                            interval,
                            window_start,
                            window_end,
                            timeout_ctx=timeout_ctx,
                        )
                    except RateLimitTimeoutSoon as rl:
                        log_data = {
                            "type": "gap_fail_fast",
                            "symbol": symbol,
                            "interval": interval,
                            "predicted_wait_s": rl.predicted_wait,
                            "remaining_s": rl.remaining,
                        }
                        logger.info(json.dumps(log_data))

                        delay = random.uniform(REQUEUE_MIN_DELAY, REQUEUE_MAX_DELAY)

                        async def _requeue() -> None:
                            await asyncio.sleep(delay)
                            queue_gap_fill(symbol, start, end_local, interval)

                        asyncio.create_task(_requeue())
                        return
                    except Exception as e:
                        import traceback as _tb

                        logger.error(
                            "fetch_error symbol=%s err=%s msg=%s\n%s",
                            symbol,
                            type(e).__name__,
                            e,
                            _tb.format_exc(),
                        )
                        break

                    df_p = df_map.get(symbol)
                    rows = len(df_p) if df_p is not None and not df_p.empty else 0
                    duration_ms = int((time.monotonic() - call_start) * 1000)
                    rate_wait_ms = int(rate_state.get("waited", 0.0) * 1000)

                    logger.info(
                        json.dumps(
                            {
                                "type": "gap_request_done",
                                "symbol": symbol,
                                "interval": interval,
                                "mode": mode,
                                "attempt": attempt,
                                "start": window_start.isoformat(),
                                "end": window_end.isoformat(),
                                "rows": rows,
                                "duration_ms": duration_ms,
                                "rate_wait_ms": rate_wait_ms,
                            }
                        )
                    )

                    if rows > 0 and df_p is not None:
                        rows_total += rows
                        upsert_bars(symbol, df_p, interval)
                        try:
                            last_bar = df_p.index.max().to_pydatetime()
                        except Exception:
                            pass
                        logger.info("fetch_ok symbol=%s rows=%d", symbol, rows)
                        break

                    if (
                        mode == "single_bucket"
                        and not retry_done
                        and util_market_is_open(now_utc)
                    ):
                        retry_done = True
                        prev_start = window_start
                        prev_end = window_end
                        window_start = prev_start - RETRY_SHIFT
                        logger.info(
                            json.dumps(
                                {
                                    "type": "gap_retry_smaller_slice",
                                    "symbol": symbol,
                                    "interval": interval,
                                    "prev_start": prev_start.isoformat(),
                                    "prev_end": prev_end.isoformat(),
                                    "next_start": window_start.isoformat(),
                                    "next_end": window_end.isoformat(),
                                }
                            )
                        )
                        attempt += 1
                        continue

                    if mode == "single_bucket":
                        logger.info(
                            json.dumps(
                                {
                                    "type": "gap_empty_near_now",
                                    "symbol": symbol,
                                    "interval": interval,
                                    "start": window_start.isoformat(),
                                    "end": window_end.isoformat(),
                                }
                            )
                        )
                    else:
                        logger.info("fetch_empty symbol=%s", symbol)
                    break

                if mode == "single_bucket":
                    cur = b
                else:
                    cur = request_end

        duration = time.monotonic() - start_time
        logger.info(
            "fetch_summary symbol=%s ranges=%d rows=%d duration=%.2f",
            symbol,
            len(to_fetch),
            rows_total,
            duration,
        )

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
                        favs = [row_to_dict(r, db) for r in db.fetchall()]
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
                                    try:
                                        favorites_alerts.enrich_and_send(
                                            favorites_alerts.FavoriteHitStub(
                                                ticker=row.get("ticker", ticker),
                                                direction=row.get("direction", "UP"),
                                                pattern=row.get("rule", ""),
                                                target_pct=row.get("target_pct", 0.0),
                                                stop_pct=row.get("stop_pct", 0.0),
                                                hit_pct=row.get("hit_pct", 0.0),
                                                avg_roi_pct=row.get("avg_roi_pct", 0.0),
                                                avg_dd_pct=row.get("avg_dd_pct", 0.0),
                                            )
                                        )
                                    except Exception:
                                        logger.exception(
                                            "favorites alert failed ticker=%s", ticker
                                        )
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
        asyncio.create_task(scalper_runner.hf_loop(market_is_open, now_et))
        asyncio.create_task(scalper_runner.lf_loop(market_is_open, now_et))

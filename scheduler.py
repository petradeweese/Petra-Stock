import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Dict, Any

from db import DB_PATH, get_settings, set_last_run
from scanner import preload_prices

logger = logging.getLogger(__name__)


async def favorites_loop(market_is_open: Callable[[datetime], bool], now_et: Callable[[], datetime], compute_scan_for_ticker: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
    logger.info("scheduler started")
    while True:
        start = asyncio.get_event_loop().time()
        try:
            ts = now_et()
            if market_is_open(ts):
                boundary = ts.replace(second=0, microsecond=0)
                boundary = boundary.replace(minute=(boundary.minute - boundary.minute % 15))
                boundary_utc_iso = boundary.astimezone(timezone.utc).isoformat()
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    db = conn.cursor()
                    st = get_settings(db)
                    throttle = int(st["throttle_minutes"] or 60)
                    last_boundary = st["last_boundary"] or ""
                    last_run_at = st["last_run_at"] or ""

                    should_run = boundary_utc_iso != last_boundary
                    if last_run_at:
                        last_dt = datetime.fromisoformat(last_run_at)
                        if (ts - last_dt).total_seconds() < throttle * 60:
                            should_run = False

                    if should_run:
                        db.execute("SELECT ticker, direction, interval, rule FROM favorites ORDER BY id DESC")
                        favs = [dict(r) for r in db.fetchall()]
                        params = dict(
                            interval="15m",
                            direction="BOTH",
                            scan_min_hit=50.0,
                            atrz_gate=0.10,
                            slope_gate_pct=0.02,
                        )
                        preload_prices([f["ticker"] for f in favs], params.get("interval", "15m"), params.get("lookback_years", 2.0))
                        hits = []
                        for f in favs:
                            row = compute_scan_for_ticker(f["ticker"], params)
                            if row and row.get("hit_pct", 0) >= 50 and row.get("avg_roi_pct", 0) > 0:
                                hits.append(row)
                        # TODO: email YES hits in a readable format
                        # TODO: archive favorites 15m scan results only if there are YES hits
                        set_last_run(boundary_utc_iso, db)
        except Exception as e:
            logger.error("scheduler error: %r", e)
        elapsed = asyncio.get_event_loop().time() - start
        await asyncio.sleep(max(0, 60 - elapsed))


def setup_scheduler(app, market_is_open, now_et, compute_scan_for_ticker):
    @app.on_event("startup")
    async def on_startup():
        asyncio.create_task(favorites_loop(market_is_open, now_et, compute_scan_for_ticker))

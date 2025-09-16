import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from threading import Thread
from typing import Callable, Optional, Sequence

from db import get_db
from indices import SP100, TOP150, TOP250
from utils import TZ, last_trading_close, market_is_open

logger = logging.getLogger(__name__)


PREMARKET_OPEN = dt_time(4, 0)


@dataclass
class UniverseResolution:
    symbols: list[str]
    universe: str
    ticker: str
    symbols_total: int
    message: str = ""
    detail: str = ""


_UNIVERSE_ALIASES = {
    "scan150": "scan150",
    "top150": "scan150",
    "options150": "scan150",
    "scan250": "scan250",
    "top250": "scan250",
    "options250": "scan250",
    "sp100": "sp100",
    "sp_100": "sp100",
    "sp100_scan": "sp100",
    "single": "single",
    "single_ticker": "single",
    "favorites": "favorites",
    "favorites_scan": "favorites",
    "favorites_universe": "favorites",
    "favorites_alerts": "favorites",
}


def _normalize_symbols(raw: object) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []

    def _add(val: object) -> None:
        sym = str(val or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    if isinstance(raw, str):
        cleaned = raw.replace("\n", ",")
        for part in cleaned.split(","):
            _add(part)
        return symbols
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        for part in raw:
            _add(part)
        return symbols
    if raw is not None:
        _add(raw)
    return symbols


def _load_favorites(db=None) -> list[str]:
    close_gen = False
    gen = None
    if db is None:
        gen = get_db()
        db = next(gen)
        close_gen = True
    try:
        rows = db.execute("SELECT ticker FROM favorites ORDER BY id").fetchall()
        raw = []
        for row in rows:
            if hasattr(row, "keys"):
                raw.append(row["ticker"])
            else:
                raw.append(row[0])
        return _normalize_symbols(raw)
    finally:
        if close_gen and gen is not None:
            gen.close()


def resolve_overnight_symbols(payload: dict, db=None) -> UniverseResolution:
    params: dict = {}
    settings = payload.get("settings")
    if isinstance(settings, dict):
        params.update(settings)
    params.update(payload)

    scan_type_raw = str(params.get("scan_type") or "").strip()
    scan_type = scan_type_raw.lower()
    canonical = _UNIVERSE_ALIASES.get(scan_type, scan_type)
    ticker = str(params.get("ticker") or "").strip().upper()

    raw_universe = params.get("universe")
    symbols: list[str] = []

    if isinstance(raw_universe, str) and raw_universe.strip().lower() in {"favorites"}:
        canonical = "favorites"
        symbols = _load_favorites(db)
    elif raw_universe not in (None, "", []):
        symbols = _normalize_symbols(raw_universe)
        if not canonical:
            canonical = "custom"
    elif canonical == "single":
        symbols = _normalize_symbols([ticker] if ticker else [])
    elif canonical == "favorites":
        symbols = _load_favorites(db)
    elif canonical == "sp100":
        symbols = _normalize_symbols(SP100)
    elif canonical == "scan250":
        symbols = _normalize_symbols(TOP250)
    else:
        canonical = canonical or "scan150"
        symbols = _normalize_symbols(TOP150)

    symbols_total = len(symbols)
    message = ""
    detail = ""
    if symbols_total == 0:
        label = canonical or (scan_type or "custom")
        if label == "single" and ticker:
            detail = f"Ticker {ticker} resolved to 0 symbols"
        else:
            detail = f"Universe {label} resolved to 0 symbols"
        message = "Universe resolves to 0 symbols"

    return UniverseResolution(
        symbols=symbols,
        universe=canonical or (scan_type_raw or "scan150"),
        ticker=ticker,
        symbols_total=symbols_total,
        message=message,
        detail=detail,
    )


class OvernightScanError(Exception):
    """Raised when an overnight scan fails to complete."""

    def __init__(self, message: str, *, meta: Optional[dict] = None) -> None:
        super().__init__(message)
        self.meta = meta or {}


def resolve_overnight_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the previous close and pre-market open window in ET."""

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_et = now.astimezone(TZ)
    last_close_utc = last_trading_close(now_et)
    last_close_et = last_close_utc.astimezone(TZ)
    next_day = last_close_et.date() + timedelta(days=1)
    while True:
        midday = datetime.combine(next_day, dt_time(12, 0), tzinfo=TZ)
        if market_is_open(midday):
            break
        next_day += timedelta(days=1)
    premarket_et = datetime.combine(next_day, PREMARKET_OPEN, tzinfo=TZ)
    window_end = premarket_et
    if now_et < last_close_et:
        window_end = now_et
    return last_close_et, window_end


def _parse_hhmm(s: str) -> dt_time:
    """Parse a HH:MM string into a ``time`` object."""
    hour, minute = [int(part) for part in s.split(":", 1)]
    return dt_time(hour=hour, minute=minute)


def in_window(now: datetime, start: str, end: str) -> bool:
    """Return True if ``now`` falls within the [start,end) window.

    Handles windows that wrap past midnight.
    """
    start_t = _parse_hhmm(start)
    end_t = _parse_hhmm(end)
    cur = now.time()
    if start_t <= end_t:
        return start_t <= cur < end_t
    return cur >= start_t or cur < end_t


_RUNNER_STATE = {"state": "idle", "batch_id": None, "position": None}


def get_runner_state() -> dict:
    return dict(_RUNNER_STATE)


class OvernightRunner:
    """Process overnight batches sequentially respecting run windows."""

    def __init__(self, scan_func: Callable[[dict, bool], int]) -> None:
        self._scan_func = scan_func
        self._recover()

    def _recover(self) -> None:
        """Mark in-flight items interrupted and reset batch statuses."""
        gen = get_db()
        db = next(gen)
        try:
            rows = db.execute(
                "SELECT id, batch_id, run_id FROM overnight_items WHERE status='running'",
            ).fetchall()
            for iid, bid, rid in rows:
                if rid is not None:
                    r = db.execute(
                        "SELECT finished_at FROM runs WHERE id=?", (rid,),
                    ).fetchone()
                    if r and r[0]:
                        db.execute(
                            "UPDATE overnight_items SET status='complete', finished_at=? WHERE id=?",
                            (r[0], iid),
                        )
                        db.execute(
                            "UPDATE overnight_batches SET status='queued', start_override=0 WHERE id=?",
                            (bid,),
                        )
                        continue
                db.execute(
                    "UPDATE overnight_items SET status='failed', error='interrupted', finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (iid,),
                )
                db.execute(
                    "UPDATE overnight_batches SET status='queued', start_override=0 WHERE id=?",
                    (bid,),
                )
            db.connection.commit()
        finally:
            gen.close()

    # --- scheduling helpers -------------------------------------------------
    def _get_window(self, db) -> tuple[str, str]:
        row = db.execute(
            "SELECT window_start, window_end FROM overnight_prefs WHERE id=1",
        ).fetchone()
        if not row:
            db.execute("INSERT INTO overnight_prefs(id) VALUES (1)")
            db.connection.commit()
            return "01:00", "08:00"
        return row[0], row[1]

    def run_ready(self, now: datetime) -> Optional[str]:
        """Run the next eligible batch if within window or override set."""
        gen = get_db()
        db = next(gen)
        try:
            row = db.execute(
                """
                SELECT id FROM overnight_batches
                WHERE start_override=1 AND status IN ('queued','paused') AND deleted_at IS NULL
                ORDER BY created_at
                LIMIT 1
                """,
            ).fetchone()
            if not row:
                start, end = self._get_window(db)
                if not in_window(now, start, end):
                    return None
                row = db.execute(
                    """
                    SELECT id FROM overnight_batches
                    WHERE status='queued' AND deleted_at IS NULL
                    ORDER BY created_at
                    LIMIT 1
                    """,
                ).fetchone()
                if not row:
                    return None
            batch_id = row[0]
            db.execute(
                "UPDATE overnight_batches SET status='running' WHERE id=?",
                (batch_id,),
            )
            db.connection.commit()
            _RUNNER_STATE.update({"state": "running", "batch_id": batch_id, "position": None})
        finally:
            gen.close()
        self.run_batch(batch_id)
        return batch_id

    # --- core execution -----------------------------------------------------
    def run_batch(self, batch_id: str) -> None:
        gen = get_db()
        db = next(gen)
        try:
            while True:
                batch_row = db.execute(
                    "SELECT status, start_override, deleted_at FROM overnight_batches WHERE id=?",
                    (batch_id,),
                ).fetchone()
                if not batch_row or batch_row[2]:
                    _RUNNER_STATE.update({"state": "idle", "batch_id": None, "position": None})
                    break
                if batch_row[0] in ("canceled", "paused"):
                    if batch_row[0] == "canceled":
                        db.execute(
                            "UPDATE overnight_batches SET start_override=0 WHERE id=?",
                            (batch_id,),
                        )
                        db.execute(
                            "UPDATE overnight_items SET status='canceled' WHERE batch_id=? AND status='queued'",
                            (batch_id,),
                        )
                        db.connection.commit()
                    _RUNNER_STATE.update({"state": "idle", "batch_id": None, "position": None})
                    break

                row = db.execute(
                    """
                    SELECT id, payload_json, created_at
                    FROM overnight_items
                    WHERE batch_id=? AND status='queued'
                    ORDER BY position
                    LIMIT 1
                    """,
                    (batch_id,),
                ).fetchone()
                if not row:
                    db.execute(
                        "UPDATE overnight_batches SET status='complete', start_override=0 WHERE id=?",
                        (batch_id,),
                    )
                    db.connection.commit()
                    prefs = self._get_window(db)
                    done = db.execute(
                        "SELECT COUNT(*) FROM overnight_items WHERE batch_id=?",
                        (batch_id,),
                    ).fetchone()[0]
                    failed = db.execute(
                        "SELECT COUNT(*) FROM overnight_items WHERE batch_id=? AND status='failed'",
                        (batch_id,),
                    ).fetchone()[0]
                    logger.info(
                        json.dumps(
                            {
                                "type": "overnight_batch",
                                "batch_id": batch_id,
                                "status": "complete",
                                "items_total": done,
                                "items_done": done - failed,
                                "items_failed": failed,
                                "window": f"{prefs[0]}-{prefs[1]}",
                            }
                        )
                    )
                    break

                item_id, payload_json, created_at = row
                payload = json.loads(payload_json)
                resolution = resolve_overnight_symbols(payload, db)
                log_context = {
                    "universe": resolution.universe,
                    "ticker": resolution.ticker,
                    "symbols_total": resolution.symbols_total,
                }
                now = datetime.utcnow()
                queue_wait = int((now - datetime.fromisoformat(created_at)).total_seconds() * 1000)
                if resolution.symbols_total == 0:
                    err_msg = resolution.detail or resolution.message or "Universe resolves to 0 symbols"
                    db.execute(
                        """
                        UPDATE overnight_items
                        SET status='failed', finished_at=?, error=?
                        WHERE id=?
                        """,
                        (now.isoformat(), err_msg, item_id),
                    )
                    db.connection.commit()
                    pos = db.execute(
                        "SELECT position FROM overnight_items WHERE id=?",
                        (item_id,),
                    ).fetchone()[0]
                    log_payload = {
                        "type": "overnight_item",
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "position": pos,
                        "status": "failed",
                        "error": err_msg,
                        "timings_ms": {"scan": 0, "queue_wait": queue_wait},
                    }
                    log_payload.update(log_context)
                    logger.error(json.dumps(log_payload))
                    continue

                db.execute(
                    "UPDATE overnight_items SET status='running', started_at=? WHERE id=?",
                    (now.isoformat(), item_id),
                )
                db.connection.commit()
                _RUNNER_STATE.update({"state": "running", "batch_id": batch_id, "position": db.execute("SELECT position FROM overnight_items WHERE id=?", (item_id,)).fetchone()[0]})

                t0 = time.perf_counter()
                scan_meta: dict = {}
                try:
                    result = self._scan_func(payload, True)
                    elapsed = int((time.perf_counter() - t0) * 1000)
                except OvernightScanError as exc:
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    scan_meta = exc.meta or {}
                    err_msg = str(exc)
                    db.execute(
                        """
                        UPDATE overnight_items
                        SET status='failed', finished_at=CURRENT_TIMESTAMP, error=?
                        WHERE id=?
                        """,
                        (err_msg, item_id),
                    )
                    db.connection.commit()
                    pos = db.execute(
                        "SELECT position FROM overnight_items WHERE id=?",
                        (item_id,),
                    ).fetchone()[0]
                    log_payload = {
                        "type": "overnight_item",
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "position": pos,
                        "status": "failed",
                        "error": err_msg,
                        "timings_ms": {"scan": elapsed, "queue_wait": queue_wait},
                    }
                    log_payload.update(log_context)
                    if scan_meta:
                        meta_log = {k: v for k, v in scan_meta.items() if k != "run_id"}
                        if meta_log:
                            log_payload["meta"] = meta_log
                    logger.error(json.dumps(log_payload))
                    continue
                except Exception as exc:
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    err_msg = repr(exc)
                    db.execute(
                        """
                        UPDATE overnight_items
                        SET status='failed', finished_at=CURRENT_TIMESTAMP, error=?
                        WHERE id=?
                        """,
                        (err_msg, item_id),
                    )
                    db.connection.commit()
                    pos = db.execute(
                        "SELECT position FROM overnight_items WHERE id=?",
                        (item_id,),
                    ).fetchone()[0]
                    log_payload = {
                        "type": "overnight_item",
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "position": pos,
                        "status": "failed",
                        "error": err_msg,
                        "timings_ms": {"scan": elapsed, "queue_wait": queue_wait},
                    }
                    log_payload.update(log_context)
                    logger.exception(
                        "overnight scan unexpected error batch=%s item=%s universe=%s symbols=%d ticker=%s",
                        batch_id,
                        item_id,
                        log_context.get("universe"),
                        log_context.get("symbols_total"),
                        log_context.get("ticker"),
                    )
                    logger.error(json.dumps(log_payload))
                    continue

                if isinstance(result, dict):
                    run_id = result.get("run_id")
                    scan_meta = result
                else:
                    run_id = result
                    scan_meta = {"run_id": run_id}

                if run_id is None:
                    err_msg = "Scan returned no run_id"
                    db.execute(
                        """
                        UPDATE overnight_items
                        SET status='failed', finished_at=CURRENT_TIMESTAMP, error=?
                        WHERE id=?
                        """,
                        (err_msg, item_id),
                    )
                    db.connection.commit()
                    pos = db.execute(
                        "SELECT position FROM overnight_items WHERE id=?",
                        (item_id,),
                    ).fetchone()[0]
                    log_payload = {
                        "type": "overnight_item",
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "position": pos,
                        "status": "failed",
                        "error": err_msg,
                        "timings_ms": {"scan": elapsed, "queue_wait": queue_wait},
                    }
                    log_payload.update(log_context)
                    if scan_meta:
                        meta_log = {k: v for k, v in scan_meta.items() if k != "run_id"}
                        if meta_log:
                            log_payload["meta"] = meta_log
                    logger.error(json.dumps(log_payload))
                    continue

                db.execute(
                    """
                    UPDATE overnight_items
                    SET status='complete', finished_at=CURRENT_TIMESTAMP, run_id=?, error=NULL
                    WHERE id=?
                    """,
                    (run_id, item_id),
                )
                db.connection.commit()

                pos = db.execute(
                    "SELECT position FROM overnight_items WHERE id=?",
                    (item_id,),
                ).fetchone()[0]
                log_payload = {
                    "type": "overnight_item",
                    "batch_id": batch_id,
                    "item_id": item_id,
                    "position": pos,
                    "status": "complete",
                    "run_id": run_id,
                    "silent": True,
                    "timings_ms": {"scan": elapsed, "queue_wait": queue_wait},
                }
                log_payload.update(log_context)
                if scan_meta:
                    meta_log = {k: v for k, v in scan_meta.items() if k != "run_id"}
                    if meta_log:
                        log_payload["meta"] = meta_log
                logger.info(json.dumps(log_payload))

                start, end = self._get_window(db)
                if not in_window(datetime.utcnow(), start, end) and not batch_row[1]:
                    db.execute(
                        "UPDATE overnight_batches SET status='paused', start_override=0 WHERE id=?",
                        (batch_id,),
                    )
                    db.connection.commit()
                    logger.info(
                        "overnight window closed; pausing batch=%s after current item", batch_id
                    )
                    _RUNNER_STATE.update({"state": "idle", "batch_id": None, "position": None})
                    break
        finally:
            gen.close()
            if _RUNNER_STATE["state"] == "running":
                _RUNNER_STATE.update({"state": "idle", "batch_id": None, "position": None})


def start_background_runner(scan_func: Callable[[dict, bool], int]) -> OvernightRunner:
    runner = OvernightRunner(scan_func)

    def _loop() -> None:
        while True:
            runner.run_ready(datetime.utcnow())
            time.sleep(5)

    Thread(target=_loop, daemon=True).start()
    gen = get_db()
    db = next(gen)
    try:
        start, end = runner._get_window(db)
    finally:
        gen.close()
    logger.info("overnight runner online window=%s-%s", start, end)
    return runner

# ruff: noqa: E501
import asyncio
import atexit
import json
import logging
import os
import sqlite3
import statistics
import time
from concurrent.futures import as_completed
from datetime import datetime, timedelta
from threading import Thread
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Body, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import settings
from db import (
    DB_PATH,
    _ensure_scanner_column,
    get_db,
    get_schema_status,
    get_settings,
    row_to_dict,
)
from indices import SP100, TOP150, TOP250
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from scanner import compute_scan_for_ticker
from services import executor, favorites_alerts, price_store
from services.notify import send_email_smtp
from services.scanner_params import coerce_scan_params
from services.telemetry import log as log_telemetry
from services.market_data import (
    expected_bar_count,
    fetch_prices,
    get_prices,
    window_from_lookback,
)
from utils import TZ, now_et

from .archive import _format_rule_summary as _format_rule_summary
from .archive import router as archive_router
from .overnight import router as overnight_router

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)
SCAN_BATCH_WRITES = os.getenv("SCAN_BATCH_WRITES", "1") not in {"0", "false", "no"}

_perf_counter = time.perf_counter


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_support_snapshot(raw: Any) -> tuple[int | None, dict[str, Any] | None]:
    data: dict[str, Any] | None = None
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
    elif isinstance(raw, dict):
        data = raw

    if isinstance(data, dict):
        support_val = data.get("count")
        if support_val is None:
            support_val = data.get("support")
        return _coerce_int(support_val), data

    return _coerce_int(raw), None


def _format_lookback(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return f"{int(rounded)}y"
    return f"{value:.1f}y"


def _normalize_favorite(row: dict[str, Any]) -> dict[str, Any]:
    support_count, support_snapshot = _parse_support_snapshot(
        row.get("support_snapshot")
    )
    lookback = _coerce_float(row.get("lookback_years"))
    if lookback is None and isinstance(support_snapshot, dict):
        lookback = _coerce_float(support_snapshot.get("lookback_years"))

    min_support = _coerce_int(row.get("min_support"))
    if support_count is None:
        support_count = min_support

    row["lookback_years"] = lookback
    row["support_count"] = support_count
    row["lookback_display"] = _format_lookback(lookback)
    row["support_display"] = (
        str(support_count) if support_count is not None else "—"
    )
    return row


_SMTP_FIELD_LABELS = {
    "smtp_host": "SMTP host",
    "smtp_port": "SMTP port",
    "smtp_user": "SMTP user",
    "smtp_pass": "SMTP password",
    "mail_from": "Mail From",
    "recipients": "Recipients",
}


def _smtp_config_status(settings_row: dict) -> tuple[dict[str, Any], list[str], bool]:
    """Normalize SMTP settings and report missing fields."""

    host = (settings_row.get("smtp_host") or "").strip()
    try:
        port = int(settings_row.get("smtp_port") or 0)
    except (TypeError, ValueError):
        port = 0
    user = (settings_row.get("smtp_user") or "").strip()
    password = (settings_row.get("smtp_pass") or "").strip()
    mail_from = (settings_row.get("mail_from") or "").strip()
    recipients = [
        r.strip()
        for r in (settings_row.get("recipients") or "").split(",")
        if r.strip()
    ]

    missing: list[str] = []
    if not host:
        missing.append("smtp_host")
    if port <= 0:
        missing.append("smtp_port")
    if not user:
        missing.append("smtp_user")
    if not password:
        missing.append("smtp_pass")
    if not mail_from:
        missing.append("mail_from")
    if not recipients:
        missing.append("recipients")

    has_any = any([host, user, password, mail_from]) or bool(recipients)

    cfg = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "mail_from": mail_from,
        "recipients": recipients,
    }
    return cfg, missing, has_any


def _format_missing_fields(missing: list[str]) -> str:
    labels = [_SMTP_FIELD_LABELS.get(field, field) for field in missing]
    return ", ".join(labels)



def _extract_test_alert_message(result: Any) -> tuple[bool, str, str]:
    ok = False
    subject = ""
    body = ""
    payload: Optional[Any] = None
    if isinstance(result, tuple):
        ok = bool(result[0])
        if len(result) > 1:
            payload = result[1]
    else:
        ok = bool(result)
        if ok:
            payload = result
    if isinstance(payload, dict):
        subject = str(payload.get("subject") or "")
        body = str(payload.get("body") or "")
    elif isinstance(payload, str):
        body = payload
    elif payload is not None:
        body = str(payload)
    return ok, subject, body

def _get_next_earnings(ticker: str):  # pragma: no cover - external service
    return None


def _get_adv(ticker: str):  # pragma: no cover - external service
    return None


def check_guardrails(
    ticker: str,
    *,
    earnings_window: int = 7,
    adv_threshold: float = 1_000_000.0,
    get_earnings: Callable[[str], pd.Timestamp | None] | None = None,
    get_adv: Callable[[str], float | None] | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate guardrails for a forward test.

    Returns ``(True, [])`` when the test may proceed. When guardrails trigger,
    returns ``(False, flags)`` where ``flags`` describes the failing rules.
    ``earnings`` is flagged when within ``earnings_window`` trading days of a
    known earnings date.  ``low_liquidity`` is flagged when average daily volume
    falls below ``adv_threshold``.
    """

    get_earnings = get_earnings or _get_next_earnings
    get_adv = get_adv or _get_adv

    flags: list[str] = []
    today = now_et().date()

    try:
        edate = get_earnings(ticker)
        if edate is not None:
            edate = pd.Timestamp(edate).date()
            diff = abs(len(pd.bdate_range(today, edate)) - 1)
            if diff <= earnings_window:
                flags.append("earnings")
    except Exception:
        logger.exception("earnings guardrail failed ticker=%s", ticker)

    try:
        adv = get_adv(ticker)
        if adv is not None and adv < adv_threshold:
            flags.append("low_liquidity")
    except Exception:
        logger.exception("liquidity guardrail failed ticker=%s", ticker)

    return (not flags, flags)


router.include_router(archive_router)
router.include_router(overnight_router)


@router.get("/history")
def history_redirect():
    """Redirect legacy history link to archive."""
    return RedirectResponse(url="/archive", status_code=308)


scan_duration = Histogram("scan_duration_seconds", "Duration of /scanner/run requests")
scan_tickers = Counter("scan_tickers_total", "Tickers processed by /scanner/run")
coverage_symbols_total = Counter(
    "coverage_symbols_total", "Symbols processed during coverage checks"
)
coverage_symbols_no_gap = Counter(
    "coverage_symbols_no_gap", "Symbols with full local coverage"
)
coverage_symbols_gap_fetched = Counter(
    "coverage_symbols_gap_fetched", "Symbols fetched due to coverage gaps"
)
coverage_elapsed_seconds = Histogram(
    "coverage_elapsed_seconds", "Time spent performing bulk coverage checks"
)


def healthz() -> dict:
    """Simple health check used by tests and the /health endpoint."""
    return {"status": "ok"}


@router.get("/health")
def health() -> dict:
    """Return app health along with schema status information."""
    return {**healthz(), **get_schema_status()}


def metrics() -> Response:
    """Expose Prometheus metrics used by tests and the /metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if settings.metrics_enabled:

    @router.get("/metrics")
    def metrics_endpoint() -> Response:
        return metrics()


_TASK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scan_tasks (
    id TEXT PRIMARY KEY,
    total INTEGER,
    done INTEGER,
    percent REAL,
    state TEXT,
    message TEXT,
    ctx TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);
"""


def _ensure_task_columns(conn: sqlite3.Connection) -> None:
    """Ensure ``scan_tasks`` has ``started_at``/``updated_at`` columns.

    Older databases created before these columns existed should still work.  We
    perform a lightweight, idempotent check on each connection and add the
    missing columns if needed.  Existing rows are backfilled so callers relying
    on the timestamps do not crash.
    """

    cur = conn.execute("PRAGMA table_info(scan_tasks)")
    cols = {r[1] for r in cur.fetchall()}
    altered = False
    if "started_at" not in cols:
        _execute_write(conn, "ALTER TABLE scan_tasks ADD COLUMN started_at TEXT")
        altered = True
    if "updated_at" not in cols:
        _execute_write(conn, "ALTER TABLE scan_tasks ADD COLUMN updated_at TEXT")
        altered = True
    if altered:
        _execute_write(
            "UPDATE scan_tasks SET started_at=COALESCE(started_at, CURRENT_TIMESTAMP), "
            "updated_at=COALESCE(updated_at, started_at)"
        )
        conn.commit()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _execute_write(conn, _TASK_TABLE_SQL)
    conn.commit()
    try:
        _ensure_task_columns(conn)
    except Exception:
        # best effort; existing scans may proceed even if this fails
        pass
    return conn


_TASK_MEM: dict[str, dict[str, Any]] = {}
_TASK_WRITE_TS: dict[str, float] = {}


def _execute_write(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()):
    assert "bars" not in sql.lower(), "writes to bars table are not allowed"
    return conn.execute(sql, params)


def _task_create(task_id: str, total: int) -> None:
    _task_gc()
    conn = _get_conn()
    try:
        now_iso = now_et().isoformat()
        _execute_write(
            conn,
            (
                "INSERT INTO scan_tasks (id, total, done, percent, state, started_at, updated_at) "
                "VALUES (?, ?, 0, 0.0, 'queued', ?, ?)"
            ),
            (task_id, total, now_iso, now_iso),
        )
        conn.commit()
    finally:
        conn.close()
    _TASK_MEM[task_id] = {
        "total": total,
        "done": 0,
        "percent": 0.0,
        "state": "queued",
        "message": "",
        "ctx": None,
        "started_at": now_iso,
        "updated_at": now_iso,
    }
    _TASK_WRITE_TS[task_id] = time.monotonic()


def _task_update_db(task_id: str, fields: dict[str, Any]) -> None:
    if not fields:
        return
    conn = _get_conn()
    try:
        cols = ["updated_at=?"]
        vals: list[Any] = [now_et().isoformat()]
        for k, v in fields.items():
            cols.append(f"{k}=?")
            if k == "ctx" and v is not None:
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        vals.append(task_id)
        _execute_write(
            conn, f"UPDATE scan_tasks SET {', '.join(cols)} WHERE id=?", tuple(vals)
        )
        conn.commit()
    finally:
        conn.close()


def _task_update(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    now_iso = now_et().isoformat()
    task = _TASK_MEM.get(task_id)
    if task is not None:
        task.update(fields)
        task["updated_at"] = now_iso
    last = _TASK_WRITE_TS.get(task_id, 0.0)
    need_persist = False
    items = settings.scan_progress_flush_items
    interval = settings.scan_status_flush_ms / 1000.0
    if task is not None:
        done = task.get("done")
        if isinstance(done, int) and items > 0 and done % items == 0:
            need_persist = True
    if time.monotonic() - last >= interval:
        need_persist = True
    if fields.get("state") in {"succeeded", "failed"}:
        need_persist = True
    if need_persist:
        _task_update_db(task_id, task if task is not None else fields)
        _TASK_WRITE_TS[task_id] = time.monotonic()


def _task_get(task_id: str) -> Optional[Dict[str, Any]]:
    task = _TASK_MEM.get(task_id)
    if task is not None:
        return task
    conn = _get_conn()
    try:
        cur = conn.execute(
            (
                "SELECT total, done, percent, state, message, ctx, started_at, updated_at "
                "FROM scan_tasks WHERE id=?"
            ),
            (task_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    total, done, percent, state, message, ctx_json, started_at, updated_at = row
    task = {
        "total": total,
        "done": done,
        "percent": percent,
        "state": state,
        "message": message,
        "ctx": json.loads(ctx_json) if ctx_json else None,
        "started_at": started_at,
        "updated_at": updated_at,
    }
    _TASK_MEM[task_id] = task
    _TASK_WRITE_TS[task_id] = time.monotonic()
    return task


def _task_delete(task_id: str) -> None:
    conn = _get_conn()
    try:
        _execute_write(conn, "DELETE FROM scan_tasks WHERE id=?", (task_id,))
        conn.commit()
    finally:
        conn.close()
    _TASK_MEM.pop(task_id, None)
    _TASK_WRITE_TS.pop(task_id, None)


def _task_gc(ttl_hours: int = 48) -> None:
    cutoff = (now_et() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn()
    try:
        _execute_write(conn, "DELETE FROM scan_tasks WHERE started_at < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()
    to_drop = [tid for tid, t in _TASK_MEM.items() if t.get("started_at", "") < cutoff]
    for tid in to_drop:
        _TASK_MEM.pop(tid, None)
        _TASK_WRITE_TS.pop(tid, None)


def _task_flush_all() -> None:
    """Persist all in-memory task states to the database."""
    for tid, task in list(_TASK_MEM.items()):
        try:
            _task_update_db(tid, task)
        except Exception:
            pass


atexit.register(_task_flush_all)


def _scan_single(t: str, params: dict) -> tuple[dict | None, float]:
    t0 = _perf_counter()
    res = compute_scan_for_ticker(t, params)
    return res, _perf_counter() - t0


def _scan_chunk(ts: list[str], params: dict) -> list[tuple[str, dict | None, float]]:
    pc = _perf_counter
    out: list[tuple[str, dict | None, float]] = []
    for t in ts:
        t0 = pc()
        try:
            res = compute_scan_for_ticker(t, params)
        except Exception as e:
            logger.error("scan failed for %s: %s", t, e)
            res = {}
        out.append((t, res, pc() - t0))
    return out


def _perform_scan(
    tickers: list[str],
    params: dict,
    sort_key: str,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    progress_every: int = 5,
) -> tuple[list[dict], int, dict]:
    start = _perf_counter()
    total = len(tickers)
    interval = params.get("interval", "15m")
    lookback = float(params.get("lookback_years", 2.0))
    window_start, window_end = window_from_lookback(lookback)

    cov_start = _perf_counter()
    cov: dict[str, tuple[datetime, datetime, int]] = {}
    batch = max(1, settings.scan_coverage_batch_size)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        cov.update(price_store.bulk_coverage(chunk, interval, window_start, window_end))
    expected = expected_bar_count(window_start, window_end, interval)
    no_gap: list[str] = []
    need_fetch: list[str] = []
    for sym in tickers:
        cmin, cmax, cnt = cov.get(sym, (None, None, 0))
        if cnt >= expected and price_store.covers(window_start, window_end, cmin, cmax):
            no_gap.append(sym)
        else:
            need_fetch.append(sym)
    cov_elapsed = (_perf_counter() - cov_start) * 1000.0
    coverage_symbols_total.inc(total)
    coverage_symbols_no_gap.inc(len(no_gap))
    coverage_symbols_gap_fetched.inc(len(need_fetch))
    coverage_elapsed_seconds.observe(cov_elapsed / 1000.0)
    logger.info(
        "coverage_mode=batch symbols=%d rows=%d elapsed_ms=%.0f",
        total,
        len(cov),
        cov_elapsed,
    )

    fetch_elapsed = 0.0
    if need_fetch:

        async def _fetch_all(symbols: list[str]) -> None:
            sem = asyncio.Semaphore(settings.scan_fetch_concurrency)

            async def _worker(sym: str) -> None:
                async with sem:
                    logger.debug("fetch_gap symbol=%s", sym)
                    await asyncio.to_thread(fetch_prices, [sym], interval, lookback)

            await asyncio.gather(*(_worker(s) for s in symbols))

        fetch_start = _perf_counter()
        asyncio.run(_fetch_all(need_fetch))
        fetch_elapsed = (_perf_counter() - fetch_start) * 1000.0

    if progress_cb:
        progress_cb(0, total, "preloading")

    rows: list[dict] = []
    skipped_missing_data = 0
    ex = executor.EXECUTOR
    chunk_size = max(1, settings.scan_symbols_per_task)
    futures = [
        ex.submit(_scan_chunk, tickers[i : i + chunk_size], params)
        for i in range(0, len(tickers), chunk_size)
    ]
    logger.info(
        "scan dispatch mode=%s workers=%d symbols_per_task=%d tasks=%d",
        executor.MODE,
        executor.WORKERS,
        chunk_size,
        len(futures),
    )
    step = max(1, int(progress_every))
    times: list[float] = []
    buffer: list[dict] = []
    seen: set[str] = set()
    processed = 0
    for fut in as_completed(futures):
        results = fut.result()
        for ticker, r, elapsed in results:
            processed += 1
            times.append(elapsed)
            if r is None:
                skipped_missing_data += 1
            elif r and ticker not in seen:
                seen.add(ticker)
                if SCAN_BATCH_WRITES:
                    buffer.append(r)
                    if len(buffer) >= 50:
                        rows.extend(buffer)
                        buffer.clear()
                else:
                    rows.append(r)
            if progress_cb and (processed % step == 0 or processed == total):
                progress_cb(processed, total, f"Scanning {processed}/{total}")
            if processed % 10 == 0:
                el = _perf_counter() - start
                logger.info(
                    "scan progress %d/%d elapsed=%.1fs avg_per_task=%.2fs",
                    processed,
                    total,
                    el,
                    el / processed,
                )
    if buffer:
        rows.extend(buffer)
    logger.info(
        "scan complete %d symbols in %.1fs",
        total,
        _perf_counter() - start,
    )

    try:
        scan_min_hit = float(params.get("scan_min_hit", 0.0))
        scan_max_dd = float(params.get("scan_max_dd", 100.0))
    except Exception:
        scan_min_hit, scan_max_dd = 0.0, 100.0

    rows = [
        r
        for r in rows
        if (r.get("hit_pct", 0.0) >= scan_min_hit)
        and (r.get("avg_dd_pct", 100.0) <= scan_max_dd)
    ]

    if sort_key == "ticker":
        rows.sort(key=lambda r: (r.get("ticker") or ""))
    elif sort_key == "roi":
        rows.sort(
            key=lambda r: (
                r.get("avg_roi_pct", 0.0),
                r.get("hit_pct", 0.0),
                r.get("support", 0),
            ),
            reverse=True,
        )
    elif sort_key == "hit":
        rows.sort(
            key=lambda r: (
                r.get("hit_pct", 0.0),
                r.get("avg_roi_pct", 0.0),
                r.get("support", 0),
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda r: (
                r.get("avg_roi_pct", 0.0),
                r.get("hit_pct", 0.0),
                r.get("support", 0),
                r.get("stability", 0.0),
            ),
            reverse=True,
        )

    duration = _perf_counter() - start
    scan_duration.observe(duration)
    scan_tickers.inc(len(tickers))
    avg_ms = (statistics.mean(times) * 1000.0) if times else 0.0
    p95_ms = (
        sorted(times)[max(int(len(times) * 0.95) - 1, 0)] * 1000.0 if times else 0.0
    )
    metrics = {
        "coverage_ms": cov_elapsed,
        "fetch_ms": fetch_elapsed,
        "symbols_no_gap": len(no_gap),
        "symbols_gap": len(need_fetch),
        "avg_per_symbol_ms": avg_ms,
        "p95_per_symbol_ms": p95_ms,
        "db_reads": 1,
        "db_writes": 0,
    }
    logger.info(
        "scan completed total=%d no_gap=%d gaps=%d avg_ms=%.1f p95_ms=%.1f db_reads=%d db_writes=%d skipped_missing_data=%d",
        len(tickers),
        len(no_gap),
        len(need_fetch),
        avg_ms,
        p95_ms,
        metrics["db_reads"],
        metrics["db_writes"],
        skipped_missing_data,
    )

    return rows, skipped_missing_data, metrics


def _sort_rows(rows, sort_key):
    if not rows or not sort_key:
        return rows
    keymap = {
        "ticker": lambda r: (r.get("ticker") or ""),
        "roi": lambda r: (r.get("avg_roi_pct") or 0.0),
        "hit": lambda r: (r.get("hit_pct") or 0.0),
    }
    keyfn = keymap.get(sort_key)
    if not keyfn:
        return rows
    reverse = sort_key != "ticker"
    return sorted(rows, key=keyfn, reverse=reverse)


def _send_email(
    st: dict,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    *,
    list_field: str = "recipients",
    allow_sms: bool = True,
) -> None:
    """Send an email using settings stored in the database.

    This helper mirrors the logic used by the desktop application: the
    configured SMTP user/password are expected to work with Gmail.  The
    function silently returns if mandatory settings are missing so the
    scanner can proceed without failing.
    """

    cfg, missing, has_any = _smtp_config_status(st)
    recips = cfg["recipients"] if list_field == "recipients" else [
        r.strip()
        for r in (st.get(list_field) or "").split(",")
        if r.strip()
    ]
    if not allow_sms:
        from services.notifications import is_carrier_address

        recips = [r for r in recips if not is_carrier_address(r)]
    if list_field != "recipients":
        missing = [m for m in missing if m != "recipients"]

    if not recips:
        if list_field == "recipients":
            missing = [m for m in missing if m != "recipients"] + ["recipients"]
        elif not has_any:
            return
        else:
            missing = missing + [list_field]

    if missing:
        if has_any or list_field != "recipients":
            logger.warning(
                "SMTP configuration incomplete: %s",
                _format_missing_fields(sorted(set(missing))),
            )
        return

    payload = html_body or body
    result = send_email_smtp(
        cfg["host"],
        cfg["port"],
        cfg["user"],
        cfg["password"],
        cfg["mail_from"] or cfg["user"],
        recips,
        subject,
        payload,
    )
    if not result.get("ok"):
        logger.warning("SMTP send failed: %s", result.get("error"))
        return
    logger.info(
        "email sent via smtp list=%s recipients=%d message_id=%s",
        list_field,
        len(recips),
        result.get("message_id"),
    )


def compile_weekly_digest(
    db: sqlite3.Cursor, ts: Optional[pd.Timestamp] = None
) -> tuple[str, str]:
    ts = pd.Timestamp(ts or now_et())
    week_start = (ts - pd.Timedelta(days=ts.weekday())).date()
    start_iso = pd.Timestamp(week_start, tz=TZ).isoformat()

    db.execute("SELECT COUNT(*) FROM forward_tests WHERE created_at>=?", (start_iso,))
    new_count = db.fetchone()[0]
    db.execute(
        "SELECT COUNT(*) FROM forward_tests WHERE status IN ('target','stop','expired') AND updated_at>=?",
        (start_iso,),
    )
    resolved_count = db.fetchone()[0]
    db.execute(
        "SELECT AVG(hit_forward), AVG(roi_forward) FROM forward_tests WHERE created_at>=?",
        (start_iso,),
    )
    hit_avg, roi_avg = db.fetchone()
    hit_avg = hit_avg or 0.0
    roi_avg = roi_avg or 0.0

    subject = (
        f"[Forward Digest] Week of {week_start:%Y-%m-%d} – {new_count} New | {resolved_count} Resolved | "
        f"Hit% {hit_avg:.0f} | Avg ROI {roi_avg:.1f}%"
    )

    db.execute(
        "SELECT reason, COUNT(*) FROM guardrail_skips WHERE created_at>=? GROUP BY reason",
        (start_iso,),
    )
    guard = {r[0]: r[1] for r in db.fetchall()}

    lines = [
        f"New tests: {new_count}",
        f"Resolved tests: {resolved_count}",
        f"Hit%: {hit_avg:.0f}",
        f"Avg ROI: {roi_avg:.1f}%",
        f"Guardrails - earnings: {guard.get('earnings',0)}, low_liquidity: {guard.get('low_liquidity',0)}",
        "Top Winners:",
    ]
    db.execute(
        "SELECT ticker, roi_forward FROM forward_tests WHERE created_at>=? ORDER BY roi_forward DESC LIMIT 3",
        (start_iso,),
    )
    for sym, roi in db.fetchall():
        lines.append(f"  {sym} {roi or 0:.1f}%")
    lines.append("Top Losers:")
    db.execute(
        "SELECT ticker, dd_forward FROM forward_tests WHERE created_at>=? ORDER BY dd_forward DESC LIMIT 3",
        (start_iso,),
    )
    for sym, dd in db.fetchall():
        lines.append(f"  {sym} {dd or 0:.1f}% DD")
    lines.append("Link: /forward?filter=last7d")
    body = "\n".join(lines)
    return subject, body


def send_weekly_digest(db: sqlite3.Cursor) -> None:
    st = get_settings(db)
    subject, body = compile_weekly_digest(db)
    _send_email(st, subject, body)


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active_tab": "scanner"})


@router.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active_tab": "scanner"})


@router.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request, db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [_normalize_favorite(row_to_dict(r, db)) for r in db.fetchall()]
    for f in favs:
        f["avg_roi_pct"] = f.get("roi_snapshot")
        f["hit_pct"] = f.get("hit_pct_snapshot")
        f["avg_dd_pct"] = f.get("dd_pct_snapshot")
        if f.get("rule_snapshot"):
            f["rule"] = f.get("rule_snapshot")
    return templates.TemplateResponse(
        request,
        "favorites.html",
        {"favorites": favs, "active_tab": "favorites"},
    )


def _window_to_minutes(value: float, unit: str) -> int:
    unit = (unit or "").lower()
    if unit.startswith("min"):
        return int(value)
    if unit.startswith("hour"):
        return int(value * 60)
    if unit.startswith("day"):
        return int(value * 60 * 24)
    if unit.startswith("week"):
        return int(value * 60 * 24 * 7)
    return int(value * 60)


def _create_forward_test(db: sqlite3.Cursor, fav: dict) -> None:
    allowed, flags = check_guardrails(fav.get("ticker"))
    if not allowed:
        logger.info(
            "forward test for %s skipped due to guardrails: %s",
            fav.get("ticker"),
            ",".join(flags),
        )
        db.execute(
            "INSERT INTO guardrail_skips(ticker, reason) VALUES (?, ?)",
            (fav.get("ticker"), ",".join(flags)),
        )
        return

    start, end = window_from_lookback(fav.get("lookback_years", 1.0))
    data = get_prices([fav["ticker"]], fav.get("interval", "15m"), start, end).get(
        fav["ticker"]
    )
    if data is None or getattr(data, "empty", True):
        return
    last_bar = data.iloc[-1]
    ts = last_bar.name
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    entry_ts = ts.astimezone(TZ).isoformat()
    entry_price = float(last_bar["Close"])
    window_minutes = _window_to_minutes(
        fav.get("window_value", 4.0), fav.get("window_unit", "Hours")
    )

    db.execute(
        """SELECT version, target_pct, stop_pct, window_minutes, rule
            FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1""",
        (fav["id"],),
    )
    row = db.fetchone()
    version = 1
    if row:
        version = row[0]
        if (
            float(row[1]) != float(fav.get("target_pct", 1.0))
            or float(row[2]) != float(fav.get("stop_pct", 0.5))
            or int(row[3]) != window_minutes
            or (row[4] or "") != fav.get("rule")
        ):
            db.execute(
                "UPDATE forward_tests SET status='closed' WHERE fav_id=? AND version=?",
                (fav["id"], row[0]),
            )
            version = row[0] + 1

    now_iso = now_et().isoformat()
    db.execute(
        """INSERT INTO forward_tests
            (fav_id, ticker, direction, interval, rule, version, entry_price,
             target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward, dd_forward,
             roi_1, roi_3, roi_5, roi_expiry, mae, mfe, time_to_hit, time_to_stop,
             option_expiry, option_strike, option_delta, option_roi_proxy,
             last_run_at, next_run_at, runs_count, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0.0, NULL, 0.0,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, ?, 0.0,
                    NULL, NULL, 0, NULL, ?, ?)""",
        (
            fav["id"],
            fav["ticker"],
            fav.get("direction", "UP"),
            fav.get("interval", "15m"),
            fav.get("rule"),
            version,
            entry_price,
            fav.get("target_pct", 1.0),
            fav.get("stop_pct", 0.5),
            window_minutes,
            fav.get("delta"),
            entry_ts,
            now_iso,
        ),
    )
    db.connection.commit()


def _update_forward_tests(db: sqlite3.Cursor) -> None:
    db.execute(
        """SELECT id, ticker, direction, interval, created_at, entry_price,
                  target_pct, stop_pct, window_minutes, status, option_delta
               FROM forward_tests
               WHERE status IN ('queued','running')"""
    )
    rows = [row_to_dict(r, db) for r in db.fetchall()]
    for row in rows:
        now_iso = now_et().isoformat()
        try:
            db.execute(
                "UPDATE forward_tests SET status='running', last_run_at=?, updated_at=?, runs_count=runs_count+1 WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
            start, end = window_from_lookback(1.0)
            data = get_prices([row["ticker"]], row["interval"], start, end).get(
                row["ticker"]
            )
            if data is None or getattr(data, "empty", True):
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            entry_ts = pd.Timestamp(row["created_at"])
            after = data[data.index > entry_ts]
            if after.empty:
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            prices = after["Close"]
            highs = after["High"]
            lows = after["Low"]
            mult = 1.0 if row["direction"] == "UP" else -1.0
            pct_series = (prices / row["entry_price"] - 1.0) * 100 * mult
            roi_curve = {1: None, 3: None, 5: None}
            for n in roi_curve:
                if len(pct_series) >= n:
                    roi_curve[n] = float(pct_series.iloc[n - 1])
            expire_ts = entry_ts + pd.Timedelta(minutes=row["window_minutes"])
            expiry_prices = prices[prices.index <= expire_ts]
            roi_expiry = (
                float((expiry_prices.iloc[-1] / row["entry_price"] - 1.0) * 100 * mult)
                if not expiry_prices.empty
                else None
            )
            if row["direction"] == "UP":
                hit_cond = highs >= row["entry_price"] * (1 + row["target_pct"] / 100)
                stop_cond = lows <= row["entry_price"] * (1 - row["stop_pct"] / 100)
                mae_series = (lows / row["entry_price"] - 1.0) * 100 * mult
                mfe_series = (highs / row["entry_price"] - 1.0) * 100 * mult
            else:
                hit_cond = lows <= row["entry_price"] * (1 - row["target_pct"] / 100)
                stop_cond = highs >= row["entry_price"] * (1 + row["stop_pct"] / 100)
                mae_series = (highs / row["entry_price"] - 1.0) * 100 * mult
                mfe_series = (lows / row["entry_price"] - 1.0) * 100 * mult
            hit_time = hit_cond[hit_cond].index[0] if hit_cond.any() else None
            stop_time = stop_cond[stop_cond].index[0] if stop_cond.any() else None
            mae = float(mae_series.min()) if not mae_series.empty else 0.0
            mfe = float(mfe_series.max()) if not mfe_series.empty else 0.0
            roi = float(pct_series.iloc[-1]) if not pct_series.empty else 0.0
            status = "ok"
            hit_pct = None
            event_time = None
            event_roi = roi
            if (
                hit_time
                and (not stop_time or hit_time < stop_time)
                and hit_time <= expire_ts
            ):
                event_time = hit_time
                event_roi = row["target_pct"]
                hit_pct = 100.0
            elif (
                stop_time
                and (not hit_time or stop_time <= hit_time)
                and stop_time <= expire_ts
            ):
                event_time = stop_time
                event_roi = -row["stop_pct"]
                hit_pct = 0.0
            elif prices.index[-1] < expire_ts:
                status = "queued"
            roi = event_roi
            if event_time is not None:
                event_idx = after.index.get_loc(event_time)
                for n in roi_curve:
                    if event_idx <= n - 1:
                        roi_curve[n] = event_roi
                roi_expiry = event_roi
            dd = float(max(0.0, -mae))
            t_hit = (
                (hit_time - entry_ts).total_seconds() / 60
                if hit_time and hit_time <= expire_ts
                else None
            )
            t_stop = (
                (stop_time - entry_ts).total_seconds() / 60
                if stop_time and stop_time <= expire_ts
                else None
            )
            delta = row.get("option_delta")
            option_roi_proxy = roi / delta if delta else None
            db.execute(
                """UPDATE forward_tests
                       SET roi_forward=?, dd_forward=?, status=?, hit_forward=?, roi_1=?, roi_3=?, roi_5=?, roi_expiry=?, mae=?, mfe=?, time_to_hit=?, time_to_stop=?, option_roi_proxy=?, last_run_at=?, next_run_at=?, updated_at=?
                       WHERE id=?""",
                (
                    roi,
                    dd,
                    status,
                    hit_pct,
                    roi_curve[1],
                    roi_curve[3],
                    roi_curve[5],
                    roi_expiry,
                    mae,
                    mfe,
                    t_hit,
                    t_stop,
                    option_roi_proxy,
                    now_et().isoformat(),
                    now_et().isoformat(),
                    now_iso,
                    row["id"],
                ),
            )
        except Exception:
            logger.exception("Forward test %s failed", row["id"])
            db.execute(
                "UPDATE forward_tests SET status='error', last_run_at=?, updated_at=? WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
    db.connection.commit()


@router.get("/forward", response_class=HTMLResponse)
def forward_page(request: Request, db=Depends(get_db)):
    try:
        db.execute("SELECT * FROM favorites ORDER BY id DESC")
        favs = [row_to_dict(r, db) for r in db.fetchall()]
        for f in favs:
            db.execute(
                "SELECT status FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1",
                (f["id"],),
            )
            row = db.fetchone()
            if row is None or row["status"] in ("ok", "error"):
                _create_forward_test(db, f)
        _update_forward_tests(db)
        db.execute(
            """SELECT ft.id AS ft_id, ft.fav_id, ft.ticker, ft.direction, ft.interval,
                      ft.roi_forward, ft.option_roi_proxy, ft.hit_forward, ft.dd_forward, ft.status, ft.created_at, ft.rule
                   FROM forward_tests ft
                   ORDER BY ft.id DESC"""
        )
        tests = [row_to_dict(r, db) for r in db.fetchall()]
        if not tests:
            tests = [
                {
                    "ft_id": None,
                    "fav_id": f["id"],
                    "ticker": f["ticker"],
                    "direction": f["direction"],
                    "interval": f["interval"],
                    "roi_forward": None,
                    "option_roi_proxy": None,
                    "hit_forward": None,
                    "dd_forward": None,
                    "status": "queued",
                    "created_at": None,
                    "rule": f.get("rule"),
                }
                for f in favs
            ]
        ctx = {"tests": tests, "active_tab": "forward"}
    except Exception:
        logger.exception("Failed to load forward page")
        ctx = {
            "tests": [],
            "active_tab": "forward",
            "error": "Unable to load forward tests",
        }
    return templates.TemplateResponse(request, "forward.html", ctx)


@router.get("/api/forward/favorites")
def api_forward_favorites(db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [_normalize_favorite(row_to_dict(r, db)) for r in db.fetchall()]
    return {"favorites": favs}


@router.post("/favorites/delete/{fav_id}")
def favorites_delete(fav_id: int, db=Depends(get_db)):
    db.execute("DELETE FROM favorites WHERE id=?", (fav_id,))
    db.connection.commit()
    return RedirectResponse(url="/favorites", status_code=302)


@router.post("/favorites/delete-duplicates")
def favorites_delete_duplicates(db=Depends(get_db)):
    db.execute(
        """
        DELETE FROM favorites
        WHERE id NOT IN (
            SELECT MIN(id) FROM favorites GROUP BY ticker, direction, interval, rule
        )
        """
    )
    db.connection.commit()
    return RedirectResponse(url="/favorites", status_code=302)


@router.post("/favorites/add")
async def favorites_add(request: Request, db=Depends(get_db)):
    payload = await request.json()
    t = (payload.get("ticker") or "").strip().upper()
    rule = payload.get("rule") or ""
    direction = (payload.get("direction") or "UP").strip().upper()
    interval = (payload.get("interval") or "15m").strip()
    ref_dd = payload.get("ref_avg_dd")
    roi = payload.get("roi_snapshot")
    hit = payload.get("hit_pct_snapshot")
    dd = payload.get("dd_pct_snapshot")
    support_raw = payload.get("support_snapshot")
    rule_snap = payload.get("rule_snapshot") or rule
    settings_raw = payload.get("settings_json_snapshot")
    try:
        ref_dd = float(ref_dd)
        if ref_dd > 1:
            ref_dd /= 100.0
    except (TypeError, ValueError):
        ref_dd = None

    if not t or not rule:
        return JSONResponse(
            {"ok": False, "error": "missing ticker or rule"}, status_code=400
        )

    settings_dict: dict[str, Any] = {}
    settings_json: str | None = None
    if isinstance(settings_raw, dict):
        settings_dict = settings_raw
        settings_json = json.dumps(settings_raw)
    elif isinstance(settings_raw, str) and settings_raw:
        try:
            settings_dict = json.loads(settings_raw)
            settings_json = json.dumps(settings_dict)
        except Exception:
            settings_json = settings_raw
    elif settings_raw not in (None, ""):
        settings_json = json.dumps(settings_raw)

    lookback = None
    min_support = None
    if settings_dict:
        params = coerce_scan_params(settings_dict)
        lookback = params.get("lookback_years")
        min_support = params.get("min_support")
    else:
        lookback = payload.get("lookback_years")
        min_support = payload.get("min_support")

    lookback = _coerce_float(lookback)
    min_support = _coerce_int(min_support)
    support = _coerce_int(support_raw)

    support_payload: dict[str, Any] = {}
    if support is not None:
        support_payload["count"] = support
    if min_support is not None:
        support_payload["min_support"] = min_support
    if lookback is not None:
        support_payload["lookback_years"] = lookback

    support_snapshot = json.dumps(support_payload) if support_payload else None

    db.execute(
        """INSERT INTO favorites(
                ticker, direction, interval, rule, ref_avg_dd,
                lookback_years, min_support, support_snapshot,
                roi_snapshot, hit_pct_snapshot, dd_pct_snapshot,
                rule_snapshot, settings_json_snapshot, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            t,
            direction,
            interval,
            rule,
            ref_dd,
            lookback,
            min_support,
            support_snapshot,
            roi,
            hit,
            dd,
            rule_snap,
            settings_json,
            now_et().isoformat(),
        ),
    )
    db.connection.commit()
    log_telemetry(
        {
            "type": "favorite_saved",
            "symbol": t,
            "lookback_years": lookback,
        }
    )
    return {"ok": True}


@router.post("/favorites/test_alert")
def favorites_test_alert(payload: dict = Body(...), db=Depends(get_db)):
    symbol = (payload.get("symbol") or payload.get("ticker") or "AAPL").upper()
    channel = (payload.get("channel") or "mms").lower()
    compact = bool(payload.get("compact"))
    ok, subject, body = _extract_test_alert_message(
        favorites_alerts.enrich_and_send_test(
            symbol, "UP", channel=channel, compact=compact
        )
    )

    base_telem = {
        "type": "favorites_test_alert",
        "symbol": symbol,
        "channel": channel,
        "compact": compact,
    }

    response = {
        "ok": ok,
        "symbol": symbol,
        "channel": channel,
        "compact": compact,
        "subject": subject,
        "body": body,
    }

    if channel == "email":
        st = get_settings(db)
        cfg, missing, _ = _smtp_config_status(st)
        if not cfg["recipients"]:
            missing = list(dict.fromkeys(missing + ["recipients"]))
        if missing:
            error = f"SMTP not configured: {_format_missing_fields(sorted(set(missing)))}"
            log_telemetry(
                {
                    "type": "favorites_test_alert_send",
                    "channel": "email",
                    "provider": "smtp",
                    "ok": False,
                    "error": error,
                }
            )
            base_telem["ok"] = False
            log_telemetry(base_telem)
            response.update({"ok": False, "error": error})
            return JSONResponse(response, status_code=400)
        if not ok:
            error = "Unable to generate alert body"
            log_telemetry(
                {
                    "type": "favorites_test_alert_send",
                    "channel": "email",
                    "provider": "smtp",
                    "ok": False,
                    "error": error,
                }
            )
            response.update({"ok": False, "error": error})
            base_telem["ok"] = False
            log_telemetry(base_telem)
            return JSONResponse(response, status_code=500)

        send_result = send_email_smtp(
            cfg["host"],
            cfg["port"],
            cfg["user"],
            cfg["password"],
            cfg["mail_from"] or cfg["user"],
            cfg["recipients"],
            subject or f"Favorites Alert Test: {symbol}",
            body,
        )
        if not send_result.get("ok"):
            error = send_result.get("error", "SMTP send failed")
            log_telemetry(
                {
                    "type": "favorites_test_alert_send",
                    "channel": "email",
                    "provider": "smtp",
                    "ok": False,
                    "error": error,
                }
            )
            response.update({"ok": False, "error": error})
            base_telem["ok"] = False
            log_telemetry(base_telem)
            return JSONResponse(response, status_code=502)

        message_id = send_result.get("message_id", "")
        response.update({"ok": True, "message_id": message_id})
        log_telemetry(
            {
                "type": "favorites_test_alert_send",
                "channel": "email",
                "provider": "smtp",
                "ok": True,
                "message_id": message_id,
            }
        )
        base_telem["ok"] = True
        log_telemetry(base_telem)
        return response

    base_telem["ok"] = ok
    log_telemetry(base_telem)
    return response


@router.post("/favorites/test_alert/preview")
def favorites_test_alert_preview(payload: dict = Body(...)):
    symbol = (payload.get("symbol") or payload.get("ticker") or "AAPL").upper()
    channel = (payload.get("channel") or "mms").lower()
    compact = bool(payload.get("compact"))
    ok, subject, body = _extract_test_alert_message(
        favorites_alerts.enrich_and_send_test(
            symbol, "UP", channel=channel, compact=compact
        )
    )
    return {
        "ok": ok,
        "symbol": symbol,
        "channel": channel,
        "compact": compact,
        "subject": subject,
        "body": body,
    }


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db=Depends(get_db)):
    st = get_settings(db)
    _, smtp_missing, smtp_has_any = _smtp_config_status(st)
    smtp_configured = not smtp_missing
    smtp_warning = smtp_has_any and bool(smtp_missing)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "st": st,
            "active_tab": "settings",
            "smtp_configured": smtp_configured,
            "smtp_missing": smtp_missing,
            "smtp_missing_label": _format_missing_fields(smtp_missing),
            "smtp_warning": smtp_warning,
            "twilio_configured": False,
        },
    )


@router.get("/info", response_class=HTMLResponse)
def info_page(request: Request):
    return templates.TemplateResponse(request, "info.html", {"active_tab": "info"})


@router.post("/settings/save")
def settings_save(
    request: Request,
    smtp_host: str = Form(""),
    smtp_port: int = Form(587),
    smtp_user: str = Form(""),
    smtp_pass: str = Form(""),
    mail_from: str = Form(""),
    recipients: str = Form(""),
    scanner_recipients: str = Form(""),
    scheduler_enabled: int = Form(1),
    throttle_minutes: int = Form(60),
    db=Depends(get_db),
):
    _ensure_scanner_column(db)
    from email.utils import parseaddr

    from services.notifications import is_carrier_address

    def _clean(raw: str, *, allow_sms: bool) -> str:
        parts = [r.strip() for r in raw.split(",") if r.strip()]
        cleaned: list[str] = []
        for r in parts:
            addr = parseaddr(r)[1]
            if "@" not in addr:
                continue
            if not allow_sms and is_carrier_address(addr):
                continue
            cleaned.append(addr)
        return ",".join(cleaned)

    clean_fav = _clean(recipients, allow_sms=True)
    clean_scan = _clean(scanner_recipients, allow_sms=False)

    try:
        port_value = int(smtp_port)
    except (TypeError, ValueError):
        port_value = 0

    password_value = smtp_pass.replace(" ", "").strip()

    db.execute(
        """
        UPDATE settings
           SET smtp_host=?, smtp_port=?, smtp_user=?, smtp_pass=?, mail_from=?, recipients=?, scanner_recipients=?, scheduler_enabled=?, throttle_minutes=?
         WHERE id=1
        """,
        (
            smtp_host.strip(),
            port_value,
            smtp_user.strip(),
            password_value,
            mail_from.strip(),
            clean_fav,
            clean_scan,
            int(scheduler_enabled),
            int(throttle_minutes),
        ),
    )
    db.connection.commit()
    return RedirectResponse(url="/settings", status_code=302)


@router.post("/scanner/run")
async def scanner_run(request: Request):
    form = await request.form()
    params = coerce_scan_params(form)

    scan_type = params.get("scan_type", "scan150")
    single_ticker = (form.get("ticker") or "").strip().upper()
    if scan_type.lower() in ("single", "single_ticker") and single_ticker:
        tickers = [single_ticker]
    elif scan_type.lower() in ("sp100", "sp_100", "sp100_scan"):
        tickers = SP100
    elif scan_type.lower() in ("top250", "scan250", "options250"):
        tickers = TOP250
    else:
        tickers = TOP150

    sort_key = (form.get("sort") or "").strip().lower()
    if sort_key not in ("ticker", "roi", "hit"):
        sort_key = ""

    task_id = uuid4().hex
    _task_create(task_id, len(tickers))

    def _task():
        start_ts = time.time()
        _task_update(task_id, state="running")
        logger.info("task_start id=%s total=%d", task_id, len(tickers))

        def prog(done: int, total: int, msg: str) -> None:
            pct = 0.0 if total == 0 else (done / total) * 100.0
            _task_update(task_id, done=done, total=total, percent=pct, state="running")
            logger.info("task_progress id=%s %d/%d", task_id, done, total)

        # Surface rate limit waits to the task table so the UI can show a
        # friendly "waiting" message during backoff.
        from services import http_client

        def wait_cb(wait: float) -> None:
            if wait > 0:
                _task_update(task_id, message=f"waiting {wait:.1f}s due to rate limit")
            else:
                _task_update(task_id, message="")

        http_client.set_wait_callback(wait_cb)

        try:
            rows, skipped, metrics = _perform_scan(
                tickers, params, sort_key, progress_cb=prog
            )
            duration = time.time() - start_ts
            ctx = {
                "rows": rows,
                "ran_at": now_et().strftime("%I:%M:%S %p").lstrip("0"),
                "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}",
                "skipped_missing_data": skipped,
                "metrics": metrics,
                "summary": {
                    "successes": len(rows),
                    "empties": skipped,
                    "errors": 0,
                    "duration": duration,
                },
                "errors": [],
            }
            _task_update(
                task_id, state="succeeded", percent=100.0, done=len(tickers), ctx=ctx
            )
            logger.info(
                "task_done id=%s duration=%.2fs successes=%d empties=%d errors=%d no_gap=%d avg_ms=%.1f p95_ms=%.1f",
                task_id,
                duration,
                len(rows),
                skipped,
                0,
                metrics.get("symbols_no_gap", 0),
                metrics.get("avg_per_symbol_ms", 0.0),
                metrics.get("p95_per_symbol_ms", 0.0),
            )
        except Exception as e:
            logger.error("task_error id=%s error=%s", task_id, e)
            _task_update(task_id, state="failed", message=str(e))
        finally:
            http_client.set_wait_callback(None)

    Thread(target=_task, daemon=True).start()
    return JSONResponse({"task_id": task_id})


@router.get("/scanner/progress/{task_id}")
async def scanner_progress(task_id: str):
    task = _task_get(task_id)
    if not task:
        return JSONResponse(
            {"done": 0, "total": 0, "percent": 0.0, "state": "failed"}, status_code=404
        )
    data = {
        "done": task.get("done", 0),
        "total": task.get("total", 0),
        "percent": task.get("percent", 0.0),
        "state": task.get("state", "running"),
        "message": task.get("message", ""),
    }
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@router.get("/scanner/status/{task_id}")
async def scanner_status(task_id: str):
    _task_gc()
    task = _task_get(task_id)
    if not task:
        return JSONResponse({}, status_code=404)
    data = {
        "id": task_id,
        "total": task.get("total"),
        "completed": task.get("done"),
        "percent": task.get("percent"),
        "state": task.get("state"),
        "message": task.get("message"),
        "started_at": task.get("started_at"),
        "updated_at": task.get("updated_at"),
        "ctx": task.get("ctx"),
    }
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@router.get("/scanner/results/{task_id}", response_class=HTMLResponse)
async def scanner_results(request: Request, task_id: str):
    task = _task_get(task_id)
    if not task or task.get("state") != "succeeded":
        return HTMLResponse("Not ready", status_code=404)
    ctx = (task.get("ctx") or {}).copy()
    logger.info("task %s rendered", task_id)
    response = templates.TemplateResponse(
        request, "results.html", ctx, headers={"Cache-Control": "no-store"}
    )
    _task_delete(task_id)
    return response


@router.post("/scanner/parity")
def scanner_parity(request: Request):
    PARAMS = dict(
        interval="15m",
        direction="BOTH",
        target_pct=1.5,
        stop_pct=0.7,
        window_value=8.0,
        window_unit="Hours",
        lookback_years=2.0,
        max_tt_bars=20,
        min_support=20,
        delta_assumed=0.25,
        theta_per_day_pct=0.20,
        atrz_gate=-0.5,
        slope_gate_pct=-0.01,
        use_regime=1,
        regime_trend_only=0,
        vix_z_max=3.0,
        slippage_bps=7.0,
        vega_scale=0.03,
        scan_min_hit=55.0,
        scan_max_dd=1.0,
    )
    sort_key = request.query_params.get("sort")
    rows = []
    for t in TOP150:
        r = compute_scan_for_ticker(t, PARAMS) or {}
        if not r:
            continue
        if (
            r.get("hit_pct", 0) >= PARAMS["scan_min_hit"]
            and r.get("avg_dd_pct", 999) <= PARAMS["scan_max_dd"]
        ):
            rows.append(r)

    rows.sort(
        key=lambda x: (x["avg_roi_pct"], x["hit_pct"], x["support"], x["stability"]),
        reverse=True,
    )

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "rows": _sort_rows(rows, sort_key),
            "ran_at": now_et().strftime("%Y-%m-%d %H:%M"),
            "note": f"TOP150 parity run • kept {len(rows)}",
        },
    )



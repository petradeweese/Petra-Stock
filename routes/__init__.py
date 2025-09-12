# ruff: noqa: E501
import atexit
import json
import logging
import os
import smtplib
import sqlite3
import ssl
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from email.message import EmailMessage
from threading import Thread
from typing import Any, Callable, Dict, Mapping, Optional, Union
from uuid import uuid4

import certifi
import pandas as pd
from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import settings
from db import DB_PATH, get_db, get_schema_status, get_settings
from indices import SP100, TOP150, TOP250
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from scanner import compute_scan_for_ticker, preload_prices
from services.market_data import get_prices, window_from_lookback
from utils import TZ, now_et

from .archive import _format_rule_summary as _format_rule_summary
from .archive import router as archive_router

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

router.include_router(archive_router)

scan_duration = Histogram("scan_duration_seconds", "Duration of /scanner/run requests")
scan_tickers = Counter("scan_tickers_total", "Tickers processed by /scanner/run")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", **get_schema_status()}


@router.get("/healthz")
def healthz() -> dict:
    """Simple health check endpoint returning only status."""
    return {"status": "ok"}


def metrics() -> Response:
    """Return Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if settings.metrics_enabled:
    router.get("/metrics")(metrics)


_scan_executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None


_TASK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scan_tasks (
    id TEXT PRIMARY KEY,
    total INTEGER,
    done INTEGER,
    percent REAL,
    state TEXT,
    message TEXT,
    ctx TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(_TASK_TABLE_SQL)
    conn.commit()
    return conn


def _task_create(task_id: str, total: int) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO scan_tasks (id, total, done, percent, state) VALUES (?, ?, 0, 0.0, 'running')",
            (task_id, total),
        )
        conn.commit()
    finally:
        conn.close()


def _task_update(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    conn = _get_conn()
    try:
        cols = []
        vals: list[Any] = []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            if k == "ctx" and v is not None:
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        vals.append(task_id)
        conn.execute(f"UPDATE scan_tasks SET {', '.join(cols)} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def _task_get(task_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT total, done, percent, state, message, ctx FROM scan_tasks WHERE id=?",
            (task_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    total, done, percent, state, message, ctx_json = row
    return {
        "total": total,
        "done": done,
        "percent": percent,
        "state": state,
        "message": message,
        "ctx": json.loads(ctx_json) if ctx_json else None,
    }


def _task_delete(task_id: str) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM scan_tasks WHERE id=?", (task_id,))
        conn.commit()
    finally:
        conn.close()


def _shutdown_executor() -> None:
    global _scan_executor
    if _scan_executor:
        _scan_executor.shutdown()
        _scan_executor = None


def _get_scan_executor() -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
    """Return a global executor reused across requests.

    The scanner is CPU intensive, so by default we use a small
    ``ProcessPoolExecutor`` to take advantage of multiple cores.  If the scan
    function can't be pickled (e.g. during tests when monkeypatching with a
    local function), we automatically fall back to a thread pool so the call
    still succeeds.
    """

    global _scan_executor
    if _scan_executor is None:
        max_workers = max(1, int(os.getenv("SCAN_WORKERS", "3")))
        exec_type = os.getenv("SCAN_EXECUTOR", "process").lower()
        Executor = ProcessPoolExecutor if exec_type == "process" else ThreadPoolExecutor
        if Executor is ProcessPoolExecutor:
            import pickle

            try:  # pragma: no cover - only fails in tests
                pickle.dumps(compute_scan_for_ticker)
            except Exception:
                Executor = ThreadPoolExecutor
        _scan_executor = Executor(max_workers=max_workers)
        atexit.register(_shutdown_executor)
    return _scan_executor


def _perform_scan(
    tickers: list[str],
    params: dict,
    sort_key: str,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    progress_every: int = 5,
) -> tuple[list[dict], int]:
    start = time.perf_counter()
    total = len(tickers)
    preload_prices(
        tickers, params.get("interval", "15m"), params.get("lookback_years", 2.0)
    )
    if progress_cb:
        progress_cb(0, total, "preloading")
    rows: list[dict] = []
    skipped_missing_data = 0
    ex = _get_scan_executor()
    future_to_ticker = {
        ex.submit(compute_scan_for_ticker, t, params): t for t in tickers
    }
    step = max(1, int(progress_every))
    done = 0
    for fut in as_completed(future_to_ticker):
        ticker = future_to_ticker[fut]
        try:
            r = fut.result()
            if r is None:
                skipped_missing_data += 1
            elif r:
                rows.append(r)
        except Exception as e:
            logger.error("scan failed for %s: %s", ticker, e)
        done += 1
        if progress_cb and (done % step == 0 or done == total):
            progress_cb(done, total, f"Scanning {done}/{total}")

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

    duration = time.perf_counter() - start
    scan_duration.observe(duration)
    scan_tickers.inc(len(tickers))
    if duration > 0:
        logger.info(
            "scan completed: %d tickers in %.2fs (%.2f tickers/sec) skipped_missing_data=%d",
            len(tickers),
            duration,
            len(tickers) / duration,
            skipped_missing_data,
        )
    else:
        logger.info(
            "scan completed: %d tickers in %.2fs skipped_missing_data=%d",
            len(tickers),
            duration,
            skipped_missing_data,
        )

    return rows, skipped_missing_data


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
    st: sqlite3.Row, subject: str, body: str, html_body: Optional[str] = None
) -> None:
    """Send an email using settings stored in the database.

    This helper mirrors the logic used by the desktop application: the
    configured SMTP user/password are expected to work with Gmail.  The
    function silently returns if mandatory settings are missing so the
    scanner can proceed without failing.
    """

    user = (st["smtp_user"] or "").strip()
    pwd = (st["smtp_pass"] or "").replace(" ", "").strip()
    recips = [r.strip() for r in (st["recipients"] or "").split(",") if r.strip()]
    if not user or not pwd or not recips:
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(recips)
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(user, pwd)
            server.send_message(msg)
    except ssl.SSLError:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(user, pwd)
            server.send_message(msg)


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "active_tab": "scanner"}
    )


@router.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "active_tab": "scanner"}
    )


@router.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request, db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [dict(r) for r in db.fetchall()]
    for f in favs:
        f["avg_roi_pct"] = f.get("roi_snapshot")
        f["hit_pct"] = f.get("hit_pct_snapshot")
        f["avg_dd_pct"] = f.get("dd_pct_snapshot")
        if f.get("rule_snapshot"):
            f["rule"] = f.get("rule_snapshot")
    return templates.TemplateResponse(
        "favorites.html",
        {"request": request, "favorites": favs, "active_tab": "favorites"},
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
    now_iso = now_et().isoformat()
    db.execute(
        """INSERT INTO forward_tests
            (fav_id, ticker, direction, interval, rule, entry_price,
             target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward, dd_forward,
             last_run_at, next_run_at, runs_count, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0.0, NULL, 0.0, NULL, NULL, 0, NULL, ?, ?)""",
        (
            fav["id"],
            fav["ticker"],
            fav.get("direction", "UP"),
            fav.get("interval", "15m"),
            fav.get("rule"),
            entry_price,
            fav.get("target_pct", 1.0),
            fav.get("stop_pct", 0.5),
            window_minutes,
            entry_ts,
            now_iso,
        ),
    )
    db.connection.commit()


def _update_forward_tests(db: sqlite3.Cursor) -> None:
    db.execute(
        """SELECT id, ticker, direction, interval, created_at, entry_price,
                  target_pct, stop_pct, window_minutes, status
               FROM forward_tests
               WHERE status IN ('queued','running')"""
    )
    rows = [dict(r) for r in db.fetchall()]
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
            mult = 1.0 if row["direction"] == "UP" else -1.0
            pct_series = (prices / row["entry_price"] - 1.0) * 100 * mult
            roi = float(pct_series.iloc[-1])
            mae = float(pct_series.min())
            status = "ok"
            hit_pct = None
            if row["direction"] == "UP":
                hit_cond = prices >= row["entry_price"] * (1 + row["target_pct"] / 100)
                stop_cond = prices <= row["entry_price"] * (1 - row["stop_pct"] / 100)
            else:
                hit_cond = prices <= row["entry_price"] * (1 - row["target_pct"] / 100)
                stop_cond = prices >= row["entry_price"] * (1 + row["stop_pct"] / 100)
            hit_time = prices[hit_cond].index[0] if hit_cond.any() else None
            stop_time = prices[stop_cond].index[0] if stop_cond.any() else None
            expire_ts = entry_ts + pd.Timedelta(minutes=row["window_minutes"])
            final_ts = after.index[-1]
            if (
                hit_time
                and (not stop_time or hit_time <= stop_time)
                and hit_time <= expire_ts
            ):
                roi = float(pct_series.loc[hit_time])
                hit_pct = 100.0
            elif (
                stop_time
                and (not hit_time or stop_time < hit_time)
                and stop_time <= expire_ts
            ):
                roi = float(pct_series.loc[stop_time])
                hit_pct = 0.0
            elif final_ts < expire_ts:
                status = "queued"
            dd = float(max(0.0, -mae))
            db.execute(
                """UPDATE forward_tests
                       SET roi_forward=?, dd_forward=?, status=?, hit_forward=?, last_run_at=?, next_run_at=?, updated_at=?
                       WHERE id=?""",
                (
                    roi,
                    dd,
                    status,
                    hit_pct,
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
        favs = [dict(r) for r in db.fetchall()]
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
                      ft.roi_forward, ft.hit_forward, ft.dd_forward, ft.status, ft.created_at, ft.rule
                   FROM forward_tests ft
                   ORDER BY ft.id DESC"""
        )
        tests = [dict(r) for r in db.fetchall()]
        ctx = {"request": request, "tests": tests, "active_tab": "forward"}
    except Exception:
        logger.exception("Failed to load forward page")
        ctx = {
            "request": request,
            "tests": [],
            "active_tab": "forward",
            "error": "Unable to load forward tests",
        }
    return templates.TemplateResponse("forward.html", ctx)


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
    rule_snap = payload.get("rule_snapshot") or rule
    settings = payload.get("settings_json_snapshot")
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

    if isinstance(settings, dict):
        settings = json.dumps(settings)
    db.execute(
        """INSERT INTO favorites(
                ticker, direction, interval, rule, ref_avg_dd,
                roi_snapshot, hit_pct_snapshot, dd_pct_snapshot,
                rule_snapshot, settings_json_snapshot, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            t,
            direction,
            interval,
            rule,
            ref_dd,
            roi,
            hit,
            dd,
            rule_snap,
            settings,
            now_et().isoformat(),
        ),
    )
    db.connection.commit()
    return {"ok": True}


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db=Depends(get_db)):
    st = get_settings(db)
    return templates.TemplateResponse(
        "settings.html", {"request": request, "st": st, "active_tab": "settings"}
    )


@router.get("/info", response_class=HTMLResponse)
def info_page(request: Request):
    return templates.TemplateResponse(
        "info.html", {"request": request, "active_tab": "info"}
    )


@router.post("/settings/save")
def settings_save(
    request: Request,
    smtp_user: str = Form(""),
    smtp_pass: str = Form(""),
    recipients: str = Form(""),
    scheduler_enabled: int = Form(1),
    throttle_minutes: int = Form(60),
    db=Depends(get_db),
):
    db.execute(
        """
        UPDATE settings
           SET smtp_user=?, smtp_pass=?, recipients=?, scheduler_enabled=?, throttle_minutes=?
         WHERE id=1
        """,
        (
            smtp_user.strip(),
            smtp_pass.strip(),
            recipients.strip(),
            int(scheduler_enabled),
            int(throttle_minutes),
        ),
    )
    db.connection.commit()
    return RedirectResponse(url="/settings", status_code=302)


@router.post("/scanner/run")
async def scanner_run(request: Request):
    form = await request.form()
    params = _coerce_scan_params(form)

    scan_type = params.get("scan_type", "scan150")
    single_ticker = str(form.get("ticker") or "").strip().upper()
    if scan_type.lower() in ("single", "single_ticker") and single_ticker:
        tickers = [single_ticker]
    elif scan_type.lower() in ("sp100", "sp_100", "sp100_scan"):
        tickers = SP100
    elif scan_type.lower() in ("top250", "scan250", "options250"):
        tickers = TOP250
    else:
        tickers = TOP150

    sort_key = str(form.get("sort") or "").strip().lower()
    if sort_key not in ("ticker", "roi", "hit"):
        sort_key = ""

    task_id = uuid4().hex
    logger.info("task %s started", task_id)
    _task_create(task_id, len(tickers))

    def _task():
        def prog(done: int, total: int, msg: str) -> None:
            pct = 0.0 if total == 0 else (done / total) * 100.0
            _task_update(task_id, done=done, total=total, percent=pct, state="running")
            logger.info("task %s progress %d/%d", task_id, done, total)

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
            rows, skipped = _perform_scan(tickers, params, sort_key, progress_cb=prog)
            ctx = {
                "rows": rows,
                "ran_at": now_et().strftime("%I:%M:%S %p").lstrip("0"),
                "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}",
                "skipped_missing_data": skipped,
            }
            _task_update(
                task_id, state="done", percent=100.0, done=len(tickers), ctx=ctx
            )
            logger.info("task %s saved results", task_id)
        except Exception as e:
            logger.error("scan task %s failed: %s", task_id, e)
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


@router.get("/scanner/results/{task_id}", response_class=HTMLResponse)
async def scanner_results(request: Request, task_id: str):
    task = _task_get(task_id)
    if not task or task.get("state") != "done":
        return HTMLResponse("Not ready", status_code=404)
    ctx = (task.get("ctx") or {}).copy()
    ctx["request"] = request
    logger.info("task %s rendered", task_id)
    response = templates.TemplateResponse(
        "results.html", ctx, headers={"Cache-Control": "no-store"}
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
        "results.html",
        {
            "request": request,
            "rows": _sort_rows(rows, sort_key),
            "ran_at": now_et().strftime("%Y-%m-%d %H:%M"),
            "note": f"TOP150 parity run • kept {len(rows)}",
        },
    )


def _coerce_scan_params(form: Mapping[str, Any]) -> dict:
    """Coerce scanner form fields into typed params (no silent hard-coding)."""

    def F(k, cast=float, default=None):
        v = form.get(k, None)
        if v in (None, ""):
            return default
        try:
            return cast(v)
        except Exception:
            return default

    return {
        "scan_type": (form.get("scan_type") or "scan150"),
        "ticker": (form.get("ticker") or "").strip().upper(),
        "interval": (form.get("interval") or "15m").strip(),
        "direction": (form.get("direction") or "BOTH").strip().upper(),
        "target_pct": F("target_pct", float, 1.0),
        "stop_pct": F("stop_pct", float, 0.5),
        "window_value": F("window_value", float, 4.0),
        "window_unit": (form.get("window_unit") or "Hours").strip(),
        "lookback_years": F("lookback_years", float, 2.0),
        "max_tt_bars": F("max_tt_bars", int, 12),
        "min_support": F("min_support", int, 20),
        "delta_assumed": F("delta_assumed", float, 0.40),
        "theta_per_day_pct": F("theta_per_day_pct", float, 0.20),
        "atrz_gate": F("atrz_gate", float, 0.10),
        "slope_gate_pct": F("slope_gate_pct", float, 0.02),
        "use_regime": F("use_regime", int, 0),
        "regime_trend_only": F("regime_trend_only", int, 0),
        "vix_z_max": F("vix_z_max", float, 3.0),
        "slippage_bps": F("slippage_bps", float, 7.0),
        "vega_scale": F("vega_scale", float, 0.03),
        "scan_min_hit": F("scan_min_hit", float, 50.0),
        "scan_max_dd": F("scan_max_dd", float, 50.0),
        "email_checkbox": (form.get("email_checkbox") or ""),
    }

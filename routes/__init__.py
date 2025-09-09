import atexit
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, Callable
import logging
import sqlite3
import smtplib
import ssl
from email.message import EmailMessage
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from uuid import uuid4
from threading import Thread, Lock
import asyncio

import certifi
import time
from fastapi import APIRouter, Request, Form, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from indices import SP100, TOP150, TOP250
from db import DB_PATH, get_db, get_settings
from scanner import compute_scan_for_ticker, preload_prices
from services.data_fetcher import fetch_prices
from utils import now_et, now_utc, TZ
import pandas as pd

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

scan_duration = Histogram("scan_duration_seconds", "Duration of /scanner/run requests")
scan_tickers = Counter("scan_tickers_total", "Tickers processed by /scanner/run")


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_scan_executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
_scan_tasks: Dict[str, Dict[str, Any]] = {}
_scan_lock = Lock()


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
) -> list[dict]:
    start = time.perf_counter()
    total = len(tickers)
    preload_prices(tickers, params.get("interval", "15m"), params.get("lookback_years", 2.0))
    if progress_cb:
        progress_cb(0, total, "preloading")
    rows: list[dict] = []
    ex = _get_scan_executor()
    future_to_ticker = {ex.submit(compute_scan_for_ticker, t, params): t for t in tickers}
    done = 0
    for fut in as_completed(future_to_ticker):
        ticker = future_to_ticker[fut]
        try:
            r = fut.result()
            if r:
                rows.append(r)
        except Exception as e:
            logger.error("scan failed for %s: %s", ticker, e)
        done += 1
        if progress_cb:
            progress_cb(done, total, f"Scanning {done}/{total}")

    try:
        scan_min_hit = float(params.get("scan_min_hit", 0.0))
        scan_max_dd = float(params.get("scan_max_dd", 100.0))
    except Exception:
        scan_min_hit, scan_max_dd = 0.0, 100.0

    rows = [
        r for r in rows
        if (r.get("hit_pct", 0.0) >= scan_min_hit) and (r.get("avg_dd_pct", 100.0) <= scan_max_dd)
    ]

    if sort_key == "ticker":
        rows.sort(key=lambda r: (r.get("ticker") or ""))
    elif sort_key == "roi":
        rows.sort(key=lambda r: (r.get("avg_roi_pct", 0.0), r.get("hit_pct", 0.0), r.get("support", 0)), reverse=True)
    elif sort_key == "hit":
        rows.sort(key=lambda r: (r.get("hit_pct", 0.0), r.get("avg_roi_pct", 0.0), r.get("support", 0)), reverse=True)
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
            "scan completed: %d tickers in %.2fs (%.2f tickers/sec)",
            len(tickers),
            duration,
            len(tickers) / duration,
        )
    else:
        logger.info("scan completed: %d tickers in %.2fs", len(tickers), duration)

    return rows


def _format_rule_summary(params: Dict[str, Any]) -> str:
    parts: list[str] = []
    def fmt_pct(v: Any) -> str:
        try:
            return f"{float(v):g}%"
        except (ValueError, TypeError):
            return f"{v}%"
    def fmt_val(v: Any) -> str:
        try:
            return f"{float(v):g}"
        except (ValueError, TypeError):
            return str(v)
    for key, label, fmt in [
        ("target_pct", "Target", fmt_pct),
        ("stop_pct", "Stop", fmt_pct),
        ("max_tt_bars", "MaxBars", fmt_val),
        ("scan_min_hit", "MinHit%", fmt_pct),
        ("scan_max_dd", "MaxDD%", fmt_pct),
        ("vega_scale", "Vega", fmt_val),
        ("vix_z_max", "VIXz", fmt_val),
    ]:
        val = params.get(key)
        if val not in (None, ""):
            parts.append(f"{label} {fmt(val)}")
    return " - ".join(parts)

def _sort_rows(rows, sort_key):
    if not rows or not sort_key:
        return rows
    keymap = {
        'ticker': lambda r: (r.get('ticker') or ''),
        'roi':    lambda r: (r.get('avg_roi_pct') or 0.0),
        'hit':    lambda r: (r.get('hit_pct') or 0.0),
    }
    keyfn = keymap.get(sort_key)
    if not keyfn:
        return rows
    reverse = sort_key != 'ticker'
    return sorted(rows, key=keyfn, reverse=reverse)


def _send_email(st: sqlite3.Row, subject: str, body: str, html_body: Optional[str] = None) -> None:
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
    return templates.TemplateResponse("index.html", {"request": request, "active_tab": "scanner"})


@router.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_tab": "scanner"})


@router.get("/results/{run_id}", response_class=HTMLResponse)
def results_from_archive(request: Request, run_id: int, db=Depends(get_db)):
    db.execute("SELECT * FROM runs WHERE id=?", (run_id,))
    run = db.fetchone()
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    db.execute(
        """SELECT ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule
           FROM run_results WHERE run_id=?""",
        (run_id,),
    )
    rows = [dict(r) for r in db.fetchall()]
    rows.sort(key=lambda r: (r["avg_roi_pct"], r["hit_pct"], r["support"], r["stability"]), reverse=True)

    params = {}
    rule_summary = ""
    ran_at = ""
    try:
        params = json.loads(run["params_json"] or "{}")
        rule_summary = _format_rule_summary(params)

        dt = datetime.fromisoformat(run["started_at"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        else:
            dt = dt.astimezone(TZ)
        ran_at = dt.strftime("%m/%d/%y,%I:%M%p").replace("/0", "/").replace(",0", ",")
    except Exception:
        pass

    return templates.TemplateResponse(
        "results_page.html",
        {
            "request": request,
            "rows": rows,
            "scan_type": run["scan_type"],
            "universe_count": len((run["universe"] or "").split(",")) if run["universe"] else 0,
            "run_id": run_id,
            "active_tab": "archive",
            "ran_at": ran_at,
            "rule_summary": rule_summary,
        },
    )


@router.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request, db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [dict(r) for r in db.fetchall()]
    for f in favs:
        # Try to pull the latest archived metrics for this favorite
        db.execute(
            """
            SELECT rr.avg_roi_pct, rr.hit_pct, rr.avg_dd_pct
            FROM run_results rr
            JOIN runs r ON rr.run_id = r.id
            WHERE rr.ticker=? AND rr.rule=?
            ORDER BY r.id DESC
            LIMIT 1
            """,
            (f["ticker"], f["rule"]),
        )
        row = db.fetchone()
        if row and row["avg_roi_pct"] is not None:
            f["avg_roi_pct"] = row["avg_roi_pct"]
            f["hit_pct"] = row["hit_pct"]
            f["avg_dd_pct"] = row["avg_dd_pct"]
            continue

        # Fall back to recomputing on the fly if no archived data exists
        params = {
            "interval": f.get("interval", "15m"),
            "direction": f.get("direction", "UP"),
            "target_pct": f.get("target_pct", 1.0),
            "stop_pct": f.get("stop_pct", 0.5),
            "window_value": f.get("window_value", 4.0),
            "window_unit": f.get("window_unit", "Hours"),
            "lookback_years": f.get("lookback_years", 2.0),
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
            # Ensure no additional filtering is applied
            "scan_min_hit": 0.0,
            "scan_max_dd": 100.0,
        }
        row = compute_scan_for_ticker(f["ticker"], params)
        if row:
            f["avg_roi_pct"] = row.get("avg_roi_pct")
            f["hit_pct"] = row.get("hit_pct")
            f["avg_dd_pct"] = row.get("avg_dd_pct")
        else:
            f["avg_roi_pct"] = None
            f["hit_pct"] = None
            f["avg_dd_pct"] = None
    return templates.TemplateResponse("favorites.html", {"request": request, "favorites": favs, "active_tab": "favorites"})


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
    data = fetch_prices([fav["ticker"]], fav.get("interval", "15m"), fav.get("lookback_years", 1.0)).get(fav["ticker"])
    if data is None or getattr(data, "empty", True):
        return
    last_bar = data.iloc[-1]
    ts = last_bar.name
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    entry_ts = ts.astimezone(timezone.utc).isoformat()
    entry_price = float(last_bar["Close"])
    window_minutes = _window_to_minutes(fav.get("window_value", 4.0), fav.get("window_unit", "Hours"))
    db.execute(
        """INSERT INTO forward_tests
            (fav_id, ticker, direction, interval, rule, entry_ts, entry_price,
             target_pct, stop_pct, window_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fav["id"],
            fav["ticker"],
            fav.get("direction", "UP"),
            fav.get("interval", "15m"),
            fav.get("rule"),
            entry_ts,
            entry_price,
            fav.get("target_pct", 1.0),
            fav.get("stop_pct", 0.5),
            window_minutes,
        ),
    )
    db.connection.commit()


def _update_forward_tests(db: sqlite3.Cursor) -> None:
    db.execute(
        """SELECT id, ticker, direction, interval, entry_ts, entry_price,
                  target_pct, stop_pct, window_minutes
               FROM forward_tests
               WHERE status='OPEN'"""
    )
    rows = [dict(r) for r in db.fetchall()]
    for row in rows:
        data = fetch_prices([row["ticker"]], row["interval"], 1.0).get(row["ticker"])
        if data is None or getattr(data, "empty", True):
            continue
        entry_ts = pd.Timestamp(row["entry_ts"])
        after = data[data.index > entry_ts]
        if after.empty:
            continue
        prices = after["Close"]
        mult = 1.0 if row["direction"] == "UP" else -1.0
        pct_series = (prices / row["entry_price"] - 1.0) * 100 * mult
        roi = float(pct_series.iloc[-1])
        mfe = float(pct_series.max())
        mae = float(pct_series.min())
        status = "OPEN"
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
        if hit_time and (not stop_time or hit_time <= stop_time) and hit_time <= expire_ts:
            roi = float(pct_series.loc[hit_time])
            status = "HIT"
            hit_pct = 100.0
        elif stop_time and (not hit_time or stop_time < hit_time) and stop_time <= expire_ts:
            roi = float(pct_series.loc[stop_time])
            status = "STOP"
            hit_pct = 0.0
        elif final_ts >= expire_ts:
            subset = after[after.index <= expire_ts]
            if not subset.empty:
                pct_subset = (subset / row["entry_price"] - 1.0) * 100 * mult
                roi = float(pct_subset.iloc[-1])
                mfe = float(pct_subset.max())
                mae = float(pct_subset.min())
            status = "EXPIRED"
            hit_pct = 0.0
        dd = float(max(0.0, -mae))
        db.execute(
            """UPDATE forward_tests
                   SET roi_pct=?, mfe_pct=?, mae_pct=?, dd_pct=?, status=?, hit_pct=?, updated_at=?
                   WHERE id=?""",
            (roi, mfe, mae, dd, status, hit_pct, now_utc().isoformat(), row["id"]),
        )
    db.connection.commit()


@router.get("/forward", response_class=HTMLResponse)
def forward_page(request: Request, db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [dict(r) for r in db.fetchall()]
    for f in favs:
        db.execute("SELECT status FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1", (f["id"],))
        row = db.fetchone()
        if row is None or row["status"] != "OPEN":
            _create_forward_test(db, f)
    _update_forward_tests(db)
    db.execute(
        """SELECT ft.id AS ft_id, ft.fav_id, ft.ticker, ft.direction, ft.interval,
                  ft.roi_pct, ft.hit_pct, ft.dd_pct, ft.status, ft.entry_ts, ft.rule
               FROM forward_tests ft
               ORDER BY ft.id DESC"""
    )
    tests = [dict(r) for r in db.fetchall()]
    for t in tests:
        ts = t.get("entry_ts")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # convert to Eastern Time for display while keeping DB stored as UTC
                t["entry_ts"] = dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M")
            except Exception:
                # keep the original string if parsing fails
                t["entry_ts"] = ts
        else:
            t["entry_ts"] = ""
    return templates.TemplateResponse("forward.html", {"request": request, "tests": tests, "active_tab": "forward"})


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
    try:
        ref_dd = float(ref_dd)
        if ref_dd > 1:
            ref_dd /= 100.0
    except (TypeError, ValueError):
        ref_dd = None

    if not t or not rule:
        return JSONResponse({"ok": False, "error": "missing ticker or rule"}, status_code=400)

    db.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule, ref_avg_dd) VALUES (?, ?, ?, ?, ?)",
        (t, direction, interval, rule, ref_dd),
    )
    db.connection.commit()
    return {"ok": True}


@router.get("/archive", response_class=HTMLResponse)
def archive_page(request: Request, db=Depends(get_db)):
    db.execute(
        "SELECT id, started_at, scan_type, params_json, universe, finished_at, hit_count FROM runs ORDER BY id DESC LIMIT 200"
    )
    runs = [dict(r) for r in db.fetchall()]
    for r in runs:
        try:
            params = json.loads(r.get("params_json") or "{}")
        except Exception:
            params = {}
        tgt = params.get("target_pct")
        stp = params.get("stop_pct")
        parts = []
        if tgt is not None:
            try:
                parts.append(f"Target,{float(tgt):g}%")
            except (ValueError, TypeError):
                parts.append(f"Target,{tgt}%")
        if stp is not None:
            try:
                parts.append(f"Stop,{float(stp):g}%")
            except (ValueError, TypeError):
                parts.append(f"Stop,{stp}%")
        rule_summary = "-".join(parts)
        try:
            dt = datetime.fromisoformat(r["started_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            else:
                dt = dt.astimezone(TZ)
            fmt = dt.strftime("%m/%d/%y,%I:%M%p").replace("/0", "/").replace(",0", ",")
            r["started_display"] = f"{fmt}-{rule_summary}" if rule_summary else fmt
        except Exception:
            r["started_display"] = r["started_at"]
    return templates.TemplateResponse("archive.html", {"request": request, "runs": runs, "active_tab": "archive"})


@router.post("/archive/save")
async def archive_save(request: Request, db=Depends(get_db)):
    """
    Accepts JSON: { params: {...}, rows: [ {ticker, direction, avg_roi_pct, hit_pct, support, avg_dd_pct, stability, rule}, ... ] }
    Writes a run to `runs` and details to `run_results`. Returns {ok: True, run_id}.
    """
    try:
        payload = await request.json()
        params = payload.get("params", {}) or {}
        rows = payload.get("rows", []) or []
        if not rows:
            return JSONResponse({"ok": False, "error": "no rows"}, status_code=400)

        started = now_utc().isoformat()
        finished = started
        scan_type = str(params.get("scan_type") or "scan150")
        universe = ",".join({r.get("ticker","") for r in rows if r.get("ticker")})

        db.execute(
            """
            INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (started, scan_type, json.dumps(params), universe, finished, len(rows)),
        )
        run_id = db.lastrowid

        for r in rows:
            db.execute(
                """
                INSERT INTO run_results
                  (run_id, ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    r.get("ticker"),
                    (r.get("direction") or "UP"),
                    float(r.get("avg_roi_pct") or 0.0),
                    float(r.get("hit_pct") or 0.0),
                    int(r.get("support") or 0),
                    float(r.get("avg_tt") or 0.0),
                    float(r.get("avg_dd_pct") or 0.0),
                    float(r.get("stability") or 0.0),
                    r.get("rule") or "",
                ),
            )
        db.connection.commit()
        return {"ok": True, "run_id": run_id}
    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db=Depends(get_db)):
    st = get_settings(db)
    return templates.TemplateResponse("settings.html", {"request": request, "st": st, "active_tab": "settings"})


@router.get("/info", response_class=HTMLResponse)
def info_page(request: Request):
    return templates.TemplateResponse("info.html", {"request": request, "active_tab": "info"})


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
        (smtp_user.strip(), smtp_pass.strip(), recipients.strip(), int(scheduler_enabled), int(throttle_minutes)),
    )
    db.connection.commit()
    return RedirectResponse(url="/settings", status_code=302)


@router.post("/scanner/run")
async def scanner_run(request: Request):
    form = await request.form()
    params = _coerce_scan_params(form)

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
    with _scan_lock:
        _scan_tasks[task_id] = {
            "total": len(tickers),
            "done": 0,
            "phase": "starting",
            "pct": 0.0,
            "message": "starting",
        }

    def _task():
        def prog(done: int, total: int, msg: str) -> None:
            pct = 0.0 if total == 0 else (done / total) * 100.0
            with _scan_lock:
                task = _scan_tasks.get(task_id)
                if task:
                    task.update({"done": done, "total": total, "pct": pct, "message": msg, "phase": "scanning"})
            logger.info("task %s progress %d/%d", task_id, done, total)

        try:
            rows = _perform_scan(tickers, params, sort_key, progress_cb=prog)
            ctx = {
                "rows": rows,
                "ran_at": now_et().strftime("%I:%M:%S %p").lstrip("0"),
                "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}"
            }
            with _scan_lock:
                _scan_tasks[task_id].update({"phase": "complete", "pct": 100.0, "done": len(tickers), "ctx": ctx})
        except Exception as e:
            logger.error("scan task %s failed: %s", task_id, e)
            with _scan_lock:
                _scan_tasks[task_id].update({"phase": "error", "message": str(e)})

    Thread(target=_task, daemon=True).start()
    return JSONResponse({"task_id": task_id})


@router.get("/scanner/progress")
async def scanner_progress(task_id: str):
    async def event_gen():
        while True:
            with _scan_lock:
                task = _scan_tasks.get(task_id)
                data = task.copy() if task else {"phase": "unknown"}
            yield f"data: {json.dumps({k: data.get(k) for k in ('pct','done','total','phase','message')})}\n\n"
            if not task or data.get("phase") in ("complete", "error", "unknown"):
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/scanner/results/{task_id}", response_class=HTMLResponse)
async def scanner_results(request: Request, task_id: str):
    with _scan_lock:
        task = _scan_tasks.get(task_id)
    if not task or task.get("phase") != "complete":
        return HTMLResponse("Not ready", status_code=404)
    ctx = task.get("ctx", {}).copy()
    ctx["request"] = request
    return templates.TemplateResponse("results.html", ctx)

@router.post("/runs/archive")
async def archive_run(request: Request, db=Depends(get_db)):
    """
    Body: JSON with {"scan_type": "...", "params": {...}, "rows": [...], "universe": [...]}
    Saves a run + run_results. Only saves rows that passed filters (what UI showed).
    """
    payload = await request.json()
    scan_type = payload.get("scan_type", "")
    params = payload.get("params", {})
    rows = payload.get("rows", [])
    universe = payload.get("universe", [])

    started_at = now_utc().isoformat()
    db.execute(
        "INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count) VALUES (?, ?, ?, ?, ?, ?)",
        (started_at, scan_type, json.dumps(params), ",".join(universe), now_utc().isoformat(), len(rows)),
    )
    run_id = db.lastrowid

    for r in rows:
        db.execute(
            """INSERT INTO run_results(run_id, ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                r.get("ticker"),
                r.get("direction", "UP"),
                float(r.get("avg_roi_pct", 0)),
                float(r.get("hit_pct", 0)),
                int(r.get("support", 0)),
                float(r.get("avg_tt", 0)),
                float(r.get("avg_dd_pct", 0)),
                float(r.get("stability", 0)),
                r.get("rule", ""),
            ),
        )
    db.connection.commit()
    return {"ok": True, "run_id": run_id}


@router.post("/scanner/parity")
def scanner_parity(request: Request):
    PARAMS = dict(
        interval="15m", direction="BOTH",
        target_pct=1.5, stop_pct=0.7,
        window_value=8.0, window_unit="Hours",
        lookback_years=2.0, max_tt_bars=20, min_support=20,
        delta_assumed=0.25, theta_per_day_pct=0.20,
        atrz_gate=-0.5, slope_gate_pct=-0.01,
        use_regime=1, regime_trend_only=0, vix_z_max=3.0,
        slippage_bps=7.0, vega_scale=0.03,
        scan_min_hit=55.0,
        scan_max_dd=1.0
    )
    sort_key = request.query_params.get("sort")
    rows = []
    for t in TOP150:
        r = compute_scan_for_ticker(t, PARAMS) or {}
        if not r:
            continue
        if r.get("hit_pct", 0) >= PARAMS["scan_min_hit"] and r.get("avg_dd_pct", 999) <= PARAMS["scan_max_dd"]:
            rows.append(r)

    rows.sort(key=lambda x: (x["avg_roi_pct"], x["hit_pct"], x["support"], x["stability"]), reverse=True)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "rows": _sort_rows(rows, sort_key),
            "ran_at": now_et().strftime("%Y-%m-%d %H:%M"),
            "note": f"TOP150 parity run • kept {len(rows)}",
        },
    )


def _coerce_scan_params(form: dict) -> dict:
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
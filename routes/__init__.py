import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import sqlite3
import smtplib
import ssl
from email.message import EmailMessage

import certifi
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from indices import SP100, TOP150, TOP250
from db import DB_PATH, get_db, get_settings
from scanner import compute_scan_for_ticker
from utils import now_et, TZ

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)


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


@router.get("/forward", response_class=HTMLResponse)
def forward_page(request: Request, db=Depends(get_db)):
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [dict(r) for r in db.fetchall()]
    tests = []
    for f in favs:
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
            "scan_min_hit": 0.0,
            "scan_max_dd": 100.0,
        }
        row = compute_scan_for_ticker(f["ticker"], params)
        ran_at = now_et().isoformat()
        result = {
            "id": f["id"],
            "ticker": f["ticker"],
            "direction": f["direction"],
            "interval": f["interval"],
            "rule": f["rule"],
            "avg_roi_pct": row.get("avg_roi_pct") if row else None,
            "hit_pct": row.get("hit_pct") if row else None,
            "avg_dd_pct": row.get("avg_dd_pct") if row else None,
            "ran_at": ran_at,
        }
        tests.append(result)
        if row:
            db.execute(
                "INSERT INTO forward_tests(fav_id, ran_at, avg_roi_pct, hit_pct, avg_dd_pct) VALUES (?, ?, ?, ?, ?)",
                (
                    f["id"],
                    ran_at,
                    row.get("avg_roi_pct"),
                    row.get("hit_pct"),
                    row.get("avg_dd_pct"),
                ),
            )
    db.connection.commit()
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

    if not t or not rule:
        return JSONResponse({"ok": False, "error": "missing ticker or rule"}, status_code=400)

    db.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES (?, ?, '15m', ?)",
        (t, direction, rule),
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

        started = now_et().isoformat()
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


@router.post("/scanner/run", response_class=HTMLResponse)
async def scanner_run(request: Request):
    """HTMX target. Reads form, normalizes params, runs the scan, and renders results.html."""
    form = await request.form()
    params = _coerce_scan_params(form)

    # Figure out ticker universe
    scan_type = params.get("scan_type", "scan150")
    single_ticker = (form.get("ticker") or "").strip().upper()
    if scan_type.lower() in ("single", "single_ticker") and single_ticker:
        tickers = [single_ticker]
    elif scan_type.lower() in ("sp100", "sp_100", "sp100_scan"):
        tickers = SP100
    elif scan_type.lower() in ("top250", "scan250", "options250"):
        tickers = TOP250
    else:
        # default Top 150
        tickers = TOP150

    # Support server-side sorting triggered by buttons
    sort_key = (form.get("sort") or "").strip().lower()
    if sort_key not in ("ticker", "roi", "hit"):
        sort_key = ""

    # Run the scan
    rows = []
    for t in tickers:
        r = compute_scan_for_ticker(t, params)
        if r:
            rows.append(r)

    # Optional filters (match desktop defaults)
    try:
        scan_min_hit = float(params.get("scan_min_hit", 0.0))
        scan_max_dd = float(params.get("scan_max_dd", 100.0))
    except Exception:
        scan_min_hit, scan_max_dd = 0.0, 100.0

    rows = [
        r for r in rows
        if (r.get("hit_pct", 0.0) >= scan_min_hit) and (r.get("avg_dd_pct", 100.0) <= scan_max_dd)
    ]

    # Sorting
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

    ctx = {
        "request": request,
        "rows": rows,
        "ran_at": datetime.now().strftime("%I:%M:%S %p").lstrip("0"),
        "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}"
    }

    if params.get("email_checkbox") == "on" and rows:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                st = get_settings(conn.cursor())
            # Build text and HTML bodies with a 5-column table
            hdr = f"PatternFinder Scan Results — {now_et().strftime('%Y-%m-%d %H:%M')}"
            lines = [hdr, "=" * len(hdr), f"Scan: {ctx['note']}", ""]

            cols = [
                ("ticker", "Ticker", "<"),
                ("direction", "Direction", "<"),
                ("hit_pct", "Hit%", ">"),
                ("avg_roi_pct", "ROI %", ">"),
                ("avg_dd_pct", "DD", ">"),
            ]
            widths = [len(label) for _, label, _ in cols]
            for r in rows:
                values = [
                    r.get("ticker", ""),
                    r.get("direction", ""),
                    f"{r.get('hit_pct', 0):.2f}",
                    f"{r.get('avg_roi_pct', 0):.2f}",
                    f"{r.get('avg_dd_pct', 0):.2f}",
                ]
                for i, val in enumerate(values):
                    widths[i] = max(widths[i], len(val))

            header = "  ".join(label.ljust(width) for (_, label, _), width in zip(cols, widths))
            lines.append(header)
            lines.append("  ".join("-" * width for width in widths))

            for r in rows:
                values = [
                    r.get("ticker", ""),
                    r.get("direction", ""),
                    f"{r.get('hit_pct', 0):.2f}",
                    f"{r.get('avg_roi_pct', 0):.2f}",
                    f"{r.get('avg_dd_pct', 0):.2f}",
                ]
                parts = []
                for (_, _, align), width, val in zip(cols, widths, values):
                    parts.append(val.ljust(width) if align == "<" else val.rjust(width))
                lines.append("  ".join(parts))

            lines.append("")
            lines.append("— Sent by PatternFinder")
            body = "\n".join(lines)

            html_lines = [
                "<html><body>",
                f"<p><strong>{hdr}</strong></p>",
                f"<p>Scan: {ctx['note']}</p>",
                "<table style='border-collapse:collapse;'>",
                "<thead><tr>",
                "<th style='text-align:left;padding:8px;'>Ticker</th>",
                "<th style='text-align:left;padding:8px;'>Direction</th>",
                "<th style='text-align:right;padding:8px;'>Hit%</th>",
                "<th style='text-align:right;padding:8px;'>ROI %</th>",
                "<th style='text-align:right;padding:8px;'>DD</th>",
                "</tr></thead>",
                "<tbody>",
            ]
            for r in rows:
                html_lines.append(
                    "<tr>"
                    f"<td style='padding:8px;border-top:1px solid #ddd;'>{r.get('ticker','')}</td>"
                    f"<td style='padding:8px;border-top:1px solid #ddd;'>{r.get('direction','')}</td>"
                    f"<td style='padding:8px;text-align:right;border-top:1px solid #ddd;'>{r.get('hit_pct',0):.2f}</td>"
                    f"<td style='padding:8px;text-align:right;border-top:1px solid #ddd;'>{r.get('avg_roi_pct',0):.2f}</td>"
                    f"<td style='padding:8px;text-align:right;border-top:1px solid #ddd;'>{r.get('avg_dd_pct',0):.2f}</td>"
                    "</tr>"
                )
            html_lines.extend([
                "</tbody></table>",
                "<p>— Sent by PatternFinder</p>",
                "</body></html>",
            ])
            html_body = "\n".join(html_lines)
            _send_email(st, "PatternFinder: Scan Results", body, html_body=html_body)
        except Exception as e:
            logger.error("scan email failed: %s", e)

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

    started_at = now_et().isoformat()
    db.execute(
        "INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count) VALUES (?, ?, ?, ?, ?, ?)",
        (started_at, scan_type, json.dumps(params), ",".join(universe), now_et().isoformat(), len(rows)),
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
            "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
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

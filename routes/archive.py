"""Archive-related routes for saving and viewing scan results."""

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from db import get_db
from utils import TZ, now_et

router = APIRouter()
templates = Jinja2Templates(directory="templates")


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
    """Save a completed scan run and its results to the archive."""
    try:
        payload = await request.json()
        params = payload.get("params", {}) or {}
        rows = payload.get("rows", []) or []
        if not rows:
            return JSONResponse({"ok": False, "error": "no rows"}, status_code=400)

        started = now_et().isoformat()
        finished = started
        scan_type = str(params.get("scan_type") or "scan150")
        universe = ",".join({r.get("ticker", "") for r in rows if r.get("ticker")})
        settings_json = json.dumps(params)

        db.execute(
            """
            INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count, settings_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (started, scan_type, settings_json, universe, finished, len(rows), settings_json),
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


@router.post("/runs/archive")
async def archive_run(request: Request, db=Depends(get_db)):
    """Compatibility endpoint for archiving runs."""
    payload = await request.json()
    scan_type = payload.get("scan_type", "")
    params = payload.get("params", {})
    rows = payload.get("rows", [])
    universe = payload.get("universe", [])

    started_at = now_et().isoformat()
    settings_json = json.dumps(params)
    db.execute(
        """
        INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count, settings_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (started_at, scan_type, settings_json, ",".join(universe), now_et().isoformat(), len(rows), settings_json),
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

    settings_items: List[Tuple[str, Any]] = []
    try:
        settings = json.loads(run["settings_json"] or "{}")
        for k, v in settings.items():
            settings_items.append((k.replace("_", " ").title(), v))
        settings_items.sort()
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
            "settings_items": settings_items,
        },
    )


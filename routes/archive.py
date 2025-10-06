"""Archive-related routes for saving and viewing scan results."""

import json
from datetime import datetime
from typing import Any, Dict, List, Mapping, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from db import get_db, row_to_dict
from utils import TZ, now_et
from .template_helpers import register_template_helpers

router = APIRouter()
templates = Jinja2Templates(directory="templates")
register_template_helpers(templates)


def _json_default(value: Any) -> str:
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def _serialize_json_blob(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            value = value.decode("latin-1", "ignore")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        return json.dumps(value, default=_json_default)
    except TypeError:
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return json.dumps(str(value))


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            value = value.decode("latin-1", "ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_archive_payload(
    payload: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    params_dict = _ensure_dict(payload.get("params"))

    settings_source: Any | None = None
    for key in ("settings_json", "settings", "params"):
        if key not in payload:
            continue
        candidate = payload.get(key)
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            settings_source = candidate
            break
        if isinstance(candidate, (bytes, bytearray)):
            try:
                candidate = candidate.decode("utf-8", "ignore")
            except Exception:
                candidate = candidate.decode("latin-1", "ignore")
        if isinstance(candidate, str):
            if not candidate.strip():
                continue
            settings_source = candidate
            break
        settings_source = candidate
        break

    if settings_source is None:
        settings_source = params_dict or {}

    if not params_dict:
        params_dict = _ensure_dict(settings_source)

    params_json_text = _serialize_json_blob(params_dict) or "{}"
    settings_json_text = _serialize_json_blob(settings_source) or params_json_text

    return params_dict, params_json_text, settings_json_text


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
        "SELECT id, started_at, scan_type, params_json, universe, "
        "finished_at, hit_count FROM runs ORDER BY id DESC LIMIT 200"
    )
    runs = [row_to_dict(r, db) for r in db.fetchall()]
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
    return templates.TemplateResponse(
        request, "archive.html", {"runs": runs, "active_tab": "archive"}
    )


@router.post("/archive/save")
async def archive_save(request: Request, db=Depends(get_db)):
    """Save a completed scan run and its results to the archive."""
    try:
        payload = await request.json()
        if not isinstance(payload, Mapping):
            return JSONResponse({"ok": False, "error": "invalid payload"}, status_code=400)

        rows = payload.get("rows", []) or []
        if not isinstance(rows, list) or not rows:
            return JSONResponse({"ok": False, "error": "no rows"}, status_code=400)

        params_dict, params_json_text, settings_json_text = _normalize_archive_payload(
            payload
        )

        started = now_et().isoformat()
        finished = started
        scan_type = str(
            params_dict.get("scan_type")
            or payload.get("scan_type")
            or "scan150"
        )
        universe = ",".join(
            {r.get("ticker", "") for r in rows if isinstance(r, Mapping) and r.get("ticker")}
        )

        db.execute(
            """
            INSERT INTO runs(
                started_at, scan_type, params_json, universe, finished_at,
                hit_count, settings_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started,
                scan_type,
                params_json_text,
                universe,
                finished,
                len(rows),
                settings_json_text,
            ),
        )
        run_id = db.lastrowid

        for r in rows:
            if not isinstance(r, Mapping):
                continue
            db.execute(
                """
                INSERT INTO run_results(
                    run_id, ticker, direction, avg_roi_pct, hit_pct, support,
                    avg_tt, avg_dd_pct, stability, rule
                )
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
    if not isinstance(payload, Mapping):
        return JSONResponse({"ok": False, "error": "invalid payload"}, status_code=400)

    scan_type = payload.get("scan_type", "")
    params_dict, params_json_text, settings_json_text = _normalize_archive_payload(
        payload
    )
    rows = payload.get("rows", [])
    universe = payload.get("universe", [])

    started_at = now_et().isoformat()
    db.execute(
        """
        INSERT INTO runs(
            started_at, scan_type, params_json, universe, finished_at,
            hit_count, settings_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started_at,
            scan_type,
            params_json_text,
            ",".join(universe),
            now_et().isoformat(),
            len(rows),
            settings_json_text,
        ),
    )
    run_id = db.lastrowid

    for r in rows:
        if not isinstance(r, Mapping):
            continue
        db.execute(
            """
            INSERT INTO run_results(
                run_id, ticker, direction, avg_roi_pct, hit_pct, support,
                avg_tt, avg_dd_pct, stability, rule
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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


@router.delete("/api/archive/{run_id}")
def archive_delete_run(run_id: int, db=Depends(get_db)):
    db.execute("SELECT id FROM runs WHERE id=?", (run_id,))
    if db.fetchone() is None:
        return JSONResponse({"ok": False, "error": "Run not found"}, status_code=404)
    db.execute("DELETE FROM run_results WHERE run_id=?", (run_id,))
    db.execute("DELETE FROM runs WHERE id=?", (run_id,))
    db.connection.commit()
    return {"ok": True, "run_id": run_id}


@router.post("/api/archive/clear")
def archive_clear_all(db=Depends(get_db)):
    db.execute("SELECT COUNT(*) FROM runs")
    row = db.fetchone()
    total = int(row[0]) if row else 0
    if total == 0:
        return {"ok": True, "cleared": 0}
    db.execute("DELETE FROM run_results WHERE run_id IN (SELECT id FROM runs)")
    db.execute("DELETE FROM runs")
    db.connection.commit()
    return {"ok": True, "cleared": total}


@router.get("/results/{run_id}", response_class=HTMLResponse)
def results_from_archive(request: Request, run_id: int, db=Depends(get_db)):
    db.execute("SELECT * FROM runs WHERE id=?", (run_id,))
    run_row = db.fetchone()
    if not run_row:
        return HTMLResponse("Run not found", status_code=404)
    run = row_to_dict(run_row, db)

    db.execute(
        """
        SELECT
            ticker, direction, avg_roi_pct, hit_pct, support, avg_tt,
            avg_dd_pct, stability, rule
        FROM run_results WHERE run_id=?
        """,
        (run_id,),
    )
    rows = [row_to_dict(r, db) for r in db.fetchall()]
    rows.sort(
        key=lambda r: (r["avg_roi_pct"], r["hit_pct"], r["support"], r["stability"]),
        reverse=True,
    )

    rule_summary = ""
    ran_at = ""
    meta: Dict[str, Any] = {}
    settings_items: List[Tuple[str, Any]] = []
    settings_json_pretty: str | None = None

    settings_dict: Dict[str, Any] | None = None
    raw_settings_text: str | None = None
    fallback_parsed: Any = None
    for source in (run.get("settings_json"), run.get("params_json")):
        if not source:
            continue
        if isinstance(source, str):
            raw_settings_text = raw_settings_text or source
            try:
                parsed = json.loads(source)
            except Exception:
                continue
        elif isinstance(source, dict):
            parsed = source
        else:
            continue
        if isinstance(parsed, dict):
            settings_dict = parsed
            break
        elif fallback_parsed is None:
            fallback_parsed = parsed

    if settings_dict:
        params = dict(settings_dict)
        meta = params.pop("_meta", {}) or {}
        rule_summary = _format_rule_summary(params)
        settings_items = list(params.items())
        if params:
            settings_json_pretty = json.dumps(params, indent=2)
    elif fallback_parsed is not None:
        try:
            settings_json_pretty = json.dumps(fallback_parsed, indent=2)
        except Exception:
            settings_json_pretty = str(fallback_parsed)
    elif raw_settings_text:
        settings_json_pretty = raw_settings_text

    try:
        dt = datetime.fromisoformat(run["started_at"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        else:
            dt = dt.astimezone(TZ)
        ran_at = dt.strftime("%m/%d/%y,%I:%M%p").replace("/0", "/").replace(",0", ",")
    except Exception:
        pass

    note = ""
    error_msg = None
    if isinstance(meta, dict):
        note = str(meta.get("note") or "")
        error_msg = meta.get("error")

    ctx = {
        "rows": rows,
        "scan_type": run.get("scan_type"),
        "universe_count": (
            len((run.get("universe") or "").split(",")) if run.get("universe") else 0
        ),
        "run_id": run_id,
        "active_tab": "archive",
        "ran_at": ran_at,
        "rule_summary": rule_summary,
        "settings_items": settings_items,
        "settings_json_pretty": settings_json_pretty,
        "note": note,
        "error": error_msg,
        "meta": meta,
    }
    return templates.TemplateResponse(request, "archive/results.html", ctx)

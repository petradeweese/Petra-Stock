from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from db import get_db, row_to_dict
from services.overnight import get_runner_state, resolve_overnight_symbols
from services.scanner_params import coerce_scan_params

# ruff: noqa: E501


logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/overnight")


ACTIVE_STATUSES: tuple[str, ...] = ("queued", "running", "paused")
FINISHED_STATUSES: tuple[str, ...] = ("complete", "canceled", "failed")


def _renumber(db, batch_id: str) -> None:
    rows = db.execute(
        "SELECT id FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    for idx, (iid,) in enumerate(rows, start=1):
        db.execute(
            "UPDATE overnight_items SET position=? WHERE id=?",
            (idx, iid),
        )
    db.connection.commit()


MAX_ITEMS = 50
MAX_PAYLOAD = 2000


@router.post("/validate")
def validate_item(payload: dict = Body(...), db=Depends(get_db)):
    item = coerce_scan_params(payload)
    resolution = resolve_overnight_symbols(item, db)
    return {
        "symbols_total": resolution.symbols_total,
        "message": resolution.message,
        "detail": resolution.detail,
        "universe": resolution.universe,
        "ticker": resolution.ticker,
    }


@router.post("/batches")
def create_batch(payload: dict = Body(...), db=Depends(get_db)):
    label = payload.get("label")
    note = payload.get("note")
    raw_items = payload.get("items", []) or []
    if len(raw_items) > MAX_ITEMS:
        return JSONResponse({"error": "too_many_items"}, status_code=400)
    batch_id = str(uuid4())
    db.execute(
        "INSERT INTO overnight_batches(id,label,note,status) VALUES(?,?,?, 'queued')",
        (batch_id, label, note),
    )
    for idx, raw in enumerate(raw_items, start=1):
        item = coerce_scan_params(raw)
        if len(json.dumps(item)) > MAX_PAYLOAD:
            return JSONResponse({"error": "payload_too_large"}, status_code=400)
        item_id = str(uuid4())
        db.execute(
            "INSERT INTO overnight_items(id,batch_id,position,payload_json,status) VALUES(?,?,?,?, 'queued')",
            (item_id, batch_id, idx, json.dumps(item)),
        )
    db.connection.commit()
    return JSONResponse(
        {"batch_id": batch_id, "queued": len(raw_items)}, status_code=201
    )


@router.get("", response_class=HTMLResponse)
def overnight_page(request: Request):
    return templates.TemplateResponse(
        "overnight.html", {"request": request, "active_tab": "overnight"}
    )


@router.get("/batches")
def list_batches(
    include_finished: bool = False,
    include_deleted: bool = False,
    db=Depends(get_db),
):
    clauses: list[str] = []
    params: list[object] = []
    if not include_deleted:
        clauses.append("b.deleted_at IS NULL")
        statuses = ACTIVE_STATUSES + (FINISHED_STATUSES if include_finished else ())
    else:
        statuses = ()
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"b.status IN ({placeholders})")
        params.extend(statuses)
    sql = """
        SELECT b.*,
            (SELECT COUNT(*) FROM overnight_items i WHERE i.batch_id=b.id) AS items_total,
            (SELECT COUNT(*) FROM overnight_items i WHERE i.batch_id=b.id AND i.status='complete') AS items_done
        FROM overnight_batches b
    """
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    db.execute(sql, tuple(params))
    batches = [row_to_dict(r, db) for r in db.fetchall()]
    db.execute(
        """
        SELECT COUNT(*) FROM overnight_batches
        WHERE status IN (?,?,?) AND deleted_at IS NULL
        """,
        FINISHED_STATUSES,
    )
    finished_count = db.fetchone()[0]
    return {
        "batches": batches,
        "runner": get_runner_state(),
        "finished_count": finished_count,
    }


@router.get("/batches/{batch_id}")
def batch_detail(batch_id: str, db=Depends(get_db)):
    db.execute("SELECT * FROM overnight_batches WHERE id=?", (batch_id,))
    batch = row_to_dict(db.fetchone(), db)
    db.execute(
        "SELECT * FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    )
    items = [row_to_dict(r, db) for r in db.fetchall()]
    return {"batch": batch, "items": items}


@router.delete("/batches/{batch_id}")
def delete_batch(batch_id: str, db=Depends(get_db)):
    row = db.execute(
        "SELECT status, deleted_at FROM overnight_batches WHERE id=?",
        (batch_id,),
    ).fetchone()
    if not row:
        return JSONResponse({"error": "not_found"}, status_code=404)
    status = row[0]
    if row[1]:
        return {"status": "already_deleted"}
    if status not in FINISHED_STATUSES:
        return JSONResponse({"error": "batch_active"}, status_code=409)
    deleted_at = datetime.utcnow().isoformat()
    db.execute(
        "UPDATE overnight_batches SET deleted_at=? WHERE id=?",
        (deleted_at, batch_id),
    )
    logger.info(
        json.dumps(
            {
                "type": "overnight_delete_batch",
                "batch_id": batch_id,
                "status": status,
                "deleted_at": deleted_at,
            }
        )
    )
    return {"status": "deleted"}


@router.post("/batches/clear_finished")
def clear_finished(db=Depends(get_db)):
    deleted_at = datetime.utcnow().isoformat()
    result = db.execute(
        """
        UPDATE overnight_batches
        SET deleted_at=?
        WHERE deleted_at IS NULL AND status IN (?,?,?)
        """,
        (deleted_at, *FINISHED_STATUSES),
    )
    cleared = result.rowcount or 0
    logger.info(
        json.dumps(
            {
                "type": "overnight_clear_finished",
                "count": cleared,
                "deleted_at": deleted_at,
            }
        )
    )
    return {"cleared": cleared}


@router.get("/batches/{batch_id}/csv", response_class=PlainTextResponse)
def batch_csv(batch_id: str, db=Depends(get_db)):
    db.execute(
        "SELECT position,payload_json,run_id FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    )
    rows = db.fetchall()
    lines = ["position,scan_type,ticker,run_id"]
    for pos, payload, run_id in rows:
        obj = json.loads(payload)
        lines.append(
            f"{pos},{obj.get('scan_type','')},{obj.get('ticker','')},{run_id or ''}"
        )
    return "\n".join(lines)


@router.get("/prefs")
def get_prefs(db=Depends(get_db)):
    row = db.execute(
        "SELECT window_start, window_end FROM overnight_prefs WHERE id=1",
    ).fetchone()
    if not row:
        db.execute("INSERT INTO overnight_prefs(id) VALUES (1)")
        db.connection.commit()
        row = ("01:00", "08:00")
    now = datetime.now().strftime("%H:%M")
    return {"window_start": row[0], "window_end": row[1], "now": now}


@router.post("/prefs")
def update_prefs(payload: dict = Body(...), db=Depends(get_db)):
    start = payload.get("window_start")
    end = payload.get("window_end")
    if not start or not end:
        return JSONResponse({"error": "invalid"}, status_code=400)
    db.execute("INSERT OR IGNORE INTO overnight_prefs(id) VALUES (1)")
    db.execute(
        "UPDATE overnight_prefs SET window_start=?, window_end=? WHERE id=1",
        (start, end),
    )
    db.connection.commit()
    return {"ok": True}


@router.post("/batches/{batch_id}/items")
def append_item(batch_id: str, item: dict = Body(...), db=Depends(get_db)):
    row = db.execute(
        "SELECT deleted_at FROM overnight_batches WHERE id=?",
        (batch_id,),
    ).fetchone()
    if not row or row[0]:
        return JSONResponse({"error": "not_found"}, status_code=404)
    item = coerce_scan_params(item)
    if len(json.dumps(item)) > MAX_PAYLOAD:
        return JSONResponse({"error": "payload_too_large"}, status_code=400)
    cur = db.execute(
        "SELECT COUNT(*) FROM overnight_items WHERE batch_id=?",
        (batch_id,),
    )
    if cur.fetchone()[0] >= MAX_ITEMS:
        return JSONResponse({"error": "too_many_items"}, status_code=400)
    item_id = str(uuid4())
    cur = db.execute(
        "SELECT COALESCE(MAX(position),0)+1 FROM overnight_items WHERE batch_id=?",
        (batch_id,),
    )
    pos = cur.fetchone()[0]
    db.execute(
        "INSERT INTO overnight_items(id,batch_id,position,payload_json,status) VALUES(?,?,?,?, 'queued')",
        (item_id, batch_id, pos, json.dumps(item)),
    )
    _renumber(db, batch_id)
    return {"item_id": item_id, "position": pos}


@router.delete("/items/{item_id}")
def delete_item(item_id: str, db=Depends(get_db)):
    row = db.execute(
        "SELECT batch_id FROM overnight_items WHERE id=?",
        (item_id,),
    ).fetchone()
    if not row:
        return {"ok": False}
    batch_id = row[0]
    db.execute("DELETE FROM overnight_items WHERE id=?", (item_id,))
    _renumber(db, batch_id)
    return {"ok": True}


@router.post("/batches/{batch_id}/reorder")
def reorder_items(batch_id: str, mapping: list[dict] = Body(...), db=Depends(get_db)):
    for obj in mapping:
        db.execute(
            "UPDATE overnight_items SET position=? WHERE id=? AND batch_id=?",
            (int(obj.get("position")), obj.get("item_id"), batch_id),
        )
    _renumber(db, batch_id)
    rows = db.execute(
        "SELECT id, position FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    ).fetchall()
    return [{"item_id": r[0], "position": r[1]} for r in rows]


@router.post("/batches/{batch_id}/start_now")
def start_now(batch_id: str, db=Depends(get_db)):
    cur = db.execute(
        "SELECT start_override, deleted_at, status FROM overnight_batches WHERE id=?",
        (batch_id,),
    )
    row = cur.fetchone()
    if not row or row[1]:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if row[0]:
        return {"status": "already_queued"}
    if row[2] in FINISHED_STATUSES:
        return JSONResponse({"error": "batch_finished"}, status_code=409)
    first_item = db.execute(
        """
        SELECT id, payload_json
        FROM overnight_items
        WHERE batch_id=? AND status='queued'
        ORDER BY position
        LIMIT 1
        """,
        (batch_id,),
    ).fetchone()
    if first_item:
        payload = json.loads(first_item[1])
        resolution = resolve_overnight_symbols(payload, db)
        if resolution.symbols_total == 0:
            message = resolution.detail or resolution.message or "Universe resolves to 0 symbols"
            return JSONResponse(
                {
                    "error": "invalid_universe",
                    "message": message,
                    "symbols_total": resolution.symbols_total,
                    "item_id": first_item[0],
                },
                status_code=400,
            )
    db.execute(
        "UPDATE overnight_batches SET start_override=1 WHERE id=?",
        (batch_id,),
    )
    logger.info(
        json.dumps(
            {
                "type": "overnight_override_start",
                "batch_id": batch_id,
                "when": datetime.utcnow().isoformat(),
                "user": "local",
            }
        )
    )
    running = db.execute(
        "SELECT id FROM overnight_batches WHERE status='running' AND deleted_at IS NULL",
    ).fetchone()
    if running and running[0] != batch_id:
        db.connection.commit()
        return {"status": "queued_after_current"}
    db.execute(
        "UPDATE overnight_batches SET status='queued' WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "started"}


@router.post("/batches/{batch_id}/pause")
def pause_batch(batch_id: str, db=Depends(get_db)):
    cur = db.execute(
        "SELECT status, deleted_at FROM overnight_batches WHERE id=?",
        (batch_id,),
    )
    row = cur.fetchone()
    if not row or row[1]:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if row[0] == "paused":
        return {"status": "paused"}
    db.execute(
        "UPDATE overnight_batches SET status='paused' WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "paused"}


@router.post("/batches/{batch_id}/resume")
def resume_batch(batch_id: str, db=Depends(get_db)):
    cur = db.execute(
        "SELECT status, deleted_at FROM overnight_batches WHERE id=?",
        (batch_id,),
    )
    row = cur.fetchone()
    if not row or row[1]:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if row[0] == "running":
        return {"status": "running"}
    db.execute(
        "UPDATE overnight_batches SET status='queued' WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "queued"}


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str, db=Depends(get_db)):
    row = db.execute(
        "SELECT deleted_at FROM overnight_batches WHERE id=?",
        (batch_id,),
    ).fetchone()
    if not row or row[0]:
        return JSONResponse({"error": "not_found"}, status_code=404)
    db.execute(
        "UPDATE overnight_batches SET status='canceled', start_override=0 WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "canceled"}


@router.get("/items/{item_id}/results")
def item_results(item_id: str, db=Depends(get_db)):
    db.execute(
        "SELECT run_id, status, error FROM overnight_items WHERE id=?",
        (item_id,),
    )
    row = db.fetchone()
    if not row:
        return {"rows": [], "error": "not_found", "status": "missing"}
    status = row["status"]
    error_msg = row["error"]
    run_id = row["run_id"]
    if status == "failed":
        return {"rows": [], "status": status, "error": error_msg}
    if run_id is None:
        return {"rows": [], "status": status}
    db.execute("SELECT * FROM run_results WHERE run_id=?", (run_id,))
    rows = [row_to_dict(r, db) for r in db.fetchall()]
    meta = {}
    db.execute("SELECT params_json FROM runs WHERE id=?", (run_id,))
    run_meta = db.fetchone()
    if run_meta:
        try:
            params = json.loads(run_meta["params_json"] or "{}")
            meta = params.get("_meta", {}) or {}
        except Exception:
            meta = {}
    return {"run_id": run_id, "rows": rows, "status": status, "meta": meta}

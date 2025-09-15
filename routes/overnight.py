from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Body, Depends
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from db import get_db, row_to_dict
from services.overnight import get_runner_state

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/overnight")


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


@router.post("/batches")
def create_batch(payload: dict = Body(...), db=Depends(get_db)):
    label = payload.get("label")
    note = payload.get("note")
    items = payload.get("items", []) or []
    if len(items) > MAX_ITEMS:
        return JSONResponse({"error": "too_many_items"}, status_code=400)
    batch_id = str(uuid4())
    db.execute(
        "INSERT INTO overnight_batches(id,label,note,status) VALUES(?,?,?, 'queued')",
        (batch_id, label, note),
    )
    for idx, item in enumerate(items, start=1):
        if len(json.dumps(item)) > MAX_PAYLOAD:
            return JSONResponse({"error": "payload_too_large"}, status_code=400)
        item_id = str(uuid4())
        db.execute(
            "INSERT INTO overnight_items(id,batch_id,position,payload_json,status) VALUES(?,?,?,?, 'queued')",
            (item_id, batch_id, idx, json.dumps(item)),
        )
    db.connection.commit()
    return {"batch_id": batch_id, "queued": len(items)}


@router.get("", response_class=HTMLResponse)
def overnight_page(request: Request):
    return templates.TemplateResponse(
        "overnight.html", {"request": request, "active_tab": "overnight"}
    )


@router.get("/batches")
def list_batches(db=Depends(get_db)):
    db.execute(
        """
        SELECT b.*, 
            (SELECT COUNT(*) FROM overnight_items i WHERE i.batch_id=b.id) AS items_total,
            (SELECT COUNT(*) FROM overnight_items i WHERE i.batch_id=b.id AND i.status='complete') AS items_done
        FROM overnight_batches b
        ORDER BY created_at
        """
    )
    batches = [row_to_dict(r, db) for r in db.fetchall()]
    return {"batches": batches, "runner": get_runner_state()}


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


@router.get("/batches/{batch_id}/csv", response_class=PlainTextResponse)
def batch_csv(batch_id: str, db=Depends(get_db)):
    db.execute(
        "SELECT position,payload_json,run_id FROM overnight_items WHERE batch_id=? ORDER BY position",
        (batch_id,),
    )
    rows = db.fetchall()
    lines = ["position,pattern,universe,run_id"]
    for pos, payload, run_id in rows:
        obj = json.loads(payload)
        lines.append(
            f"{pos},{obj.get('pattern','')},{obj.get('universe','')},{run_id or ''}"
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
        "SELECT start_override FROM overnight_batches WHERE id=?",
        (batch_id,),
    )
    row = cur.fetchone()
    if row and row[0]:
        return {"status": "already_queued"}
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
        "SELECT id FROM overnight_batches WHERE status='running'",
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
    cur = db.execute("SELECT status FROM overnight_batches WHERE id=?", (batch_id,))
    row = cur.fetchone()
    if row and row[0] == "paused":
        return {"status": "paused"}
    db.execute(
        "UPDATE overnight_batches SET status='paused' WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "paused"}


@router.post("/batches/{batch_id}/resume")
def resume_batch(batch_id: str, db=Depends(get_db)):
    cur = db.execute("SELECT status FROM overnight_batches WHERE id=?", (batch_id,))
    row = cur.fetchone()
    if row and row[0] == "running":
        return {"status": "running"}
    db.execute(
        "UPDATE overnight_batches SET status='queued' WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "queued"}


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str, db=Depends(get_db)):
    db.execute(
        "UPDATE overnight_batches SET status='canceled', start_override=0 WHERE id=?",
        (batch_id,),
    )
    db.connection.commit()
    return {"status": "canceled"}


@router.get("/items/{item_id}/results")
def item_results(item_id: str, db=Depends(get_db)):
    db.execute(
        "SELECT run_id FROM overnight_items WHERE id=?",
        (item_id,),
    )
    row = db.fetchone()
    if not row or row["run_id"] is None:
        return {"rows": []}
    run_id = row["run_id"]
    db.execute("SELECT * FROM run_results WHERE run_id=?", (run_id,))
    rows = [row_to_dict(r, db) for r in db.fetchall()]
    return {"run_id": run_id, "rows": rows}

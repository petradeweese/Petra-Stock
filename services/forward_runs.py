import hashlib
import sqlite3
from typing import Any

import db as db_module
from db import row_to_dict
from services.forward_summary import invalidate_forward_summary


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS forward_runs (
        favorite_id TEXT NOT NULL,
        entry_ts TEXT NOT NULL,
        entry_px REAL,
        exit_ts TEXT,
        exit_px REAL,
        outcome TEXT,
        roi REAL,
        tt_bars INTEGER,
        dd REAL,
        rule_hash TEXT,
        simulated INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(favorite_id, entry_ts)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forward_runs_fav_entry ON forward_runs(favorite_id, entry_ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_forward_runs_fav_exit ON forward_runs(favorite_id, exit_ts DESC);",
]


def ensure_forward_runs_schema(db: sqlite3.Cursor) -> None:
    conn = getattr(db, "connection", None)
    if conn is None:
        raise RuntimeError("forward_runs.ensure_forward_runs_schema requires a DB connection")

    if getattr(conn, "in_transaction", False):
        for statement in SCHEMA_STATEMENTS:
            db.execute(statement)
        return

    def _apply() -> None:
        db.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA_STATEMENTS:
                db.execute(statement)
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    db_module.retry_locked(_apply)


def _normalize_favorite_id(favorite_id: Any) -> str | None:
    if favorite_id in (None, ""):
        return None
    return str(favorite_id)


def forward_rule_hash(rule: Any) -> str | None:
    if rule in (None, ""):
        return None
    text = rule if isinstance(rule, str) else str(rule)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_forward_entry(
    db: sqlite3.Cursor,
    favorite_id: Any,
    entry_ts: str | None,
    entry_px: float | None,
    rule_hash: str | None,
    *,
    simulated: bool = False,
) -> None:
    ensure_forward_runs_schema(db)
    fav_key = _normalize_favorite_id(favorite_id)
    if not fav_key or not entry_ts:
        return
    try:
        entry_price = float(entry_px) if entry_px is not None else None
    except (TypeError, ValueError):
        entry_price = None
    db.execute(
        """
        INSERT INTO forward_runs (favorite_id, entry_ts, entry_px, rule_hash, simulated)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(favorite_id, entry_ts)
        DO UPDATE SET entry_px=excluded.entry_px,
                      rule_hash=excluded.rule_hash,
                      simulated=excluded.simulated
        """,
        (fav_key, entry_ts, entry_price, rule_hash, 1 if simulated else 0),
    )
    invalidate_forward_summary(fav_key)


def log_forward_exit(
    db: sqlite3.Cursor,
    favorite_id: Any,
    entry_ts: str | None,
    exit_ts: str | None,
    exit_px: float | None,
    outcome: str | None,
    roi: float | None,
    tt_bars: int | None,
    dd: float | None,
    *,
    simulated: bool = False,
) -> None:
    ensure_forward_runs_schema(db)
    fav_key = _normalize_favorite_id(favorite_id)
    if not fav_key or not entry_ts:
        return
    normalized_outcome = (outcome or "").strip().lower() or None
    if normalized_outcome == "target":
        normalized_outcome = "hit"
    try:
        exit_price = float(exit_px) if exit_px is not None else None
    except (TypeError, ValueError):
        exit_price = None
    try:
        roi_value = float(roi) if roi is not None else None
    except (TypeError, ValueError):
        roi_value = None
    try:
        bars_value = int(tt_bars) if tt_bars is not None else None
    except (TypeError, ValueError):
        bars_value = None
    try:
        dd_value = float(dd) if dd is not None else None
    except (TypeError, ValueError):
        dd_value = None
    db.execute(
        """
        UPDATE forward_runs
           SET exit_ts=?, exit_px=?, outcome=?, roi=?, tt_bars=?, dd=?, simulated=?
         WHERE favorite_id=? AND entry_ts=?
        """,
        (
            exit_ts,
            exit_price,
            normalized_outcome,
            roi_value,
            bars_value,
            dd_value,
            1 if simulated else 0,
            fav_key,
            entry_ts,
        ),
    )
    invalidate_forward_summary(fav_key)


def _fetch_history(
    db: sqlite3.Cursor,
    favorite_id: str,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = 0
    if limit_value <= 0:
        return []
    try:
        offset_value = int(offset)
    except (TypeError, ValueError):
        offset_value = 0
    if offset_value < 0:
        offset_value = 0
    db.execute(
        """
        SELECT favorite_id, entry_ts, entry_px, exit_ts, exit_px, outcome,
               roi, tt_bars, dd, rule_hash, simulated
          FROM forward_runs
         WHERE favorite_id=?
         ORDER BY entry_ts DESC
         LIMIT ? OFFSET ?
        """,
        (favorite_id, limit_value, offset_value),
    )
    rows = db.fetchall()
    return [row_to_dict(row, db) for row in rows]


def get_forward_history(
    favorite_id: str, limit: int = 10, offset: int = 0
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(
        db_module.DB_PATH,
        check_same_thread=False,
        isolation_level=None,
    )
    db_module._apply_pragmas(conn)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    ensure_forward_runs_schema(cur)
    try:
        return _fetch_history(cur, favorite_id, limit, offset)
    finally:
        conn.close()


def get_forward_history_for_cursor(
    db: sqlite3.Cursor,
    favorite_id: Any,
    limit: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    ensure_forward_runs_schema(db)
    fav_key = _normalize_favorite_id(favorite_id)
    if not fav_key:
        return []
    return _fetch_history(db, fav_key, limit, offset)

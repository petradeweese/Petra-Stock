from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

import db
from db import row_to_dict
from services.forward_summary import invalidate_forward_summary


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
) -> None:
    fav_key = _normalize_favorite_id(favorite_id)
    if not fav_key or not entry_ts:
        return
    try:
        entry_price = float(entry_px) if entry_px is not None else None
    except (TypeError, ValueError):
        entry_price = None
    db.execute(
        """
        INSERT INTO forward_runs (favorite_id, entry_ts, entry_px, rule_hash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(favorite_id, entry_ts)
        DO UPDATE SET entry_px=excluded.entry_px, rule_hash=excluded.rule_hash
        """,
        (fav_key, entry_ts, entry_price, rule_hash),
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
) -> None:
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
           SET exit_ts=?, exit_px=?, outcome=?, roi=?, tt_bars=?, dd=?
         WHERE favorite_id=? AND entry_ts=?
        """,
        (
            exit_ts,
            exit_price,
            normalized_outcome,
            roi_value,
            bars_value,
            dd_value,
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
               roi, tt_bars, dd, rule_hash
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
    conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
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
    fav_key = _normalize_favorite_id(favorite_id)
    if not fav_key:
        return []
    return _fetch_history(db, fav_key, limit, offset)


from __future__ import annotations
import json
import logging
import sqlite3
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"UP", "DOWN"}


def canonical_direction(value: Any) -> str | None:
    """Return a normalized direction (``UP``/``DOWN``) or ``None``."""

    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            value = value.decode("latin-1", "ignore")
    direction = str(value).strip().upper()
    if direction in _VALID_DIRECTIONS:
        return direction
    return None


def _direction_from_settings(raw: Any) -> str | None:
    """Extract a direction choice from a JSON snapshot if present."""

    if raw in (None, "", b""):
        return None

    data: Any
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
    else:
        data = raw

    if not isinstance(data, dict):
        return None

    return canonical_direction(data.get("direction"))


def _row_to_dict(row: Any, keys: Sequence[str]) -> dict[str, Any]:
    if row is None:
        return {}
    try:
        return dict(row)
    except Exception:
        result: dict[str, Any] = {}
        for idx, key in enumerate(keys):
            try:
                result[key] = row[idx]
            except Exception:
                result[key] = None
        return result


def ensure_favorite_directions(db: sqlite3.Cursor) -> None:
    """Backfill ``favorites.direction`` to a concrete side if missing."""

    try:
        db.execute("SELECT id, direction, settings_json_snapshot FROM favorites")
    except sqlite3.Error:
        logger.exception("Unable to query favorites for direction backfill")
        return

    rows = db.fetchall()
    if not rows:
        return

    updates: list[tuple[str, int]] = []
    for row in rows:
        row_dict = _row_to_dict(row, ("id", "direction", "settings_json_snapshot"))
        fav_id = row_dict.get("id")
        if fav_id is None:
            continue
        current = canonical_direction(row_dict.get("direction"))
        if current:
            continue
        inferred = _direction_from_settings(row_dict.get("settings_json_snapshot"))
        if inferred is None:
            inferred = "UP"
            logger.warning(
                "Favorite %s missing saved direction; defaulting to UP",
                fav_id,
            )
        try:
            fid_int = int(fav_id)
        except Exception:
            logger.exception("Unable to coerce favorite id %r to int", fav_id)
            continue
        updates.append((inferred, fid_int))

    if not updates:
        return

    try:
        db.executemany("UPDATE favorites SET direction=? WHERE id=?", updates)
        db.connection.commit()
    except sqlite3.Error:
        logger.exception("Failed to backfill favorite directions")


__all__ = ["canonical_direction", "ensure_favorite_directions"]

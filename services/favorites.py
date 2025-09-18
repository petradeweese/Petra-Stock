from __future__ import annotations
import json
import logging
import sqlite3
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"UP", "DOWN"}


class FavoriteMatch(dict):
    """Mapping wrapper exposing attribute-style access to favorite fields."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute fallback
            raise AttributeError(name) from exc

    @property
    def id(self) -> str:
        value = self.get("id") or self.get("favorite_id")
        return "" if value in (None, "") else str(value)


from db import DB_PATH, row_to_dict


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


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_rule(value: Any) -> str:
    if value in (None, "", b""):
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:  # pragma: no cover - fallback decoding
            value = value.decode("latin-1", "ignore")
    text = str(value).strip()
    return " ".join(text.split())


def match_favorite(row: Any) -> Optional[FavoriteMatch]:
    """Return the favorites record matching ``row`` if available."""

    if row is None:
        return None

    fav_id = _value(row, "favorite_id") or _value(row, "id")
    ticker_raw = _value(row, "ticker") or _value(row, "symbol")
    ticker = str(ticker_raw or "").strip().upper()
    direction = canonical_direction(_value(row, "direction"))
    if not direction:
        direction = "UP"
    rule = _normalize_rule(_value(row, "rule") or _value(row, "pattern"))

    query: str
    params: tuple[Any, ...]
    if fav_id not in (None, ""):
        query = "SELECT * FROM favorites WHERE id=? LIMIT 1"
        params = (fav_id,)
    else:
        if not (ticker and rule):
            return None
        query = (
            "SELECT * FROM favorites WHERE ticker=? AND direction=? AND rule=?"
            " ORDER BY id DESC LIMIT 1"
        )
        params = (ticker, direction, rule)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            db = conn.cursor()
            db.execute(query, params)
            match = db.fetchone()
            if not match:
                return None
            payload = row_to_dict(match, db)
    except sqlite3.Error:
        logger.exception("failed to match favorite")
        return None

    payload.setdefault("ticker", ticker or payload.get("ticker", ""))
    payload["ticker"] = str(payload.get("ticker", "")).strip().upper()
    payload.setdefault("direction", direction)
    payload["direction"] = canonical_direction(payload.get("direction")) or "UP"
    payload.setdefault("rule", rule)
    payload["rule"] = _normalize_rule(payload.get("rule"))
    if "greeks_profile_json" not in payload and payload.get("settings_json_snapshot") is not None:
        payload["greeks_profile_json"] = payload.get("settings_json_snapshot")
    fav_id_value = payload.get("id")
    if fav_id_value is not None:
        payload["favorite_id"] = fav_id_value
    return FavoriteMatch(payload)


__all__ = ["canonical_direction", "ensure_favorite_directions", "match_favorite", "FavoriteMatch"]

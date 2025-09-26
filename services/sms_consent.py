from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, time, timezone
from typing import Iterator

from db import get_db, row_to_dict

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"
DEFAULT_DAILY_LIMIT = 25
_FOOTER_TEXT = "STOP=opt-out, HELP=help, Msg&data rates may apply."


@contextmanager
def _db_context(db_cursor):
    if db_cursor is not None:
        yield db_cursor
        return

    gen: Iterator = get_db()
    cursor = next(gen)
    try:
        yield cursor
        try:
            next(gen)
        except StopIteration:
            pass
    finally:
        gen.close()


def normalize_phone(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if text.startswith("+"):
        digits = "+" + "".join(ch for ch in text[1:] if ch.isdigit())
    else:
        digits_only = "".join(ch for ch in text if ch.isdigit())
        if not digits_only:
            return ""
        if len(digits_only) == 10:
            digits = "+1" + digits_only
        elif len(digits_only) == 11 and digits_only.startswith("1"):
            digits = "+" + digits_only
        else:
            digits = "+" + digits_only
    if len(digits) < 4:
        return ""
    return digits


def append_footer(body: str) -> str:
    message = (body or "").strip()
    if not message:
        return _FOOTER_TEXT
    if _FOOTER_TEXT.lower() in message.lower():
        return message
    return f"{message}\n\n{_FOOTER_TEXT}"


def active_destinations(
    *, user_id: str | None = None, db_cursor=None
) -> list[dict[str, object]]:
    with _db_context(db_cursor) as cursor:
        if user_id:
            cursor.execute(
                "SELECT * FROM sms_consent WHERE user_id=? AND revoked_at IS NULL ORDER BY consent_at DESC",
                (str(user_id),),
            )
        else:
            cursor.execute(
                "SELECT * FROM sms_consent WHERE revoked_at IS NULL ORDER BY consent_at DESC"
            )
        rows = cursor.fetchall()
        return [row_to_dict(row, cursor) for row in rows]


def active_numbers(*, user_id: str | None = None, db_cursor=None) -> list[str]:
    destinations = active_destinations(user_id=user_id, db_cursor=db_cursor)
    return [str(dest.get("phone_e164") or "") for dest in destinations if dest.get("phone_e164")]


def latest_for_user(
    user_id: str | None,
    *,
    include_revoked: bool = False,
    db_cursor=None,
) -> dict[str, object]:
    target = str(user_id or DEFAULT_USER_ID)
    query = "SELECT * FROM sms_consent WHERE user_id=?"
    if not include_revoked:
        query += " AND revoked_at IS NULL"
    query += " ORDER BY consent_at DESC LIMIT 1"
    with _db_context(db_cursor) as cursor:
        cursor.execute(query, (target,))
        row = cursor.fetchone()
        return row_to_dict(row, cursor) if row else {}


def history_for_user(
    user_id: str | None,
    *,
    limit: int = 10,
    db_cursor=None,
) -> list[dict[str, object]]:
    target = str(user_id or DEFAULT_USER_ID)
    with _db_context(db_cursor) as cursor:
        cursor.execute(
            "SELECT * FROM sms_consent WHERE user_id=? ORDER BY consent_at DESC LIMIT ?",
            (target, int(limit)),
        )
        rows = cursor.fetchall()
        return [row_to_dict(row, cursor) for row in rows]


def latest_for_phone(
    phone: str,
    *,
    include_revoked: bool = True,
    db_cursor=None,
) -> dict[str, object]:
    normalized = normalize_phone(phone)
    if not normalized:
        return {}
    query = "SELECT * FROM sms_consent WHERE phone_e164=?"
    if not include_revoked:
        query += " AND revoked_at IS NULL"
    query += " ORDER BY consent_at DESC LIMIT 1"
    with _db_context(db_cursor) as cursor:
        cursor.execute(query, (normalized,))
        row = cursor.fetchone()
        return row_to_dict(row, cursor) if row else {}


def record_consent(
    user_id: str | None,
    phone: str,
    consent_text: str,
    *,
    ip: str | None = None,
    user_agent: str | None = None,
    verification_id: str | None = None,
    method: str = "settings",
    db_cursor=None,
) -> dict[str, object]:
    normalized = normalize_phone(phone)
    if not normalized:
        raise ValueError("Invalid phone number")
    consent_text = (consent_text or "").strip()
    if not consent_text:
        raise ValueError("Consent text required")
    method_value = (method or "settings").strip() or "settings"
    user_value = str(user_id or DEFAULT_USER_ID)
    now_iso = datetime.now(timezone.utc).isoformat()
    with _db_context(db_cursor) as cursor:
        cursor.execute(
            "UPDATE sms_consent SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
            (now_iso, user_value),
        )
        cursor.execute(
            """
            INSERT INTO sms_consent (
                user_id, phone_e164, consent_text, consent_at, ip, user_agent, method, verification_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_value,
                normalized,
                consent_text,
                now_iso,
                ip,
                user_agent,
                method_value,
                verification_id,
            ),
        )
        inserted_id = cursor.lastrowid
        cursor.execute("SELECT * FROM sms_consent WHERE id=?", (inserted_id,))
        row = cursor.fetchone()
        result = row_to_dict(row, cursor) if row else {}
    logger.info(
        "sms_consent_recorded",
        extra={
            "user_id": user_value,
            "phone": normalized,
            "method": method_value,
        },
    )
    return result


def revoke_phone(
    phone: str,
    *,
    user_id: str | None = None,
    db_cursor=None,
) -> bool:
    normalized = normalize_phone(phone)
    if not normalized:
        return False
    now_iso = datetime.now(timezone.utc).isoformat()
    params: list[str] = [now_iso, normalized]
    query = "UPDATE sms_consent SET revoked_at=? WHERE phone_e164=? AND revoked_at IS NULL"
    if user_id:
        query += " AND user_id=?"
        params.append(str(user_id))
    with _db_context(db_cursor) as cursor:
        cursor.execute(query, tuple(params))
        updated = cursor.rowcount
    if updated:
        logger.info("sms_consent_revoked", extra={"phone": normalized, "user_id": user_id})
    return bool(updated)


def allow_sending(
    phone: str,
    *,
    user_id: str | None = None,
    limit_per_day: int = DEFAULT_DAILY_LIMIT,
    db_cursor=None,
) -> tuple[bool, dict[str, object] | None]:
    normalized = normalize_phone(phone)
    if not normalized:
        return False, None
    with _db_context(db_cursor) as cursor:
        params: list[str] = [normalized]
        query = "SELECT * FROM sms_consent WHERE phone_e164=? AND revoked_at IS NULL"
        if user_id:
            query += " AND user_id=?"
            params.append(str(user_id))
        query += " ORDER BY consent_at DESC LIMIT 1"
        cursor.execute(query, tuple(params))
        row = cursor.fetchone()
        if not row:
            return False, None
        consent = row_to_dict(row, cursor)
        effective_user = str(consent.get("user_id") or user_id or DEFAULT_USER_ID)
        if limit_per_day:
            start = datetime.now(timezone.utc)
            start_of_day = datetime.combine(start.date(), time(0, tzinfo=timezone.utc))
            cursor.execute(
                "SELECT COUNT(*) FROM sms_delivery_log WHERE user_id=? AND phone_e164=? AND sent_at >= ?",
                (effective_user, normalized, start_of_day.isoformat()),
            )
            count_row = cursor.fetchone()
            count = count_row[0] if count_row else 0
            if count >= limit_per_day:
                logger.info(
                    "sms_consent_rate_limited",
                    extra={"user_id": effective_user, "phone": normalized, "limit": limit_per_day},
                )
                return False, consent
        return True, consent


def record_delivery(
    phone: str,
    user_id: str | None,
    body: str,
    *,
    message_type: str = "alert",
    db_cursor=None,
) -> None:
    normalized = normalize_phone(phone)
    if not normalized:
        return
    digest = hashlib.sha256((body or "").encode("utf-8")).hexdigest()[:32]
    now_iso = datetime.now(timezone.utc).isoformat()
    user_value = str(user_id or DEFAULT_USER_ID)
    with _db_context(db_cursor) as cursor:
        cursor.execute(
            """
            INSERT INTO sms_delivery_log (user_id, phone_e164, sent_at, message_type, body_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_value, normalized, now_iso, message_type, digest),
        )


__all__ = [
    "DEFAULT_DAILY_LIMIT",
    "DEFAULT_USER_ID",
    "active_destinations",
    "active_numbers",
    "allow_sending",
    "append_footer",
    "history_for_user",
    "latest_for_phone",
    "latest_for_user",
    "normalize_phone",
    "record_consent",
    "record_delivery",
    "revoke_phone",
]

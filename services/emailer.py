"""Shared helpers for sending transactional emails."""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Mapping, Optional

from db import DB_PATH, get_settings
from services.notify import send_email_smtp

logger = logging.getLogger(__name__)


def _normalize_recipients(recipients: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for entry in recipients:
        if not entry:
            continue
        addr = entry.strip()
        if addr:
            cleaned.append(addr)
    return cleaned


def load_smtp_settings() -> dict[str, object]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        db = conn.cursor()
        row = get_settings(db) or {}
    return row


def _smtp_config() -> dict[str, object]:
    return load_smtp_settings()


def send_email(
    mail_from: str,
    recipients: Iterable[str],
    subject: str,
    html_body: str,
    *,
    context: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Send an HTML email using the stored SMTP configuration."""

    to = _normalize_recipients(recipients)
    if not to:
        logger.warning("emailer_skip no_recipients subject=%s", subject)
        return {"ok": False, "error": "no_recipients"}

    cfg = _smtp_config()
    host = str(cfg.get("smtp_host") or "").strip()
    port_raw = cfg.get("smtp_port")
    try:
        port = int(port_raw) if port_raw is not None else 0
    except (TypeError, ValueError):
        port = 0
    user = str(cfg.get("smtp_user") or "").strip()
    password = str(cfg.get("smtp_pass") or "")
    configured_from = str(cfg.get("mail_from") or "").strip()

    sender = mail_from.strip() if mail_from else configured_from
    if not sender:
        logger.warning("emailer_skip missing_from subject=%s", subject)
        return {"ok": False, "error": "missing_sender"}

    if not host or port <= 0:
        logger.warning("emailer_skip smtp_not_configured subject=%s", subject)
        return {"ok": False, "error": "smtp_not_configured"}

    result = send_email_smtp(
        host,
        port,
        user,
        password,
        sender,
        to,
        subject,
        html_body,
        context=context,
    )
    if not result.get("ok"):
        logger.warning("emailer_error subject=%s error=%s", subject, result.get("error"))
    else:
        logger.info(
            "emailer_sent subject=%s recipients=%d message_id=%s",
            subject,
            len(to),
            result.get("message_id"),
        )
    return result


__all__ = ["load_smtp_settings", "send_email"]

"""Notification helpers for sending email via SMTP."""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from typing import Any, Dict, List, Mapping, Optional

import certifi


logger = logging.getLogger(__name__)


def _log_delivery(
    success: bool,
    recipients: List[str],
    context: Optional[Mapping[str, object]],
) -> None:
    if not context:
        return
    tag = "alert_delivery_ok" if success else "alert_delivery_fail"
    base = {k: v for k, v in context.items() if v is not None}
    for dest in recipients:
        payload = dict(base)
        payload["to"] = dest
        logger.info(tag, extra=payload)


def send_email_smtp(
    host: str,
    port: int,
    user: str,
    password: str,
    mail_from: str,
    to: List[str],
    subject: str,
    body: str,
    *,
    context: Optional[Mapping[str, object]] = None,
    raise_exceptions: bool = False,
) -> Dict[str, Any]:
    """Send an email using SMTP with STARTTLS.

    Parameters mirror the fields stored in the ``settings`` table.  The
    connection uses STARTTLS which is compatible with providers such as Gmail
    when ``port`` is 587.  When ``raise_exceptions`` is true ``SMTPException``
    instances are re-raised to allow callers to implement retry logic.
    """

    display_name, from_email = parseaddr(mail_from)
    domain_hint = from_email.split("@", 1)[-1] if "@" in from_email else None
    message_id = make_msgid(domain=domain_hint)
    msg = EmailMessage()
    msg["Message-ID"] = message_id
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject
    msg["From"] = mail_from or user
    msg["To"] = ", ".join(to)
    msg.set_content(body)

    tls_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP(host, int(port), timeout=20) as server:
            server.ehlo()
            server.starttls(context=tls_context)
            server.ehlo()
            if user:
                server.login(user, password)
            server.send_message(msg, from_addr=from_email or user, to_addrs=to)
    except Exception as exc:  # pragma: no cover - network interaction
        _log_delivery(False, to, context)
        if raise_exceptions and isinstance(exc, smtplib.SMTPException):
            raise
        return {"ok": False, "error": str(exc)}

    _log_delivery(True, to, context)
    return {"ok": True, "provider": "smtp", "message_id": message_id}

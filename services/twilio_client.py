"""Thin Twilio client wrapper used for Favorites alerts."""
from __future__ import annotations

import logging
from typing import Mapping, Optional

from config import settings

try:  # pragma: no cover - dependency import guard
    from twilio.base.exceptions import TwilioException
    from twilio.rest import Client
except Exception:  # pragma: no cover - allow running without Twilio installed
    Client = None  # type: ignore[assignment]
    TwilioException = Exception  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_CLIENT: Optional[Client] = None
_ENABLED: Optional[bool] = None
_FROM_NUMBER: Optional[str] = None


def _log_delivery(success: bool, to: str, context: Optional[Mapping[str, object]]) -> None:
    if not context:
        return
    payload = {k: v for k, v in context.items() if v is not None}
    payload["to"] = to
    tag = "alert_delivery_ok" if success else "alert_delivery_fail"
    logger.info(tag, extra=payload)


def _initialize() -> None:
    """Create the Twilio client on first use."""

    global _CLIENT, _ENABLED, _FROM_NUMBER
    if _ENABLED is not None:
        return

    account_sid = settings.twilio_account_sid
    auth_token = settings.twilio_auth_token
    from_number = settings.twilio_from_number

    if not (account_sid and auth_token and from_number and Client is not None):
        logger.info("twilio disabled: missing configuration or dependency")
        _ENABLED = False
        _CLIENT = None
        _FROM_NUMBER = None
        return

    try:
        _CLIENT = Client(account_sid, auth_token)
        _FROM_NUMBER = from_number
        _ENABLED = True
        logger.debug("twilio client initialized")
    except Exception:  # pragma: no cover - network interaction
        logger.exception("unable to initialize twilio client")
        _CLIENT = None
        _FROM_NUMBER = None
        _ENABLED = False


def is_enabled() -> bool:
    """Return ``True`` if the Twilio client is configured for sends."""

    _initialize()
    return bool(_ENABLED and _CLIENT and _FROM_NUMBER)


def send_mms(
    to: str,
    body: str,
    *,
    context: Optional[Mapping[str, object]] = None,
) -> bool:
    """Send an MMS ``body`` to ``to``. Return ``False`` on failures."""

    _initialize()
    if not _ENABLED or _CLIENT is None or not _FROM_NUMBER:
        return False

    if not to:
        logger.warning("twilio send skipped: missing destination")
        return False

    try:
        message = _CLIENT.messages.create(to=to, from_=_FROM_NUMBER, body=body)
    except TwilioException:  # pragma: no cover - network interaction
        logger.exception("twilio send failed")
        _log_delivery(False, to, context)
        return False
    except Exception:  # pragma: no cover - network interaction
        logger.exception("unexpected error sending twilio message")
        _log_delivery(False, to, context)
        return False

    if not getattr(message, "sid", None):
        logger.warning("twilio send returned no sid")
        _log_delivery(False, to, context)
        return False

    logger.info("twilio send ok sid=%s to=%s", message.sid, to)
    _log_delivery(True, to, context)
    return True


__all__ = ["is_enabled", "send_mms"]

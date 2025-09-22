"""Thin Twilio client wrapper used for Favorites alerts."""
from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

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
_VERIFY_ENABLED: Optional[bool] = None
_FROM_NUMBER: Optional[str] = None
_VERIFY_SERVICE: Optional[str] = None


def _log_delivery(success: bool, to: str, context: Optional[Mapping[str, object]]) -> None:
    if not context:
        return
    payload = {k: v for k, v in context.items() if v is not None}
    payload["to"] = to
    tag = "alert_delivery_ok" if success else "alert_delivery_fail"
    logger.info(tag, extra=payload)


def _initialize() -> None:
    """Create the Twilio client on first use."""

    global _CLIENT, _ENABLED, _FROM_NUMBER, _VERIFY_ENABLED, _VERIFY_SERVICE
    if _CLIENT is not None and _ENABLED is not None and _VERIFY_ENABLED is not None:
        return

    account_sid = settings.twilio_account_sid
    auth_token = settings.twilio_auth_token
    from_number = settings.twilio_from_number
    verify_service = getattr(settings, "twilio_verify_service_sid", "")

    if not (account_sid and auth_token and Client is not None):
        logger.info("twilio disabled: missing configuration or dependency")
        _ENABLED = False
        _VERIFY_ENABLED = False
        _CLIENT = None
        _FROM_NUMBER = None
        _VERIFY_SERVICE = None
        return

    try:
        _CLIENT = Client(account_sid, auth_token)
        _FROM_NUMBER = from_number or None
        _VERIFY_SERVICE = verify_service or None
        _ENABLED = bool(_FROM_NUMBER)
        _VERIFY_ENABLED = bool(_VERIFY_SERVICE)
        logger.debug(
            "twilio client initialized messaging=%s verify=%s",
            _ENABLED,
            _VERIFY_ENABLED,
        )
    except Exception:  # pragma: no cover - network interaction
        logger.exception("unable to initialize twilio client")
        _CLIENT = None
        _FROM_NUMBER = None
        _ENABLED = False
        _VERIFY_SERVICE = None
        _VERIFY_ENABLED = False


def is_enabled() -> bool:
    """Return ``True`` if the Twilio client is configured for sends."""

    _initialize()
    return bool(_ENABLED and _CLIENT and _FROM_NUMBER)


def is_verify_enabled() -> bool:
    """Return ``True`` when Twilio Verify can be used."""

    _initialize()
    return bool(_VERIFY_ENABLED and _CLIENT and _VERIFY_SERVICE)


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


def start_verification(to: str, *, channel: str = "sms") -> Optional[str]:
    """Trigger a verification code send via Twilio Verify."""

    _initialize()
    if not is_verify_enabled():
        logger.info("twilio verify start skipped: not configured")
        return None

    assert _CLIENT is not None  # for mypy
    assert _VERIFY_SERVICE is not None

    try:
        verification = _CLIENT.verify.services(_VERIFY_SERVICE).verifications.create(
            to=to,
            channel=channel,
        )
    except TwilioException:  # pragma: no cover - network interaction
        logger.exception("twilio verify start failed")
        return None
    except Exception:  # pragma: no cover - network interaction
        logger.exception("unexpected error starting twilio verification")
        return None

    sid = getattr(verification, "sid", None)
    status = getattr(verification, "status", "")
    logger.info(
        "twilio verify started", extra={"sid": sid, "to": to, "status": status}
    )
    return sid if isinstance(sid, str) else None


def check_verification(to: str, code: str) -> Tuple[bool, Optional[str]]:
    """Validate a verification ``code`` for ``to``."""

    _initialize()
    if not is_verify_enabled():
        logger.info("twilio verify check skipped: not configured")
        return False, None

    assert _CLIENT is not None
    assert _VERIFY_SERVICE is not None

    try:
        verification = _CLIENT.verify.services(_VERIFY_SERVICE).verification_checks.create(
            to=to,
            code=code,
        )
    except TwilioException:  # pragma: no cover - network interaction
        logger.exception("twilio verify check failed")
        return False, None
    except Exception:  # pragma: no cover - network interaction
        logger.exception("unexpected error checking twilio verification")
        return False, None

    status = getattr(verification, "status", "")
    sid = getattr(verification, "sid", None)
    ok = isinstance(status, str) and status.lower() == "approved"
    if ok:
        logger.info("twilio verify approved sid=%s to=%s", sid, to)
    else:
        logger.info("twilio verify rejected status=%s to=%s", status, to)
    return ok, sid if isinstance(sid, str) else None


__all__ = [
    "check_verification",
    "is_enabled",
    "is_verify_enabled",
    "send_mms",
    "start_verification",
]

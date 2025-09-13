"""Utilities for formatting Favorite Alert messages for email-to-SMS gateways.

This module defines carrier domain mappings and helpers for generating SMS
and MMS message bodies according to the project specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr

# Mapping of supported carriers to their email gateway domains.  The ``sms``
# domain is used for short 160 character messages while ``mms`` allows longer
# multi-line messages.
CARRIER_DOMAINS: dict[str, dict[str, str]] = {
    "att": {"sms": "txt.att.net", "mms": "mms.att.net"},
    # Mint Mobile uses the T‑Mobile gateway
    "mint": {"sms": "tmomail.net", "mms": "tmomail.net"},
    "tmobile": {"sms": "tmomail.net", "mms": "tmomail.net"},
    "verizon": {"sms": "vtext.com", "mms": "vzwpix.com"},
}

# Flat set of all carrier domains for quick membership tests
_CARRIER_DOMAIN_SET = {
    domain
    for info in CARRIER_DOMAINS.values()
    for domain in info.values()
}

def build_recipient(
    number: str,
    carrier: str,
    *,
    mms: bool = False,
    custom_domain: str | None = None,
) -> str:
    """Return an email address for the given phone ``number`` and ``carrier``.

    ``carrier`` may be one of the keys in :data:`CARRIER_DOMAINS` or ``"custom"``
    to supply an arbitrary ``custom_domain``.  ``mms`` selects the MMS gateway
    when available.
    """

    number = number.replace("-", "").replace(" ", "")
    if carrier == "custom":
        if not custom_domain:
            raise ValueError("custom_domain required for custom carrier")
        domain = custom_domain
    else:
        info = CARRIER_DOMAINS.get(carrier.lower())
        if not info:
            raise KeyError(f"unsupported carrier: {carrier}")
        domain = info["mms" if mms else "sms"]
    return f"{number}@{domain}"


@dataclass
class AlertDetails:
    ticker: str
    direction: str  # "UP" or "DOWN"
    hit: float
    target: float
    stop: float
    expiry: str  # expected MM/DD string
    strike: str  # e.g. "190C" or "325P"
    hit_pct: int
    support: int | None = None


def format_sms(details: AlertDetails) -> str:
    """Return a compact single line SMS alert (<=160 chars)."""

    direction = details.direction.upper()
    msg = (
        f"{details.ticker} {direction} hit {details.hit:.2f} | "
        f"T:{details.target:.1f} S:{details.stop:.1f} | "
        f"Exp {details.expiry} {details.strike} | Hit%:{details.hit_pct}"
    )
    if len(msg) > 160:
        raise ValueError("SMS alert exceeds 160 characters")
    return msg


def format_mms(details: AlertDetails) -> tuple[str, str]:
    """Return (subject, body) for an MMS alert."""

    subject = f"Pattern Alert: {details.ticker} {details.direction.upper()}"
    body_lines = [
        f"{details.ticker} {details.direction.upper()} hit {details.hit:.2f}",
        f"Target {details.target:.1f} | Stop {details.stop:.1f}",
        f"Expiry {details.expiry} {details.strike}",
        f"Hit% {details.hit_pct}"
        + (f" | Support {details.support}" if details.support is not None else ""),
    ]
    return subject, "\n".join(body_lines)


def is_carrier_address(address: str) -> bool:
    """Return ``True`` if ``address`` appears to be an email-to-SMS/MMS gateway."""

    _, email = parseaddr(address)
    if "@" not in email:
        return False
    domain = email.split("@", 1)[1].lower()
    return domain in _CARRIER_DOMAIN_SET

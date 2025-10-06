from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from fastapi.templating import Jinja2Templates

from services import config as services_config


def fmt_percent(value: Any, decimals: int = 2) -> str:
    """Format ``value`` as a percentage string with a ``0.00%`` fallback."""

    try:
        decimals_int = max(0, int(decimals))
    except (TypeError, ValueError):
        decimals_int = 2

    if value is None or (isinstance(value, float) and math.isnan(value)):
        num = 0.0
    else:
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = 0.0
        else:
            if not math.isfinite(num):
                num = 0.0

    if -1.0 <= num <= 1.0:
        num *= 100.0

    return f"{num:.{decimals_int}f}%"


def fmt_recent3(entries: Any) -> str:
    """Render the last three trade outcomes into a compact pipe-delimited string."""

    if not entries:
        return ""

    parts: list[str] = []
    for raw in entries:
        entry = raw or {}
        date_val = entry.get("date")
        date_str = "" if date_val is None else str(date_val).strip()
        outcome_raw = entry.get("outcome") or ""
        outcome = str(outcome_raw).strip().replace("\n", " ")

        tt_raw = entry.get("tt", 0)
        if tt_raw is None or pd.isna(tt_raw):
            tt_val = 0
        else:
            try:
                tt_val = int(round(float(tt_raw)))
            except (TypeError, ValueError):
                tt_val = 0

        roi_raw = entry.get("roi", 0.0)
        if roi_raw is None or pd.isna(roi_raw):
            roi_val = 0.0
        else:
            try:
                roi_val = float(roi_raw)
            except (TypeError, ValueError):
                roi_val = 0.0

        sign = "+" if roi_val >= 0 else ""
        safe_date = (date_str or "").replace("\n", " ")
        prefix = " ".join(part for part in [safe_date, outcome] if part)
        parts.append(f"{prefix} {sign}{roi_val * 100:.2f}% @{tt_val}b".strip())

    return " | ".join(parts)


def register_template_helpers(templates: Jinja2Templates) -> None:
    """Attach shared filters/globals to ``templates`` if not already registered."""

    templates.env.filters["fmt_percent"] = fmt_percent
    templates.env.globals["_fmt_recent3"] = fmt_recent3
    templates.env.globals["current_year"] = (
        lambda: datetime.now(timezone.utc).year
    )
    templates.env.globals["sms_frequency_copy"] = sms_frequency_copy
    templates.env.globals["business_contact"] = business_contact


def sms_frequency_copy() -> str:
    limit = max(1, int(getattr(services_config, "SMS_MAX_PER_MONTH", 50)))
    return f"No more than {limit} texts/month. Msg & data rates may apply."


def business_contact() -> Dict[str, str]:
    phone = str(getattr(services_config, "BUSINESS_PHONE", "")).strip()
    if not phone:
        phone = "+1 (555) 555-1212"
    tel_href = re.sub(r"[^0-9+]", "", phone)
    if tel_href and not tel_href.startswith("+"):
        tel_href = f"+{tel_href}"

    address_1 = str(getattr(services_config, "BUSINESS_ADDRESS_1", "")).strip()
    address_2 = str(getattr(services_config, "BUSINESS_ADDRESS_2", "")).strip()
    city = str(getattr(services_config, "BUSINESS_CITY", "")).strip()
    region = str(getattr(services_config, "BUSINESS_REGION", "")).strip()
    postal = str(getattr(services_config, "BUSINESS_POSTAL", "")).strip()

    return {
        "name": "Petra Stock, LLC",
        "phone": phone,
        "phone_href": tel_href or phone,
        "address_1": address_1 or "123 Example Street",
        "address_2": address_2,
        "city": city or "City",
        "region": region or "ST",
        "postal": postal or "00000",
    }


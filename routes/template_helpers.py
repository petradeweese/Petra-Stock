from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi.templating import Jinja2Templates


def fmt_percent(value: Any, decimals: int = 2) -> str:
    """Format ``value`` as a percentage string with a ``0.00%`` fallback."""

    try:
        decimals_int = max(0, int(decimals))
    except (TypeError, ValueError):
        decimals_int = 2

    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    else:
        if not math.isfinite(num):
            num = 0.0

    if num <= 1.0:
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
        outcome = str(outcome_raw).strip()

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
        prefix = " ".join(part for part in [date_str, outcome] if part)
        parts.append(f"{prefix} {sign}{roi_val * 100:.2f}% @{tt_val}b".strip())

    return " | ".join(parts)


def register_template_helpers(templates: Jinja2Templates) -> None:
    """Attach shared filters/globals to ``templates`` if not already registered."""

    templates.env.filters["fmt_percent"] = fmt_percent
    templates.env.globals["_fmt_recent3"] = fmt_recent3


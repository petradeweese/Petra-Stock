"""Scheduled job and API endpoint for the intraday forecast email."""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Callable, Iterable

from fastapi import APIRouter, Query

from services.emailer import send_email
from services.forecast_selector import (
    is_trading_day,
    select_forecast_top5,
)
from utils import TZ, now_et

logger = logging.getLogger(__name__)

router = APIRouter()


def _env_recipients() -> list[str]:
    raw = os.getenv("FORECAST_EMAIL_RECIPIENTS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_sender() -> str:
    default = "alerts@example.com"
    return os.getenv("ALERTS_FROM_EMAIL", default).strip() or default


def _format_pct(value: float | None, *, signed: bool = False, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    fmt = f"{{value:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value=value)


def _format_range(values: Iterable[float]) -> str:
    items = list(values)
    if len(items) != 2:
        return "n/a"
    left = _format_pct(items[0], signed=True)
    right = _format_pct(items[1], signed=True)
    if left == "n/a" or right == "n/a":
        return "n/a"
    return f"{left}–{right}"


def _pct_with_unit(value: float | None, *, signed: bool = False, digits: int = 1) -> str:
    pct = _format_pct(value, signed=signed, digits=digits)
    return pct if pct == "n/a" else f"{pct}%"


def render_email(asof: dt.datetime, top5: list[dict], run_label: str) -> tuple[str, str]:
    """Return the email subject and HTML body for the forecast run."""

    asof = asof if asof.tzinfo else asof.replace(tzinfo=dt.timezone.utc)
    asof_et = asof.astimezone(TZ)
    subject = f"Today’s Forecast — Top 5 ({run_label})"
    header_time = asof_et.strftime("%Y-%m-%d %H:%M")
    time_param = asof_et.strftime("%H%M")
    header = "Today’s Forecast — Top 5 intraday matches"

    parts = [
        "<html><body>",
        f"<h2>{header}</h2>",
        f"<p>Run label: <strong>{run_label}</strong><br/>As of {header_time} ET</p>",
        "<hr/>",
    ]

    for row in top5:
        ticker = row.get("ticker", "?")
        conf = row.get("confidence") or 0.0
        conf_pct = round(conf * 100)
        n = row.get("n") or 0
        median = _pct_with_unit(row.get("median_close_pct"), signed=True)
        iqr = _format_range(row.get("iqr_close_pct", []))
        high_tail = _pct_with_unit(row.get("median_high_pct"), signed=True)
        low_tail = _pct_with_unit(row.get("median_low_pct"), signed=True)
        implied_move = _pct_with_unit(row.get("implied_eod_move_pct"), digits=2)
        edge_value = row.get("edge") if row.get("edge") is not None else 0.0
        edge_str = _pct_with_unit(edge_value, digits=2)
        bias = row.get("bias", "?")
        link = f"/forecast/{ticker}?time={time_param}"
        asof_str = asof_et.strftime("%H:%M")
        edge_note = " (|m|-e)" if row.get("implied_eod_move_pct") is not None else ""
        row_text = " | ".join(
            [
                f"<strong>{ticker}</strong> (Conf {conf_pct}%) — as of {asof_str} ET",
                f"Similar days: {n}",
                f"Median close Δ: {median} (IQR {iqr})",
                f"High {high_tail} / Low {low_tail}",
                f"Implied EOD: {implied_move}",
                f"Edge: {edge_str}{edge_note}",
                f"Bias: <strong>{bias}</strong>",
            ]
        )
        parts.append(
            "".join(
                [
                    "<p>",
                    row_text,
                    "<br/>",
                    f"Link: <a href=\"{link}\">{link}</a>",
                    "</p>",
                ]
            )
        )
    parts.append("</body></html>")
    html = "".join(parts)
    return subject, html


def run_and_email(asof: dt.datetime, run_label: str) -> None:
    asof = asof if asof.tzinfo else asof.replace(tzinfo=dt.timezone.utc)
    if not is_trading_day(asof):
        logger.info(
            "forecast_email skip reason=market_closed label=%s asof=%s",
            run_label,
            asof.isoformat(),
        )
        return

    recipients = _env_recipients()
    logger.info(
        "forecast_email start label=%s asof=%s recipients=%d",
        run_label,
        asof.isoformat(),
        len(recipients),
    )
    top5 = select_forecast_top5(asof)
    if not top5:
        logger.info(
            "forecast_email skip reason=no_candidates label=%s asof=%s",
            run_label,
            asof.isoformat(),
        )
        return

    logger.info(
        "forecast_email top5 label=%s tickers=%s",
        run_label,
        [row.get("ticker") for row in top5],
    )
    if not recipients:
        logger.warning(
            "forecast_email skip reason=no_recipients label=%s", run_label
        )
        return
    subj, html = render_email(asof, top5, run_label)
    context = {
        "job": "forecast_email",
        "run_label": run_label,
        "asof": asof.isoformat(),
        "tickers": [row.get("ticker") for row in top5],
    }
    result = send_email(_env_sender(), recipients, subj, html, context=context)
    if result.get("ok"):
        logger.info(
            "forecast_email sent label=%s recipients=%d tickers=%s",
            run_label,
            len(recipients),
            [row.get("ticker") for row in top5],
        )
    else:
        logger.warning(
            "forecast_email send_failed label=%s error=%s",
            run_label,
            result.get("error"),
        )


@router.post("/jobs/forecast-email")
def trigger_forecast_email(label: str = Query("adhoc")) -> dict[str, object]:
    now = now_et()
    run_and_email(now, label)
    return {"ok": True, "label": label, "asof": now.isoformat()}


def schedule_jobs(scheduler, now_fn: Callable[[], dt.datetime]) -> None:
    """Register cron triggers for the APScheduler instance."""

    def _run_1000() -> None:
        run_and_email(now_fn(), "10:00 ET")

    def _run_1400() -> None:
        run_and_email(now_fn(), "14:00 ET")

    scheduler.add_job(
        _run_1000,
        "cron",
        day_of_week="mon-fri",
        hour=10,
        minute=0,
        timezone="America/New_York",
        id="forecast_email_1000",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_1400,
        "cron",
        day_of_week="mon-fri",
        hour=14,
        minute=0,
        timezone="America/New_York",
        id="forecast_email_1400",
        replace_existing=True,
    )
    logger.info("forecast_email scheduler_registered")


__all__ = ["router", "render_email", "run_and_email", "schedule_jobs"]

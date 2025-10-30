"""Scheduled job and API endpoint for the intraday forecast email."""
from __future__ import annotations

import datetime as dt
import logging
import math
import smtplib
import time
from typing import Callable, Iterable, Mapping, Sequence

from fastapi import APIRouter, Query

from services.emailer import load_smtp_settings
from services.forecast_selector import (
    is_trading_day,
    select_forecast_top5,
)
from services.notify import send_email_smtp
from utils import TZ, now_et

logger = logging.getLogger(__name__)

router = APIRouter()


def _iter_candidates(raw: object) -> Sequence[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.split(",")
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    return [str(raw)]


def _normalize_recipients(raw: object) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for entry in _iter_candidates(raw):
        addr = str(entry or "").strip()
        if not addr:
            continue
        key = addr.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(addr)
    return result


def _settings_recipients(settings: Mapping[str, object]) -> list[str]:
    recipients = _normalize_recipients(settings.get("forecast_recipients"))
    if recipients:
        return recipients
    fallback = _normalize_recipients(settings.get("scanner_recipients"))
    return fallback


def _settings_sender(settings: Mapping[str, object]) -> str:
    mail_from = str(settings.get("mail_from") or "").strip()
    if mail_from:
        return mail_from
    smtp_user = str(settings.get("smtp_user") or "").strip()
    return smtp_user


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


def _format_decimal(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(num) or math.isinf(num):
        return "n/a"
    return f"{num:.{digits}f}"


def render_email(asof: dt.datetime, top5: list[dict], run_label: str) -> tuple[str, str]:
    """Return the email subject and HTML body for the forecast run."""

    asof = asof if asof.tzinfo else asof.replace(tzinfo=dt.timezone.utc)
    asof_et = asof.astimezone(TZ)
    subject = f"Today’s Forecast — {asof_et.strftime('%H:%M')} ET — Top 5 Opportunities"
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
        final_score = _format_decimal(row.get("final_score"), digits=4)
        breakdown = row.get("similarity_breakdown") or {}
        weights = breakdown.get("weights") if isinstance(breakdown, Mapping) else {}
        if isinstance(weights, Mapping):
            weight_str = ", ".join(
                f"{key}:{_format_decimal(val, digits=2)}" for key, val in weights.items()
            ) or "n/a"
        else:
            weight_str = "n/a"
        sim_line = " | ".join(
            [
                f"S5m={_format_decimal(breakdown.get('S5m'), digits=4)}",
                f"S30m={_format_decimal(breakdown.get('S30m'), digits=4)}",
                f"S1d={_format_decimal(breakdown.get('S1d'), digits=4)}",
                f"Weights: {weight_str}",
                f"Final={final_score}",
            ]
        )
        raw_options_hint = row.get("options_hint")
        options_hint = raw_options_hint if isinstance(raw_options_hint, Mapping) else {}
        opt_bias = str(options_hint.get("bias") or "n/a")
        exp_move_raw = options_hint.get("exp_move_pct")
        exp_move_display = (
            _pct_with_unit(exp_move_raw * 100.0 if isinstance(exp_move_raw, (int, float)) else None, digits=2)
            if isinstance(exp_move_raw, (int, float))
            else "n/a"
        )
        suggested_delta = _format_decimal(options_hint.get("suggested_delta"), digits=2)
        suggested_expiry = options_hint.get("suggested_expiry") or "n/a"
        options_line = " | ".join(
            [
                f"Bias: {opt_bias}",
                f"Exp move: {exp_move_display}",
                f"Delta: {suggested_delta}",
                f"Expiry: {suggested_expiry}",
            ]
        )
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
                    f"Similarity: {sim_line}",
                    "<br/>",
                    f"Options hint: {options_line}",
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

    settings = load_smtp_settings()
    host = str(settings.get("smtp_host") or "").strip()
    port_raw = settings.get("smtp_port")
    try:
        port = int(port_raw) if port_raw is not None else 0
    except (TypeError, ValueError):
        port = 0
    user = str(settings.get("smtp_user") or "").strip()
    password = str(settings.get("smtp_pass") or "")
    sender = _settings_sender(settings)
    recipients = _settings_recipients(settings)

    if not recipients:
        logger.warning(
            "forecast_email skip reason=no_recipients label=%s host=%s port=%s",
            run_label,
            host or "?",
            port,
        )
        return

    if not host or port <= 0:
        logger.warning(
            "forecast_email skip reason=smtp_not_configured label=%s host=%s port=%s",
            run_label,
            host or "?",
            port,
        )
        return

    if not sender:
        logger.warning(
            "forecast_email skip reason=no_sender label=%s host=%s port=%s",
            run_label,
            host or "?",
            port,
        )
        return

    start_clock = time.monotonic()
    logger.info(
        "forecast_email start label=%s asof=%s host=%s port=%s recipients=%d",
        run_label,
        asof.isoformat(),
        host,
        port,
        len(recipients),
    )

    top5, stats = select_forecast_top5(asof, include_metadata=True)
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
    subj, html = render_email(asof, top5, run_label)
    context = {
        "job": "forecast_email",
        "run_label": run_label,
        "asof": asof.isoformat(),
        "tickers": [row.get("ticker") for row in top5],
    }
    send_ok = False
    message_id = None
    last_error = None
    attempts = 0
    max_attempts = 2
    while attempts < max_attempts:
        attempts += 1
        try:
            result = send_email_smtp(
                host,
                port,
                user,
                password,
                sender,
                recipients,
                subj,
                html,
                context=context,
                raise_exceptions=True,
            )
        except smtplib.SMTPException as exc:
            last_error = str(exc)
            logger.warning(
                "forecast_email smtp_error label=%s attempt=%d error=%s",
                run_label,
                attempts,
                exc,
            )
            if attempts < max_attempts:
                time.sleep(2.0)
                continue
            result = {"ok": False, "error": last_error}
        if result.get("ok"):
            send_ok = True
            message_id = result.get("message_id")
            break
        last_error = str(result.get("error"))
        logger.warning(
            "forecast_email send_failed label=%s attempt=%d error=%s",
            run_label,
            attempts,
            last_error,
        )
        break

    end_clock = time.monotonic()
    duration = end_clock - start_clock
    logger.info(
        "forecast_email complete label=%s asof=%s status=%s host=%s port=%s recipients=%d processed=%s selected=%s duration=%.2fs message_id=%s",
        run_label,
        asof.isoformat(),
        "sent" if send_ok else "failed",
        host,
        port,
        len(recipients),
        stats.get("processed"),
        stats.get("selected"),
        duration,
        message_id,
    )

    if not send_ok and last_error:
        logger.warning(
            "forecast_email final_failure label=%s error=%s",
            run_label,
            last_error,
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

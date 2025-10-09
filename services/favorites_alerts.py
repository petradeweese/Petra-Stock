"""Favorites alert helpers for contract selection and evaluation.

The production system enriches favorites scan hits with option contract
information and sends multi-line MMS alerts.  For the unit tests in this
repository we provide a greatly simplified but fully functional subset of the
logic described in the specification.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import settings

from . import events_provider, options_provider, sms_consent, twilio_client
from .notify import send_email_smtp
from .telemetry import log as log_telemetry


logger = logging.getLogger(__name__)

_SENT_ALERTS: Dict[str, float] = {}
_SENT_LOCK = threading.Lock()
_REDIS_CLIENT: Any | None = None
_REDIS_READY: Optional[bool] = None
_SQLITE_CONN: sqlite3.Connection | None = None
_SQLITE_PATH = Path(__file__).resolve().parent.parent / "alerts.sqlite"
_REDIS_TTL_SECONDS = 2 * 60 * 60

def _outcomes_mode() -> str:
    raw = getattr(settings, "alert_outcomes", getattr(settings, "ALERT_OUTCOMES", "hit"))
    value = str(raw or "hit").strip().lower()
    return value or "hit"


def _startup_delivery_safety() -> None:
    try:
        configured_channel = _normalize_channel(getattr(settings, "alert_channel", "Email"))
        if configured_channel == "email":
            email_targets = getattr(settings, "alert_email_to", ()) or ()
            if not email_targets:
                logger.warning("Favorites alerts skipped: no recipients for Email.")
        elif configured_channel in {"mms", "sms"}:
            sms_targets = getattr(settings, "alert_sms_to", ()) or ()
            if not sms_targets:
                logger.warning("Favorites alerts skipped: no recipients for MMS.")
            if configured_channel == "mms" and not twilio_client.is_enabled():
                logger.warning("Favorites alerts fallback to Email: Twilio not configured.")
                setattr(settings, "alert_channel", "Email")
                setattr(settings, "ALERT_CHANNEL", "Email")
    except Exception:  # pragma: no cover - safety guard
        logger.debug("favorites alert startup delivery check failed", exc_info=True)


_startup_delivery_safety()


def _get_redis_client():
    global _REDIS_CLIENT, _REDIS_READY
    if _REDIS_READY is not None:
        return _REDIS_CLIENT if _REDIS_READY else None
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        _REDIS_READY = False
        _REDIS_CLIENT = None
        return None
    try:  # pragma: no cover - optional dependency
        import redis  # type: ignore

        client = redis.Redis.from_url(url)
        _REDIS_CLIENT = client
        _REDIS_READY = True
        return client
    except Exception:  # pragma: no cover - redis optional
        logger.exception("favorites alert redis initialization failed")
        _REDIS_CLIENT = None
        _REDIS_READY = False
        return None


def _ensure_sqlite() -> sqlite3.Connection:
    global _SQLITE_CONN
    if _SQLITE_CONN is not None:
        return _SQLITE_CONN
    conn = sqlite3.connect(_SQLITE_PATH, timeout=5, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sent_alerts (
            dedupe_key TEXT PRIMARY KEY,
            ts INTEGER,
            simulated INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    _SQLITE_CONN = conn
    return conn


def _now() -> float:
    return time.time()


def _prune_in_memory(now_ts: Optional[float] = None) -> None:
    if not _SENT_ALERTS:
        return
    current = now_ts if now_ts is not None else _now()
    expired = [
        key
        for key, expires_at in list(_SENT_ALERTS.items())
        if expires_at <= current
    ]
    for key in expired:
        _SENT_ALERTS.pop(key, None)


def _dedupe_storage_get(key: str) -> bool:
    if not key:
        return False
    client = _get_redis_client()
    if client is not None:
        try:
            return bool(client.exists(key))
        except Exception:  # pragma: no cover - network interaction
            logger.debug("favorites alert redis exists failed key=%s", key)
    try:
        conn = _ensure_sqlite()
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, simulated FROM sent_alerts WHERE dedupe_key=? LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return False
        ts_raw = row[0]
        try:
            ts_value = float(ts_raw)
        except (TypeError, ValueError):
            ts_value = 0.0
        if ts_value and _now() - ts_value > _REDIS_TTL_SECONDS:
            conn.execute("DELETE FROM sent_alerts WHERE dedupe_key=?", (key,))
            conn.commit()
            return False
        return True
    except sqlite3.Error:
        logger.exception("favorites alert sqlite dedupe check failed")
        return False


def _dedupe_storage_set(key: str, *, simulated: bool = False) -> None:
    if not key:
        return
    client = _get_redis_client()
    if client is not None:
        try:
            payload = json.dumps({"ts": int(_now()), "simulated": bool(simulated)})
            client.setex(key, _REDIS_TTL_SECONDS, payload)
        except Exception:  # pragma: no cover - network interaction
            logger.debug("favorites alert redis set failed key=%s", key)
    try:
        conn = _ensure_sqlite()
        conn.execute(
            """
            INSERT OR REPLACE INTO sent_alerts(dedupe_key, ts, simulated)
            VALUES(?, ?, ?)
            """,
            (key, int(_now()), 1 if simulated else 0),
        )
        conn.commit()
    except sqlite3.Error:
        logger.exception("favorites alert sqlite dedupe set failed")


def _coerce_bar_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)) and not math.isnan(float(value)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _dedupe_key(fav_id: Any, interval: Any, bar_dt_utc: Any) -> Optional[str]:
    if fav_id in (None, "", b""):
        return None
    dt = _coerce_bar_datetime(bar_dt_utc)
    if dt is None:
        return None
    interval_str = str(interval or "").strip().lower() or "unknown"
    fav_str = str(fav_id)
    return f"{fav_str}|{interval_str}|{dt.strftime('%Y-%m-%dT%H:%M')}"


def was_sent(
    favorite_id: Any,
    bar_time: Any,
    *,
    interval: Any = None,
    dedupe_key: Optional[str] = None,
) -> bool:
    """Return ``True`` if the alert for ``favorite_id``/``bar_time`` was sent."""

    return was_sent_key(favorite_id, bar_time, interval=interval, dedupe_key=dedupe_key)


def mark_sent(
    favorite_id: Any,
    bar_time: Any,
    *,
    interval: Any = None,
    dedupe_key: Optional[str] = None,
    simulated: bool = False,
) -> None:
    """Record that an alert was delivered for the given favorite/bar."""

    mark_sent_key(
        favorite_id,
        bar_time,
        interval=interval,
        dedupe_key=dedupe_key,
        simulated=simulated,
    )


def was_sent_key(
    favorite_id: Any,
    bar_time: Any,
    *,
    interval: Any = None,
    dedupe_key: Optional[str] = None,
) -> bool:
    key = dedupe_key or _dedupe_key(favorite_id, interval, bar_time)
    if not key:
        return False
    with _SENT_LOCK:
        _prune_in_memory()
        if key in _SENT_ALERTS:
            return True
    if _dedupe_storage_get(key):
        with _SENT_LOCK:
            _SENT_ALERTS[key] = _now() + _REDIS_TTL_SECONDS
        return True
    return False


def mark_sent_key(
    favorite_id: Any,
    bar_time: Any,
    *,
    interval: Any = None,
    dedupe_key: Optional[str] = None,
    simulated: bool = False,
) -> None:
    key = dedupe_key or _dedupe_key(favorite_id, interval, bar_time)
    if not key:
        return
    with _SENT_LOCK:
        _prune_in_memory()
        _SENT_ALERTS[key] = _now() + _REDIS_TTL_SECONDS
    _dedupe_storage_set(key, simulated=simulated)


@dataclass
class FavoriteHitStub:
    ticker: str
    direction: str  # "UP" or "DOWN"
    pattern: str
    target_pct: float = 0.0
    stop_pct: float = 0.0
    hit_pct: float = 0.0
    avg_roi_pct: float = 0.0
    avg_dd_pct: float = 0.0
    favorite_id: Optional[str] = None
    bar_time: Optional[str] = None


@dataclass
class Check:
    name: str
    symbol: str
    value: float
    passed: bool
    explanation: str | None = None


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    result.append(item)
        return result
    return []


def _extract_favorite_id(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, FavoriteHitStub):
        return obj.favorite_id
    if isinstance(obj, Mapping):
        for key in ("favorite_id", "id"):
            value = obj.get(key)
            if value not in (None, ""):
                return str(value)
    for attr in ("favorite_id", "id"):
        value = getattr(obj, attr, None)
        if value not in (None, ""):
            return str(value)
    return None


def _extract_bar_time(row: Any) -> Optional[str]:
    if row is None:
        return None
    if isinstance(row, Mapping):
        for key in ("bar_time", "bar_ts", "timestamp", "bar_timestamp"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    for attr in ("bar_time", "bar_ts", "timestamp", "bar_timestamp"):
        value = getattr(row, attr, None)
        if value not in (None, ""):
            return str(value)
    return None


def _normalize_channel(channel: Optional[str]) -> Optional[str]:
    if channel is None:
        return None
    value = str(channel).strip().lower()
    if not value:
        return None
    if value in {"email", "mms", "sms"}:
        return value
    return value


def _should_skip_non_entry(row: Any) -> bool:
    if row is None:
        return False
    if isinstance(row, Mapping):
        values = [row.get(key) for key in ("event", "signal", "signal_type", "status", "reason")]
    else:
        values = [getattr(row, key, None) for key in ("event", "signal", "signal_type", "status", "reason")]
    for value in values:
        if not value or not isinstance(value, str):
            continue
        lowered = value.lower()
        if "stop" in lowered or "timeout" in lowered:
            return True
    return False


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _format_threshold_ratio(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "?"
    if math.isnan(num):
        return "?"
    scaled = num
    if abs(scaled) >= 10:
        scaled = scaled / 100.0
    return f"{scaled:.2f}"


def _log_skip(reason: str, detail: Optional[str] = None) -> None:
    logger.info(
        "favorites_alert_skip",
        extra={"reason": reason, "detail": detail},
    )


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            if not value.strip():
                return default
            return float(value)
        return float(value)
    except Exception:
        return default


def _favorite_snapshot_numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        return number

    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8")
        except Exception:
            return None
        return _favorite_snapshot_numeric(decoded)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return None
            return _favorite_snapshot_numeric(parsed)

    if isinstance(value, Mapping):
        keys = (
            "avg_roi_pct",
            "avg_roi",
            "roi_pct",
            "roi",
            "value",
            "pct",
            "percent",
            "percentage",
        )
        for key in keys:
            if key in value:
                parsed = _favorite_snapshot_numeric(value.get(key))
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = _favorite_snapshot_numeric(nested)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            parsed = _favorite_snapshot_numeric(item)
            if parsed is not None:
                return parsed

    return None


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            if not value.strip():
                return default
            return int(float(value))
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:
        return default


def should_alert_on_row(row: Any, fav: Any) -> bool:
    """Return ``True`` when ``row`` represents an entry signal for ``fav``."""

    if row is None or fav is None:
        return False
    include_all = _outcomes_mode() == "all"
    if not include_all and _should_skip_non_entry(row):
        _log_skip("skip_non_entry")
        return False

    event_values: List[str] = []
    for key in ("event", "signal", "signal_type", "status", "reason"):
        value = _value(row, key)
        if not value or not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if not lowered:
            continue
        if not include_all and ("stop" in lowered or "timeout" in lowered):
            return False
        event_values.append(lowered)

    if event_values and not include_all:
        entry_keywords = ("entry", "detect", "pattern", "trigger", "alert")
        if not any(any(word in ev for word in entry_keywords) for ev in event_values):
            return False

    support = _coerce_int(_value(row, "support"))
    min_support = _coerce_int(_value(fav, "min_support"))
    if min_support and support and support < min_support:
        _log_skip("skip_min_support", f"({int(support)}<{int(min_support)})")
        return False

    hit_pct = _coerce_float(_value(row, "hit_pct"))
    if hit_pct <= 0:
        hit_rate = _coerce_float(_value(row, "hit_rate"))
        if 0 <= hit_rate <= 1.0:
            hit_pct = hit_rate * 100.0
        else:
            hit_pct = hit_rate

    hit_threshold = _coerce_float(_value(fav, "scan_min_hit"), 50.0)
    if hit_threshold <= 0:
        hit_threshold = 50.0
    if hit_pct < hit_threshold:
        _log_skip(
            "skip_min_hit",
            f"({_format_threshold_ratio(hit_pct)}<{_format_threshold_ratio(hit_threshold)})",
        )
        return False

    avg_roi = _coerce_float(_value(row, "avg_roi_pct"))
    if avg_roi == 0.0:
        avg_roi = _coerce_float(_value(row, "avg_roi"))
        if abs(avg_roi) <= 1.0:
            avg_roi *= 100.0

    min_roi = _coerce_float(_value(fav, "min_avg_roi_pct"), 0.0)
    if avg_roi <= max(0.0, min_roi):
        _log_skip(
            "skip_min_avg_roi",
            f"({_format_threshold_ratio(avg_roi)}<{_format_threshold_ratio(max(0.0, min_roi))})",
        )
        return False

    return True


def _merge_smtp_config(fav: Any, row: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    sources: Iterable[Any] = ()
    if isinstance(fav, Mapping):
        sources = (fav,)
    elif fav is not None:
        sources = (fav.__dict__,)
    for source in sources:
        for key, dest in (
            ("smtp_host", "host"),
            ("host", "host"),
            ("smtp_port", "port"),
            ("port", "port"),
            ("smtp_user", "user"),
            ("user", "user"),
            ("smtp_pass", "password"),
            ("password", "password"),
            ("mail_from", "mail_from"),
            ("from", "mail_from"),
        ):
            value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
            if value not in (None, ""):
                config.setdefault(dest, value)
    if isinstance(row, Mapping):
        for key in ("smtp", "smtp_config"):
            embedded = row.get(key)
            if isinstance(embedded, Mapping):
                for sub_key, value in embedded.items():
                    if value not in (None, ""):
                        config.setdefault(sub_key, value)
    return config


def _coalesce(values: Iterable[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _deliver_alert(
    channel: Optional[str],
    subject: str,
    body: str,
    *,
    recipients: Optional[Sequence[str]] = None,
    favorite_id: Optional[str] = None,
    bar_time: Optional[str] = None,
    interval: Any = None,
    dedupe_key: Optional[str] = None,
    smtp_config: Optional[Mapping[str, Any]] = None,
    delivery_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Optional[str], Optional[str], str]:
    normalized = _normalize_channel(channel)
    if normalized is None:
        logger.info(
            "favorites_alert_delivery_result",
            extra={
                "channel": None,
                "dedupe_key": dedupe_key,
                "reason": "delivery_error",
                "ok": False,
                "fav_id": favorite_id,
                "bar_time": bar_time,
                "interval": interval,
            },
        )
        return False, None, None, "delivery_error"

    context_base: Dict[str, Any] = {}
    if delivery_context:
        context_base.update({k: v for k, v in delivery_context.items() if v is not None})
    if favorite_id is not None and "fav_id" not in context_base:
        context_base["fav_id"] = str(favorite_id)

    dedupe_value = dedupe_key or _dedupe_key(favorite_id, interval, bar_time)
    if dedupe_value and was_sent_key(
        favorite_id,
        bar_time,
        interval=interval,
        dedupe_key=dedupe_value,
    ):
        _log_skip("skip_dedupe", dedupe_value)
        logger.info(
            "favorites_alert_delivery_result",
            extra={
                "channel": normalized,
                "dedupe_key": dedupe_value,
                "reason": "throttled",
                "ok": False,
                "fav_id": favorite_id,
                "bar_time": bar_time,
                "interval": interval,
            },
        )
        return False, dedupe_value, normalized, "throttled"

    if normalized == "email":
        send_to = [r for r in (recipients or []) if r]
        if not send_to:
            logger.info("favorites alert email skipped: no recipients configured")
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "missing_recipient",
                    "ok": False,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return False, dedupe_value, normalized, "missing_recipient"
        cfg = dict(smtp_config or {})
        host = cfg.get("host")
        port = cfg.get("port")
        user = cfg.get("user", "")
        password = cfg.get("password", "")
        mail_from = cfg.get("mail_from") or user
        if not host or not port or not mail_from:
            logger.warning("favorites alert email skipped: incomplete smtp config")
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "delivery_error",
                    "ok": False,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return False, dedupe_value, normalized, "delivery_error"
        try:
            result = send_email_smtp(
                str(host),
                int(port),
                str(user or ""),
                str(password or ""),
                str(mail_from),
                [str(r) for r in send_to],
                subject,
                body,
                context={**context_base, "channel": "email"},
            )
        except Exception:
            logger.exception("favorites alert email send raised")
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "delivery_error",
                    "ok": False,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return False, dedupe_value, normalized, "delivery_error"
        if result.get("ok"):
            logger.info(
                "favorites alert email sent recipients=%d", len(send_to)
            )
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "sent",
                    "ok": True,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return True, dedupe_value, normalized, "sent"
        logger.warning("favorites alert email failed: %s", result.get("error"))
        logger.info(
            "favorites_alert_delivery_result",
            extra={
                "channel": normalized,
                "dedupe_key": dedupe_value,
                "reason": "delivery_error",
                "ok": False,
                "fav_id": favorite_id,
                "bar_time": bar_time,
                "interval": interval,
            },
        )
        return False, dedupe_value, normalized, "delivery_error"

    if normalized in {"mms", "sms"}:
        if not twilio_client.is_enabled():
            raise RuntimeError("Twilio not configured")

        destinations: list[dict[str, str | None]] = []
        if recipients is not None:
            for entry in recipients:
                if isinstance(entry, Mapping):
                    phone = (
                        entry.get("phone_e164")
                        or entry.get("phone")
                        or entry.get("to")
                    )
                    if phone:
                        destinations.append(
                            {
                                "phone_e164": str(phone),
                                "user_id": entry.get("user_id"),
                            }
                        )
                else:
                    destinations.append({"phone_e164": str(entry)})
        else:
            destinations = sms_consent.active_destinations()

        if not destinations:
            logger.info("favorites alert mms skipped: no consented numbers configured")
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "missing_recipient",
                    "ok": False,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return False, dedupe_value, normalized, "missing_recipient"

        seen: set[str] = set()
        message_body = sms_consent.append_footer(body)
        delivered = False
        for dest in destinations:
            number_raw = dest.get("phone_e164") or dest.get("phone") or dest.get("to")
            number_str = sms_consent.normalize_phone(str(number_raw or ""))
            if not number_str or number_str in seen:
                continue
            seen.add(number_str)
            allowed, consent_row = sms_consent.allow_sending(number_str)
            if not allowed:
                logger.info(
                    "favorites alert mms skipped number=%s reason=no-consent-or-rate",
                    number_str,
                )
                continue
            user_for_log = (consent_row or {}).get("user_id") or dest.get("user_id")
            context = {
                **context_base,
                "channel": normalized,
                "to": number_str,
                "user_id": user_for_log,
            }
            try:
                send_ok = twilio_client.send_mms(number_str, message_body, context=context)
            except Exception as exc:  # pragma: no cover - network interaction
                raise RuntimeError(f"Twilio send error: {exc}") from exc
            if send_ok:
                sms_consent.record_delivery(number_str, user_for_log, message_body)
                delivered = True
            else:
                logger.warning("favorites alert %s failed number=%s", normalized, number_str)

        if delivered:
            logger.info(
                "favorites_alert_delivery_result",
                extra={
                    "channel": normalized,
                    "dedupe_key": dedupe_value,
                    "reason": "sent",
                    "ok": True,
                    "fav_id": favorite_id,
                    "bar_time": bar_time,
                    "interval": interval,
                },
            )
            return True, dedupe_value, normalized, "sent"
        raise RuntimeError("Twilio send failed for all recipients")

    logger.info("favorites alert skipped: unknown channel=%s", normalized)
    logger.info(
        "favorites_alert_delivery_result",
        extra={
            "channel": normalized,
            "dedupe_key": dedupe_value,
            "reason": "delivery_error",
            "ok": False,
            "fav_id": favorite_id,
            "bar_time": bar_time,
            "interval": interval,
        },
    )
    return False, dedupe_value, normalized, "delivery_error"
@dataclass
class SelectionResult:
    contract: Optional[options_provider.OptionContract]
    alternatives: List[options_provider.OptionContract]
    rejects: List[Dict[str, str]]
    note: Optional[str] = None
    event_note: Optional[str] = None


_FAIL_EXPLANATIONS = {
    "delta_high": "Delta too high — Option is too sensitive to stock moves, could swing too much.",
    "delta_low": "Delta too low — Option won’t track the stock closely enough.",
    "gamma_low": "Gamma too low — Delta won’t adjust quickly, limiting responsiveness.",
    "theta_low": "Theta too negative — Contract will lose value too fast each day.",
    "vega_high": "Vega too high — Price depends heavily on volatility, making it unstable.",
    "ivr_high": "IV Rank high — Options are overpriced compared to their history.",
    "spread_high": "Spread too wide — Cost to trade is high, making fills expensive.",
    "oi_low": "Open interest too low — Not enough contracts exist, liquidity is weak.",
    "volume_low": "Volume too low — Too few trades today, fills may be hard.",
    "dte_out": "DTE outside range — Expiration is not in your configured time window.",
}

_PASS_SUMMARIES = {
    "Delta": "in preferred range.",
    "Gamma": "responsive delta change.",
    "Theta": "daily time decay.",
    "Vega": "sensitivity to volatility shifts.",
    "IV Rank": "within comfort zone.",
}


def merge_profiles(global_profile: Dict, override: Optional[Dict]) -> Dict:
    """Merge ``override`` into ``global_profile`` returning a new dict."""

    result = json.loads(json.dumps(global_profile))  # deep copy
    if not override:
        return result
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_profiles(result[k], v)
        else:
            result[k] = v
    return result


def load_profile(settings_json: str, override_json: Optional[str]) -> Dict:
    g = json.loads(settings_json or "{}")
    o = json.loads(override_json) if override_json else {}
    return merge_profiles(g, o)


def select_contract(ticker: str, side: str, profile: Dict) -> SelectionResult:
    """Return the best matching option contract for ``ticker`` and ``side``."""

    chain = options_provider.get_chain(ticker)
    candidates = [c for c in chain if c.side.lower() == side.lower()]
    dte_min = profile.get("dte_min")
    dte_max = profile.get("dte_max")
    candidates = [
        c
        for c in candidates
        if (dte_min is None or dte_min <= c.dte)
        and (dte_max is None or c.dte <= dte_max)
    ]
    target_delta = profile.get("target_delta", 0.0)
    rejects: List[Dict[str, str]] = []
    passes: List[options_provider.OptionContract] = []
    min_oi = profile.get("min_open_interest")
    min_volume = profile.get("min_volume")
    max_spread = profile.get("max_spread_pct")
    for c in candidates:
        reason = None
        if min_oi is not None and c.open_interest < min_oi:
            reason = "open interest too low"
        elif min_volume is not None and c.volume < min_volume:
            reason = "volume too low"
        elif max_spread is not None and c.spread_pct > max_spread:
            reason = "spread too wide"
        if reason:
            rejects.append({"occ": c.occ, "reason": reason})
        else:
            passes.append(c)

    avoid_days = profile.get("avoid_event_days", 0) or 0
    event_note: Optional[str] = None
    if avoid_days and passes:
        evs = events_provider.next_events(ticker)
        non_conflict: List[options_provider.OptionContract] = []
        conflicts: List[tuple[options_provider.OptionContract, Dict[str, str]]] = []
        for c in passes:
            conflict = False
            for ev in evs:
                try:
                    edate = datetime.fromisoformat(ev["date"]).date()
                except Exception:
                    continue
                if abs((c.expiry - edate).days) <= avoid_days:
                    conflict = True
                    conflicts.append((c, ev))
                    break
            if not conflict:
                non_conflict.append(c)
        if non_conflict:
            passes = non_conflict
            if conflicts:
                ev = conflicts[0][1]
                event_note = f"{ev['type']} within {avoid_days}d of expiry (⚠️ avoided)"
        elif conflicts:
            # all conflicted, keep list but set note
            ev = conflicts[0][1]
            event_note = f"{ev['type']} within {avoid_days}d of expiry"

    if passes:
        passes.sort(key=lambda c: (abs(c.delta - target_delta), c.spread_pct, -c.volume, abs(c.delta)))
        return SelectionResult(passes[0], [], rejects, None, event_note)

    candidates.sort(key=lambda c: abs(c.delta - target_delta))
    alternatives = candidates[:2]
    note = "no liquid match; best alternatives shown" if alternatives else None
    return SelectionResult(None, alternatives, rejects, note, event_note)


def evaluate_contract(contract: options_provider.OptionContract, profile: Dict) -> List[Check]:
    checks: List[Check] = []

    def _add(name: str, symbol: str, value: float, passed: bool, reason: str | None):
        checks.append(Check(name, symbol, value, passed, _FAIL_EXPLANATIONS.get(reason) if reason else None))

    delta = contract.delta
    reason = None
    if profile.get("delta_min") is not None and delta < profile["delta_min"]:
        reason = "delta_low"
    if profile.get("delta_max") is not None and delta > profile["delta_max"]:
        reason = "delta_high"
    _add("Delta", "Δ", delta, reason is None, reason)

    gamma = contract.gamma
    reason = None
    if profile.get("gamma_min") is not None and gamma < profile["gamma_min"]:
        reason = "gamma_low"
    if profile.get("gamma_max") is not None and gamma > profile["gamma_max"]:
        reason = "gamma_high"
    _add("Gamma", "Γ", gamma, reason is None, reason)

    theta = contract.theta
    reason = None
    if profile.get("theta_min") is not None and theta < profile["theta_min"]:
        reason = "theta_low"
    if profile.get("theta_max") is not None and theta > profile["theta_max"]:
        reason = "theta_high"
    _add("Theta", "Θ", theta, reason is None, reason)

    vega = contract.vega
    reason = None
    if profile.get("vega_min") is not None and vega < profile["vega_min"]:
        reason = "vega_low"
    if profile.get("vega_max") is not None and vega > profile["vega_max"]:
        reason = "vega_high"
    _add("Vega", "ν", vega, reason is None, reason)

    ivr = contract.iv_rank
    reason = None
    if profile.get("iv_rank_min") is not None and ivr < profile["iv_rank_min"]:
        reason = "ivr_low"
    if profile.get("iv_rank_max") is not None and ivr > profile["iv_rank_max"]:
        reason = "ivr_high"
    _add("IV Rank", "IVR", ivr, reason is None, reason)

    oi = contract.open_interest
    reason = None
    if profile.get("min_open_interest") is not None and oi < profile["min_open_interest"]:
        reason = "oi_low"
    _add("Open Interest", "OI", oi, reason is None, reason)

    vol = contract.volume
    reason = None
    if profile.get("min_volume") is not None and vol < profile["min_volume"]:
        reason = "volume_low"
    _add("Volume", "Vol", vol, reason is None, reason)

    spread = contract.spread_pct
    reason = None
    if profile.get("max_spread_pct") is not None and spread > profile["max_spread_pct"]:
        reason = "spread_high"
    _add("Spread %", "Spread", spread, reason is None, reason)

    dte = contract.dte
    reason = None
    if profile.get("dte_min") is not None and dte < profile["dte_min"]:
        reason = "dte_out"
    if profile.get("dte_max") is not None and dte > profile["dte_max"]:
        reason = "dte_out"
    _add("DTE", "DTE", dte, reason is None, reason)

    return checks


def _format_number(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            if abs(value) >= 100 or math.isclose(value, round(value)):
                return f"{value:.0f}"
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)
    return str(value)


def _split_explanation(text: str) -> tuple[str, str]:
    if "—" in text:
        head, tail = text.split("—", 1)
        return head.strip(), tail.strip()
    return text.strip(), ""


def _format_feedback_reason(chk: Check) -> str:
    explanation = chk.explanation or f"{chk.name} needs review"
    reason, detail = _split_explanation(explanation)
    if detail:
        return f"{reason} → {detail}"
    return reason


def _format_greek_line(chk: Check, *, compact: bool) -> str | None:
    if compact and chk.passed:
        return None
    value = _format_number(chk.value)
    status = "✅" if chk.passed else "❌"
    explanation = chk.explanation or ""
    if chk.passed:
        explanation = chk.explanation or _PASS_SUMMARIES.get(
            chk.name, "within preferred range."
        )
    else:
        if explanation:
            reason, detail = _split_explanation(explanation)
            short_reason = reason
            if chk.name.lower() in reason.lower():
                short_reason = reason[len(chk.name) :].strip()
            short_reason = short_reason.lstrip("-—:").strip()
            if detail:
                if short_reason:
                    explanation = f"{short_reason}; {detail}"
                else:
                    explanation = detail
        if not explanation:
            explanation = f"{chk.name} outside preferred range."
    line = f"• {chk.name} ({value}) {status}"
    if explanation:
        line += f" — {explanation}"
    return line


def _format_targets_line(targets: Dict | None) -> str:
    targets = targets or {}
    target = _format_number(targets.get("target"))
    stop = _format_number(targets.get("stop"))
    hit = _format_number(targets.get("hit"))
    roi = _format_number(targets.get("roi"))
    dd = _format_number(targets.get("dd"))
    return f"Targets: {target} | Stop: {stop} | Hit% {hit} | ROI {roi} | DD {dd}"


def _format_percent_value(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(num):
        return "—"
    if abs(num) <= 1:
        num *= 100.0
    return f"{num:.2f}%"


def _format_picked(contract: options_provider.OptionContract | None) -> str:
    if not contract:
        return "No contract selected"
    side = contract.side.title() if contract.side else "Contract"
    expiry = contract.expiry.isoformat() if hasattr(contract.expiry, "isoformat") else str(contract.expiry)
    price = contract.mid or contract.last or contract.bid or contract.ask or 0.0
    price_str = _format_number(price)
    return f"{contract.occ} {side}, Exp {expiry} @ ${price_str}"


def _format_email_alert(
    symbol: str,
    direction: str,
    contract: options_provider.OptionContract | None,
    checks: List[Check],
    *,
    compact: bool,
    include_symbols: bool,
    pattern: str | None,
) -> str:
    header_parts = [symbol.upper(), direction.upper()]
    if pattern:
        header_parts.append(pattern)
    lines = [" ".join(part for part in header_parts if part)]
    if contract:
        lines.append(f"Contract {contract.occ}")
    for chk in checks:
        if compact and chk.passed:
            continue
        name = chk.name
        if include_symbols and chk.symbol:
            name = f"{name} ({chk.symbol})"
        status = "✅" if chk.passed else "❌"
        value = _format_number(chk.value)
        line = f"{name}: {value} {status}"
        if chk.explanation and not chk.passed:
            line += f" — {chk.explanation}"
        lines.append(line)
    if contract:
        spread = f"{contract.spread_pct:.1f}%" if contract.spread_pct is not None else "—"
        lines.append(
            "Why this contract: "
            f"Δ {contract.delta:.2f} | DTE {contract.dte} | spread {spread} | IVR {contract.iv_rank}"
        )
    return "\n".join(line for line in lines if line).strip()


def _format_mms_alert(
    symbol: str,
    direction: str,
    contract: options_provider.OptionContract | None,
    checks: List[Check],
    targets: Dict | None,
    *,
    compact: bool,
    pattern: str | None,
) -> str:
    header = f"{symbol.upper()} {direction.upper()}"
    header_parts = [header]
    if contract:
        header_parts.append(f"Picked: {_format_picked(contract)}")
    header_line = " | ".join(part for part in header_parts if part)
    sections: List[List[str]] = [[header_line]] if header_line else []
    if sections and pattern:
        sections[0].append(f"Setup: {pattern}")
    elif pattern:
        sections.append([f"Setup: {pattern}"])

    greeks_lines: List[str] = []
    for chk in checks:
        line = _format_greek_line(chk, compact=compact)
        if line:
            greeks_lines.append(line)
    if greeks_lines:
        sections.append(["Greeks & IV:", *greeks_lines])
    else:
        sections.append(["Greeks/IV not available (fallback data)."])

    failing = [chk for chk in checks if not chk.passed]
    feedback_lines: List[str] = []
    if failing:
        feedback_lines.append("Feedback:")
        for chk in failing:
            feedback_lines.append(f"• {_format_feedback_reason(chk)}")
    elif not compact:
        feedback_lines.append("Feedback:")
        feedback_lines.append("• All checks passing ✅")
    if feedback_lines:
        sections.append(feedback_lines)

    sections.append([_format_targets_line(targets)])
    return "\n\n".join("\n".join(block) for block in sections if block)


def _format_sms_alert(
    symbol: str,
    direction: str,
    targets: Dict | None,
    *,
    pattern: str | None = None,
) -> str:
    targets = targets or {}
    parts: List[str] = [f"{symbol.upper()} {direction.upper()}"]
    metrics: List[str] = []

    hit_text = _format_percent_value(targets.get("hit"))
    if hit_text != "—":
        metrics.append(f"Hit {hit_text}")
    roi_text = _format_percent_value(targets.get("roi"))
    if roi_text != "—":
        metrics.append(f"ROI {roi_text}")
    dd_text = _format_percent_value(targets.get("dd"))
    if dd_text != "—":
        metrics.append(f"DD {dd_text}")

    target_text = _format_percent_value(targets.get("target"))
    stop_text = _format_percent_value(targets.get("stop"))
    if target_text != "—" or stop_text != "—":
        span = "/".join(
            value for value in (target_text, stop_text) if value != "—"
        )
        if span:
            metrics.append(f"Targets {span}")

    if metrics:
        parts.append(" · ".join(metrics))
    if pattern:
        parts.append(f"{pattern}")
    return " ".join(part for part in parts if part).strip()


def format_favorites_alert(
    symbol: str,
    direction: str,
    picked_contract: options_provider.OptionContract | None,
    checks: List[Check],
    targets: Dict | None,
    *,
    compact: bool,
    channel: str,
    pattern: str | None = None,
    include_symbols: bool = True,
) -> str:
    channel = (channel or "mms").lower()
    if channel == "email":
        return _format_email_alert(
            symbol,
            direction,
            picked_contract,
            checks,
            compact=compact,
            include_symbols=include_symbols,
            pattern=pattern,
        )
    if channel == "sms":
        return _format_sms_alert(
            symbol,
            direction,
            targets,
            pattern=pattern,
        )
    return _format_mms_alert(
        symbol,
        direction,
        picked_contract,
        checks,
        targets,
        compact=compact,
        pattern=pattern,
    )


def enrich_and_send(
    fav: FavoriteHitStub | Mapping[str, Any],
    row: Optional[Mapping[str, Any]] = None,
    channel: Optional[str] = None,
    *,
    recipients: Optional[Sequence[str]] = None,
    smtp_config: Optional[Mapping[str, Any]] = None,
    is_test: bool = False,
    simulated: bool = False,
) -> Dict[str, Any]:  # pragma: no cover - orchestrator
    """Enrich a favorites hit, format the alert and optionally deliver it."""

    try:
        if isinstance(fav, FavoriteHitStub):
            hit = fav
        else:
            ticker = _coalesce(
                [
                    (row or {}).get("ticker") if isinstance(row, Mapping) else None,
                    fav.get("ticker") if isinstance(fav, Mapping) else None,
                ]
            ) or "?"
            direction = _coalesce(
                [
                    (row or {}).get("direction") if isinstance(row, Mapping) else None,
                    fav.get("direction") if isinstance(fav, Mapping) else None,
                ]
            ) or "UP"
            pattern = _coalesce(
                [
                    (row or {}).get("pattern") if isinstance(row, Mapping) else None,
                    fav.get("rule") if isinstance(fav, Mapping) else None,
                ]
            ) or ""
            target_pct = _coalesce(
                [
                    (row or {}).get("target_pct") if isinstance(row, Mapping) else None,
                    fav.get("target_pct") if isinstance(fav, Mapping) else None,
                ]
            )
            stop_pct = _coalesce(
                [
                    (row or {}).get("stop_pct") if isinstance(row, Mapping) else None,
                    fav.get("stop_pct") if isinstance(fav, Mapping) else None,
                ]
            )
            hit_pct = _coalesce(
                [
                    (row or {}).get("hit_pct") if isinstance(row, Mapping) else None,
                    fav.get("hit_pct_snapshot") if isinstance(fav, Mapping) else None,
                ]
            ) or 0.0
            avg_roi_raw = _coalesce(
                [
                    (row or {}).get("avg_roi_pct") if isinstance(row, Mapping) else None,
                    fav.get("roi_snapshot") if isinstance(fav, Mapping) else None,
                ]
            )
            avg_roi = _favorite_snapshot_numeric(avg_roi_raw) or 0.0
            avg_dd_raw = _coalesce(
                [
                    (row or {}).get("avg_dd_pct") if isinstance(row, Mapping) else None,
                    fav.get("dd_pct_snapshot") if isinstance(fav, Mapping) else None,
                ]
            )
            avg_dd = _favorite_snapshot_numeric(avg_dd_raw) or 0.0
            hit = FavoriteHitStub(
                ticker=str(ticker),
                direction=str(direction),
                pattern=str(pattern),
                target_pct=float(target_pct) if target_pct is not None else 0.0,
                stop_pct=float(stop_pct) if stop_pct is not None else 0.0,
                hit_pct=float(hit_pct),
                avg_roi_pct=float(avg_roi),
                avg_dd_pct=float(avg_dd),
                favorite_id=_extract_favorite_id(fav),
                bar_time=_extract_bar_time(row),
            )
            if isinstance(fav, Mapping):
                for key in ("greeks_profile_json", "greeks_override_json"):
                    if fav.get(key) is not None:
                        setattr(hit, key, fav.get(key))

        favorite_id = _extract_favorite_id(fav) or hit.favorite_id
        bar_time = hit.bar_time or _extract_bar_time(row)
        channel_choice = channel
        if channel_choice is None:
            if isinstance(row, Mapping):
                channel_choice = row.get("channel") or row.get("delivery_channel")
            if channel_choice is None and isinstance(fav, Mapping):
                channel_choice = fav.get("channel") or fav.get("delivery_channel")

        settings_profile = getattr(hit, "greeks_profile_json", "{}") or "{}"
        override_json = getattr(hit, "greeks_override_json", None)
        profile_all = load_profile(settings_profile, override_json)
        profile = profile_all.get("direction_profiles", {}).get(hit.direction.upper(), {})
        side = "call" if hit.direction.upper() == "UP" else "put"
        sel = select_contract(hit.ticker, side, profile)
        if not sel.contract:
            return False

        checks = evaluate_contract(sel.contract, profile)
        targets = {
            "target": hit.target_pct,
            "stop": hit.stop_pct,
            "hit": hit.hit_pct,
            "roi": hit.avg_roi_pct,
            "dd": hit.avg_dd_pct,
        }
        compact_pref = bool(profile.get("compact_mms"))
        include_symbols = bool(profile.get("include_symbols_in_alerts", True))
        format_channel = "email" if _normalize_channel(channel_choice) == "email" else "mms"
        body = format_favorites_alert(
            hit.ticker,
            hit.direction,
            sel.contract,
            checks,
            targets,
            compact=compact_pref,
            channel=format_channel,
            pattern=hit.pattern,
            include_symbols=include_symbols,
        )
        subject = f"Favorites Alert: {hit.ticker.upper()} {hit.direction.upper()}"
        if hit.pattern:
            subject += f" {hit.pattern}"

        normalized_channel = _normalize_channel(channel_choice)
        include_all_outcomes = _outcomes_mode() == "all"
        skip_non_entry = _should_skip_non_entry(row)
        symbol_for_log = (
            (_value(row, "symbol") or _value(row, "ticker") or hit.ticker)
            if row is not None
            else hit.ticker
        )
        direction_for_log = (
            (_value(row, "direction") or hit.direction)
            if row is not None
            else hit.direction
        )
        delivery_context = {
            "symbol": symbol_for_log,
            "direction": direction_for_log,
            "bar_time": str(bar_time) if bar_time is not None else None,
        }
        if favorite_id:
            delivery_context["fav_id"] = favorite_id
        delivery_context["simulated"] = bool(simulated)

        outcome_for_log: str | None = None
        if isinstance(row, Mapping):
            for key in ("outcome", "status", "event", "signal"):
                raw_val = row.get(key)
                if isinstance(raw_val, str) and raw_val.strip():
                    outcome_for_log = raw_val.strip().lower()
                    break
        if not outcome_for_log and not include_all_outcomes:
            outcome_for_log = "hit"

        fallback_reason: str | None = None

        def _emit(payload: Dict[str, Any], *, fallback: str | None = None) -> Dict[str, Any]:
            extra = {
                "symbol": symbol_for_log,
                "direction": direction_for_log,
                "outcome": outcome_for_log,
                "channel": payload.get("channel"),
                "dedupe_key": payload.get("dedupe_key"),
                "reason": payload.get("reason"),
                "ok": bool(payload.get("ok")),
                "fallback": fallback,
                "simulated": bool(simulated),
            }
            logger.info("favorites_alert_delivery", extra=extra)
            return payload

        email_recipients = list(recipients or [])
        if not email_recipients:
            if isinstance(row, Mapping):
                email_recipients = _parse_list(row.get("recipients"))
            if not email_recipients and isinstance(fav, Mapping):
                email_recipients = _parse_list(fav.get("recipients"))

        smtp_cfg = dict(smtp_config or {})
        if not smtp_cfg:
            smtp_cfg = _merge_smtp_config(fav, row)

        interval_value = None
        if isinstance(row, Mapping):
            interval_value = row.get("interval")
        if interval_value in (None, "") and isinstance(fav, Mapping):
            interval_value = fav.get("interval")
        if interval_value in (None, ""):
            interval_value = profile.get("interval") or getattr(hit, "interval", None) or "15m"

        dedupe_value = _dedupe_key(favorite_id, interval_value, bar_time)
        if dedupe_value and was_sent_key(
            favorite_id,
            bar_time,
            interval=interval_value,
            dedupe_key=dedupe_value,
        ):
            _log_skip("skip_dedupe", dedupe_value)
            final_channel = normalized_channel or _normalize_channel(channel_choice) or "mms"
            delivered = False
            reason = "throttled"
            payload = {
                "ok": delivered,
                "channel": final_channel or "mms",
                "dedupe_key": dedupe_value,
                "reason": reason,
            }
            log_telemetry(
                {
                    "type": "favorites_alert_test" if is_test else "favorites_alert",
                    "task_id": "<task_id_or_test>",
                    "ticker": hit.ticker,
                    "direction": hit.direction,
                    "channel": payload["channel"],
                    "delivered": delivered,
                }
            )
            return _emit(payload)

        delivered = False
        dedupe_result: Optional[str] = dedupe_value
        chosen_channel = normalized_channel or "mms"
        final_channel = chosen_channel
        reason = "delivery_error"
        bodies = {
            "email": format_favorites_alert(
                hit.ticker,
                hit.direction,
                sel.contract,
                checks,
                targets,
                compact=compact_pref,
                channel="email",
                pattern=hit.pattern,
                include_symbols=include_symbols,
            ),
            "mms": body,
            "sms": format_favorites_alert(
                hit.ticker,
                hit.direction,
                sel.contract,
                checks,
                targets,
                compact=compact_pref,
                channel="sms",
                pattern=hit.pattern,
                include_symbols=include_symbols,
            ),
        }

        def _finish() -> Dict[str, Any]:
            payload = {
                "ok": bool(delivered),
                "channel": final_channel or "mms",
                "dedupe_key": dedupe_result,
                "reason": reason,
            }
            log_telemetry(
                {
                    "type": "favorites_alert_test" if is_test else "favorites_alert",
                    "task_id": "<task_id_or_test>",
                    "ticker": hit.ticker,
                    "direction": hit.direction,
                    "channel": payload["channel"],
                    "delivered": delivered,
                }
            )
            return _emit(payload, fallback=fallback_reason)

        if not (include_all_outcomes or not skip_non_entry):
            reason = "skipped"
            logger.info("favorites alert skipped non-entry event favorite_id=%s", favorite_id)
            delivered = False
            return _finish()

        recipients_arg = email_recipients if chosen_channel == "email" else None
        try:
            send_body = bodies.get(chosen_channel, body)
            delivered, dedupe_result, sent_channel, send_reason = _deliver_alert(
                chosen_channel,
                subject,
                send_body,
                recipients=recipients_arg,
                favorite_id=favorite_id,
                bar_time=bar_time,
                interval=interval_value,
                dedupe_key=dedupe_value,
                smtp_config=smtp_cfg,
                delivery_context=delivery_context,
            )
            if sent_channel:
                final_channel = sent_channel
            reason = send_reason
        except RuntimeError as exc:
            delivered = False
            reason = "delivery_error"
            error_message = str(exc)
            fallback_reason = error_message or "twilio_unavailable"
            logger.warning(
                "favorites alert %s send failed: %s",
                chosen_channel,
                error_message,
            )
            fallback_body = f"{bodies.get('sms', body)}\n\n[Sent via Email — MMS unavailable]"
            fallback_context = {**delivery_context, "fallback_from": chosen_channel}
            delivered, dedupe_result, sent_channel, send_reason = _deliver_alert(
                "email",
                subject,
                fallback_body,
                recipients=email_recipients,
                favorite_id=favorite_id,
                bar_time=bar_time,
                interval=interval_value,
                dedupe_key=dedupe_value,
                smtp_config=smtp_cfg,
                delivery_context=fallback_context,
            )
            if delivered:
                final_channel = sent_channel or "email"
                reason = send_reason
                bodies["mms"] = fallback_body
                chosen_channel = "email"
            else:
                reason = send_reason or "delivery_error"

        if delivered:
            logger.info(
                "favorite_alert_sent",
                extra={
                    "symbol": symbol_for_log,
                    "direction": direction_for_log,
                    "fav_id": favorite_id,
                    "bar_time": str(bar_time) if bar_time is not None else None,
                    "channel": final_channel,
                    "outcomes": getattr(
                        settings,
                        "ALERT_OUTCOMES",
                        getattr(settings, "alert_outcomes", "hit"),
                    ),
                    "hit_lb95": _value(row, "hit_lb95"),
                    "support": _value(row, "support"),
                    "avg_roi": _value(row, "avg_roi_pct") or _value(row, "avg_roi"),
                },
            )
            if dedupe_result:
                mark_sent_key(
                    favorite_id,
                    bar_time,
                    interval=interval_value,
                    dedupe_key=dedupe_result,
                    simulated=simulated,
                )
            reason = "sent"
        return _finish()
    except Exception:
        logger.exception("favorites alert enrichment failed")
        failure_payload = {
            "ok": False,
            "channel": _normalize_channel(channel) or "mms" if channel else "mms",
            "dedupe_key": None,
            "reason": "delivery_error",
        }
        logger.info("favorites_alert_delivery %s", json.dumps(failure_payload, sort_keys=True))
        return failure_payload


def deliver_preview_alert(
    subject: str,
    bodies: Mapping[str, str],
    *,
    channel: str,
    favorite_id: Any = None,
    bar_time: Any = None,
    interval: Any = "15m",
    dedupe_key: Optional[str] = None,
    recipients: Optional[Sequence[str]] = None,
    smtp_config: Optional[Mapping[str, Any]] = None,
    simulated: bool = False,
    outcome: str | None = None,
    symbol: str | None = None,
    direction: str | None = None,
) -> Dict[str, Any]:
    normalized_channel = _normalize_channel(channel) or "email"
    dedupe_value = dedupe_key or _dedupe_key(favorite_id, interval, bar_time)
    if dedupe_value and was_sent_key(
        favorite_id,
        bar_time,
        interval=interval,
        dedupe_key=dedupe_value,
    ):
        _log_skip("skip_dedupe", dedupe_value)
        payload = {
            "ok": False,
            "channel": normalized_channel,
            "dedupe_key": dedupe_value,
            "reason": "throttled",
        }
        logger.info(
            "favorites_alert_delivery",
            extra={
                "symbol": symbol,
                "direction": direction,
                "outcome": outcome,
                "channel": payload["channel"],
                "dedupe_key": dedupe_value,
                "reason": payload["reason"],
                "ok": False,
                "fallback": None,
                "simulated": bool(simulated),
            },
        )
        return payload

    bodies_map = dict(bodies or {})
    body_value = bodies_map.get(normalized_channel) or bodies_map.get("mms") or bodies_map.get("email")
    if body_value is None:
        body_value = subject

    smtp_cfg = dict(smtp_config or {}) or None

    delivery_context = {
        "symbol": symbol,
        "direction": direction,
        "bar_time": str(bar_time) if bar_time is not None else None,
        "simulated": bool(simulated),
        "outcome": outcome,
    }

    recipients_arg = recipients if normalized_channel == "email" else None
    fallback_reason: str | None = None
    try:
        delivered, dedupe_result, sent_channel, reason = _deliver_alert(
            normalized_channel,
            subject,
            body_value,
            recipients=recipients_arg,
            favorite_id=favorite_id,
            bar_time=bar_time,
            interval=interval,
            dedupe_key=dedupe_value,
            smtp_config=smtp_cfg,
            delivery_context=delivery_context,
        )
        final_channel = sent_channel or normalized_channel
    except RuntimeError as exc:
        fallback_reason = str(exc) or "twilio_unavailable"
        base_email_body = bodies_map.get("email") or body_value
        email_body = f"{base_email_body}\n\n[Sent via Email — MMS unavailable]"
        email_recipients = list(recipients or getattr(settings, "alert_email_to", ()))
        delivered, dedupe_result, sent_channel, reason = _deliver_alert(
            "email",
            subject,
            email_body,
            recipients=email_recipients,
            favorite_id=favorite_id,
            bar_time=bar_time,
            interval=interval,
            dedupe_key=dedupe_value,
            smtp_config=smtp_cfg,
            delivery_context={**delivery_context, "fallback_from": normalized_channel},
        )
        final_channel = sent_channel or "email"
        body_value = email_body

    payload = {
        "ok": bool(delivered),
        "channel": final_channel,
        "dedupe_key": dedupe_result or dedupe_value,
        "reason": reason,
    }

    if delivered and payload["dedupe_key"]:
        mark_sent_key(
            favorite_id,
            bar_time,
            interval=interval,
            dedupe_key=payload["dedupe_key"],
            simulated=simulated,
        )

    logger.info(
        "favorites_alert_delivery",
        extra={
            "symbol": symbol,
            "direction": direction,
            "outcome": outcome,
            "channel": payload["channel"],
            "dedupe_key": payload["dedupe_key"],
            "reason": payload["reason"],
            "ok": bool(payload["ok"]),
            "fallback": fallback_reason,
            "simulated": bool(simulated),
        },
    )
    log_telemetry(
        {
            "type": "favorites_alert_sim" if simulated else "favorites_alert",
            "task_id": "simulation" if simulated else "preview",
            "ticker": symbol,
            "direction": direction,
            "channel": payload["channel"],
            "delivered": bool(payload["ok"]),
        }
    )
    return payload


def enrich_and_send_test(
    ticker: str,
    direction: str,
    channel: str = "mms",
    *,
    compact: bool = False,
    outcomes: str = "hit",
) -> tuple[bool, dict[str, str]]:
    symbol = (ticker or "AAPL").upper()
    direction_norm = (direction or "UP").upper()
    side = "call" if direction_norm == "UP" else "put"
    normalized_channel = (channel or "mms").strip().lower() or "mms"
    normalized_outcomes = (outcomes or "hit").strip().lower() or "hit"
    outcomes_label = (
        "Hit + Stop + Timeout"
        if normalized_outcomes == "all"
        else "Hit only"
    )
    outcomes_subject = "Hit + Stop + Timeout" if normalized_outcomes == "all" else "Hit-only"
    today = datetime.utcnow().date()
    contract = options_provider.OptionContract(
        occ=f"{symbol}TEST",
        side=side,
        strike=190.0,
        expiry=today,
        bid=3.9,
        ask=4.5,
        mid=4.2,
        last=4.2,
        open_interest=1250,
        volume=480,
        delta=0.78,
        gamma=0.05,
        theta=-0.18,
        vega=0.28,
        spread_pct=8.5,
        dte=30,
        iv_rank=88.0,
    )
    checks = [
        Check(
            "Delta",
            "Δ",
            0.78,
            False,
            "Delta too high — moves almost 1:1 with stock; limited leverage.",
        ),
        Check(
            "Gamma",
            "Γ",
            0.05,
            False,
            "Gamma too low — option won’t pick up delta fast enough.",
        ),
        Check("Theta", "Θ", -0.18, True, "daily time decay."),
        Check("Vega", "ν", 0.28, True, "sensitivity to volatility changes."),
        Check(
            "IV Rank",
            "IVR",
            88.0,
            False,
            "IV Rank high — premiums rich vs. history; consider spreads.",
        ),
    ]
    targets = {"target": 185.0, "stop": 194.0, "hit": 76, "roi": 19, "dd": 12}
    body = format_favorites_alert(
        symbol,
        direction_norm,
        contract,
        checks,
        targets,
        compact=compact,
        channel=normalized_channel,
        pattern="Manual Test",
        include_symbols=True,
    )
    if outcomes_label:
        if normalized_channel == "sms":
            body = f"{body} • Outcomes {outcomes_label}".strip()
        else:
            body = f"{body}\n\nOutcomes Mode: {outcomes_label}".strip()
    subject = f"[TEST {outcomes_subject}] {symbol} — {direction_norm} Hit (Manual Test)"
    channel_field = (
        normalized_channel if normalized_channel in {"email", "sms"} else "mms"
    )
    return True, {
        "subject": subject,
        "body": body,
        "channel": channel_field,
        "outcomes": normalized_outcomes,
    }


def build_preview(
    symbol: str,
    direction: str = "UP",
    *,
    channel: str = "Email",
    outcomes: str = "hit",
    compact: bool = False,
    outcome_mode: str | None = None,
    simulated: bool = False,
) -> tuple[str, str]:
    normalized_channel = (channel or "Email").strip().lower() or "email"
    normalized_outcomes = (outcomes or "hit").strip().lower() or "hit"
    direction_norm = (direction or "UP").strip().upper() or "UP"
    ok, payload = enrich_and_send_test(
        symbol,
        direction_norm,
        channel=normalized_channel,
        compact=compact,
        outcomes=normalized_outcomes,
    )
    if not ok or not isinstance(payload, dict):
        return "", ""
    subject = str(payload.get("subject", ""))
    body = str(payload.get("body", ""))
    mode_value = (outcome_mode or normalized_outcomes or "hit").strip().lower()
    tag = "Hit-only" if mode_value != "all" else "All"
    prefix = f"[TEST {tag}]"
    if prefix not in subject:
        subject = f"{prefix} {subject}" if subject else prefix
    if simulated and "(Simulated)" not in subject:
        subject = subject.replace("Favorites Alert:", "Favorites Alert (Simulated):")
    return subject, body

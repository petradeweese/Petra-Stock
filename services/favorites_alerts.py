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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import settings

from . import events_provider, options_provider, sms_consent, twilio_client
from .notify import send_email_smtp
from .telemetry import log as log_telemetry


logger = logging.getLogger(__name__)

_SENT_ALERTS: set[Tuple[str, str]] = set()


def _outcomes_mode() -> str:
    raw = getattr(settings, "alert_outcomes", getattr(settings, "ALERT_OUTCOMES", "hit"))
    value = str(raw or "hit").strip().lower()
    return value or "hit"


def was_sent(favorite_id: Any, bar_time: Any) -> bool:
    """Return ``True`` if the alert for ``favorite_id``/``bar_time`` was sent."""

    if favorite_id in (None, "", b"") or bar_time in (None, "", b""):
        return False
    key = (str(favorite_id), str(bar_time))
    return key in _SENT_ALERTS


def mark_sent(favorite_id: Any, bar_time: Any) -> None:
    """Record that an alert was delivered for the given favorite/bar."""

    if favorite_id in (None, "", b"") or bar_time in (None, "", b""):
        return
    key = (str(favorite_id), str(bar_time))
    _SENT_ALERTS.add(key)


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
    if value in {"email", "mms"}:
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
        return False

    avg_roi = _coerce_float(_value(row, "avg_roi_pct"))
    if avg_roi == 0.0:
        avg_roi = _coerce_float(_value(row, "avg_roi"))
        if abs(avg_roi) <= 1.0:
            avg_roi *= 100.0

    min_roi = _coerce_float(_value(fav, "min_avg_roi_pct"), 0.0)
    if avg_roi <= max(0.0, min_roi):
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
    smtp_config: Optional[Mapping[str, Any]] = None,
    delivery_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Optional[Tuple[str, str]], Optional[str]]:
    normalized = _normalize_channel(channel)
    if normalized is None:
        return False, None, None

    context_base: Dict[str, Any] = {}
    if delivery_context:
        context_base.update({k: v for k, v in delivery_context.items() if v is not None})
    if favorite_id is not None and "fav_id" not in context_base:
        context_base["fav_id"] = str(favorite_id)

    dedupe_key: Optional[Tuple[str, str]] = None
    if favorite_id and bar_time:
        fav_id_str = str(favorite_id)
        bar_time_str = str(bar_time)
        dedupe_key = (fav_id_str, bar_time_str)
        if was_sent(fav_id_str, bar_time_str):
            logger.info(
                "favorites alert dedupe skip favorite_id=%s bar_time=%s",
                fav_id_str,
                bar_time_str,
            )
            return False, None, normalized

    success = False
    if normalized == "email":
        send_to = [r for r in (recipients or []) if r]
        if not send_to:
            logger.info("favorites alert email skipped: no recipients configured")
        else:
            cfg = dict(smtp_config or {})
            host = cfg.get("host")
            port = cfg.get("port")
            user = cfg.get("user", "")
            password = cfg.get("password", "")
            mail_from = cfg.get("mail_from") or user
            if not host or not port or not mail_from:
                logger.warning("favorites alert email skipped: incomplete smtp config")
            else:
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
                    result = {"ok": False}
                if result.get("ok"):
                    success = True
                    logger.info(
                        "favorites alert email sent recipients=%d", len(send_to)
                    )
                else:
                    logger.warning(
                        "favorites alert email failed: %s", result.get("error")
                    )
    elif normalized == "mms":
        destinations = []
        if recipients is not None:
            for entry in recipients:
                if isinstance(entry, Mapping):
                    phone = entry.get("phone_e164") or entry.get("phone") or entry.get("to")
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
        else:
            seen: set[str] = set()
            message_body = sms_consent.append_footer(body)
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
                    "channel": "mms",
                    "to": number_str,
                    "user_id": user_for_log,
                }
                if twilio_client.send_mms(number_str, message_body, context=context):
                    sms_consent.record_delivery(number_str, user_for_log, message_body)
                    success = True
                else:
                    logger.warning("favorites alert mms failed number=%s", number_str)
    else:
        logger.info("favorites alert skipped: unknown channel=%s", normalized)

    return success, dedupe_key if success else None, normalized
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
    picked = _format_picked(contract)
    header_line = f"{header} | Picked: {picked}"
    sections: List[List[str]] = [[header_line]]
    if pattern:
        sections[0].append(f"Setup: {pattern}")

    greeks_lines = ["Greeks & IV:"]
    for chk in checks:
        line = _format_greek_line(chk, compact=compact)
        if line:
            greeks_lines.append(line)
    if len(greeks_lines) == 1:
        greeks_lines.append("• No checks evaluated")
    sections.append(greeks_lines)

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
) -> bool:  # pragma: no cover - orchestrator
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
            avg_roi = _coalesce(
                [
                    (row or {}).get("avg_roi_pct") if isinstance(row, Mapping) else None,
                    fav.get("roi_snapshot") if isinstance(fav, Mapping) else None,
                ]
            ) or 0.0
            avg_dd = _coalesce(
                [
                    (row or {}).get("avg_dd_pct") if isinstance(row, Mapping) else None,
                    fav.get("dd_pct_snapshot") if isinstance(fav, Mapping) else None,
                ]
            ) or 0.0
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

        delivered = False
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

        if channel_choice and (include_all_outcomes or not skip_non_entry):
            email_recipients = list(recipients or [])
            if not email_recipients:
                if isinstance(row, Mapping):
                    email_recipients = _parse_list(row.get("recipients"))
                if not email_recipients and isinstance(fav, Mapping):
                    email_recipients = _parse_list(fav.get("recipients"))
            smtp_cfg = dict(smtp_config or {})
            if not smtp_cfg:
                smtp_cfg = _merge_smtp_config(fav, row)
            recipients_arg = email_recipients if normalized_channel == "email" else None
            delivered, dedupe_key, normalized_sent = _deliver_alert(
                channel_choice,
                subject,
                body,
                recipients=recipients_arg,
                favorite_id=favorite_id,
                bar_time=bar_time,
                smtp_config=smtp_cfg,
                delivery_context=delivery_context,
            )
            if delivered:
                logger.info(
                    "favorite_alert_sent",
                    extra={
                        "symbol": symbol_for_log,
                        "direction": direction_for_log,
                        "fav_id": favorite_id,
                        "bar_time": str(bar_time) if bar_time is not None else None,
                        "channel": getattr(settings, "alert_channel", "Email"),
                        "outcomes": getattr(settings, "ALERT_OUTCOMES", getattr(settings, "alert_outcomes", "hit")),
                        "hit_lb95": _value(row, "hit_lb95"),
                        "support": _value(row, "support"),
                        "avg_roi": _value(row, "avg_roi_pct") or _value(row, "avg_roi"),
                    },
                )
                if dedupe_key:
                    mark_sent(*dedupe_key)
            normalized_channel = normalized_sent
        elif channel_choice and not include_all_outcomes and skip_non_entry:
            logger.info("favorites alert skipped non-entry event favorite_id=%s", favorite_id)

        log_telemetry(
            {
                "type": "favorites_alert_test" if is_test else "favorites_alert",
                "task_id": "<task_id_or_test>",
                "ticker": hit.ticker,
                "direction": hit.direction,
                "channel": channel_choice,
                "delivered": delivered,
            }
        )
        return delivered
    except Exception:
        logger.exception("favorites alert enrichment failed")
        return False


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
        body = f"{body}\n\nOutcomes Mode: {outcomes_label}".strip()
    subject = f"Favorites Alert Test: {symbol} {direction_norm}"
    return True, {
        "subject": subject,
        "body": body,
        "channel": "email" if normalized_channel == "email" else "mms",
        "outcomes": normalized_outcomes,
    }


def build_preview(
    symbol: str,
    *,
    channel: str = "Email",
    outcomes: str = "hit",
    compact: bool = False,
) -> tuple[str, str]:
    normalized_channel = (channel or "Email").strip().lower() or "email"
    normalized_outcomes = (outcomes or "hit").strip().lower() or "hit"
    ok, payload = enrich_and_send_test(
        symbol,
        "UP",
        channel=normalized_channel,
        compact=compact,
        outcomes=normalized_outcomes,
    )
    if not ok or not isinstance(payload, dict):
        return "", ""
    subject = str(payload.get("subject", ""))
    body = str(payload.get("body", ""))
    return subject, body

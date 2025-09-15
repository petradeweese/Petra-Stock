"""Favorites alert helpers for contract selection and evaluation.

The production system enriches favorites scan hits with option contract
information and sends multi-line MMS alerts.  For the unit tests in this
repository we provide a greatly simplified but fully functional subset of the
logic described in the specification.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from . import options_provider, events_provider
from .telemetry import log as log_telemetry


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


@dataclass
class Check:
    name: str
    symbol: str
    value: float
    passed: bool
    explanation: str | None = None


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


def enrich_and_send(hit: FavoriteHitStub, is_test: bool = False) -> bool:  # pragma: no cover - orchestrator
    """Orchestrate fetching data, evaluating and sending an alert.

    The full implementation would interact with external services.  For tests we
    simply run through the selection and formatting steps and log telemetry.
    Returns ``True`` when data was available, ``False`` when falling back to the
    legacy minimal alert.
    """

    try:
        settings_profile = hit.__dict__.get("greeks_profile_json", "{}")
        override_json = hit.__dict__.get("greeks_override_json")
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
        _ = format_favorites_alert(
            hit.ticker,
            hit.direction,
            sel.contract,
            checks,
            targets,
            compact=compact_pref,
            channel="mms",
            pattern=hit.pattern,
            include_symbols=include_symbols,
        )
        log_telemetry(
            {
                "type": "favorites_alert_test" if is_test else "favorites_alert",
                "task_id": "<task_id_or_test>",
                "ticker": hit.ticker,
                "direction": hit.direction,
            }
        )
        return True
    except Exception:
        return False


def enrich_and_send_test(
    ticker: str, direction: str, channel: str = "mms", compact: bool = False
) -> tuple[bool, dict[str, str]]:
    symbol = (ticker or "AAPL").upper()
    direction_norm = (direction or "UP").upper()
    side = "call" if direction_norm == "UP" else "put"
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
        channel=channel,
        pattern="Manual Test",
        include_symbols=True,
    )
    subject = f"Favorites Alert Test: {symbol} {direction_norm}"
    return True, {"subject": subject, "body": body}

"""Favorites alert helpers for contract selection and evaluation.

The production system enriches favorites scan hits with option contract
information and sends multi-line MMS alerts.  For the unit tests in this
repository we provide a greatly simplified but fully functional subset of the
logic described in the specification.
"""
from __future__ import annotations

import json
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
        if (dte_min is None or c.dte >= dte_min)
        and (dte_max is None or c.dte <= dte_max)
    ]
    target_delta = profile.get("target_delta", 0.0)
    rejects: List[Dict[str, str]] = []
    passes: List[options_provider.OptionContract] = []
    for c in candidates:
        reason = None
        min_oi = profile.get("min_open_interest")
        min_vol = profile.get("min_volume")
        max_spread = profile.get("max_spread_pct")
        if min_oi is not None and c.open_interest < min_oi:
            reason = "open interest too low"
        elif min_vol is not None and c.volume < min_vol:
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


def format_mms(
    hit: FavoriteHitStub,
    selection: options_provider.OptionContract,
    checks: List[Check],
    profile: Dict,
    settings: Dict | None = None,
) -> str:
    """Return a formatted MMS body."""

    lines = [f"{hit.ticker} {hit.direction} {hit.pattern}"]
    if selection:
        lines.append(f"Contract {selection.occ}")
    compact = bool(profile.get("compact_mms"))
    include_symbols = bool(profile.get("include_symbols_in_alerts", True))
    for chk in checks:
        if compact and chk.passed:
            continue
        name = chk.name
        if include_symbols and chk.symbol:
            name = f"{name} ({chk.symbol})"
        status = "✅" if chk.passed else "❌"
        line = f"{name}: {chk.value} {status}"
        if chk.explanation and not chk.passed:
            line += f" — {chk.explanation}"
        lines.append(line)
    if selection:
        lines.append(
            "Why this contract: "
            f"Δ {selection.delta:.2f} | DTE {selection.dte} | spread {selection.spread_pct:.1f}% | IVR {selection.iv_rank}"
        )
    return "\n".join(lines)


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
        format_mms(hit, sel.contract, checks, profile)
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


def enrich_and_send_test(ticker: str, direction: str) -> bool:
    fake_hit = FavoriteHitStub(ticker=ticker, direction=direction, pattern="Manual Test")
    return enrich_and_send(fake_hit, is_test=True)

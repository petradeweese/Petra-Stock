"""Similarity search over intraday feature states."""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from services.forecast_features import (
    FEATURE_ORDER,
    FRAME_ORDER,
    build_state,
    ensure_utc,
    load_price_frame,
    LOOKBACK_DAYS,
    tod_minute,
)
from utils import OPEN_TIME, TZ, market_is_open


logger = logging.getLogger(__name__)


_DEFAULT_WEIGHTS = {"5m": 0.5, "30m": 0.3, "1d": 0.2}
_WEIGHT_ENV_MAP = {"5m": "FORECAST_W5", "30m": "FORECAST_W30", "1d": "FORECAST_W1D"}


def _coerce_weight(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return max(0.0, parsed)


class SimilarityScorer:
    """Compute weighted multi-frame similarity scores."""

    def __init__(self) -> None:
        weights: Dict[str, float] = {}
        for frame, default in _DEFAULT_WEIGHTS.items():
            env_value = os.getenv(_WEIGHT_ENV_MAP[frame])
            weights[frame] = _coerce_weight(env_value, default)
        total = sum(weights.values()) or 1.0
        self._base_weights = {frame: weight / total for frame, weight in weights.items()}

    @property
    def base_weights(self) -> Dict[str, float]:
        return dict(self._base_weights)

    def score(self, frame_scores: Dict[str, float]) -> Tuple[Dict[str, object], float]:
        components: Dict[str, float] = {}
        for frame in _DEFAULT_WEIGHTS:
            val = frame_scores.get(frame)
            if val is None or math.isnan(val) or math.isinf(val):
                continue
            components[frame] = float(val)

        if not components:
            return {"weights": {}, "final_score": 0.0}, 0.0

        weight_total = sum(self._base_weights.get(frame, 0.0) for frame in components)
        if weight_total <= 0:
            normalized = {frame: 1.0 / len(components) for frame in components}
        else:
            normalized = {
                frame: self._base_weights.get(frame, 0.0) / weight_total for frame in components
            }

        final_score = float(sum(normalized[frame] * components[frame] for frame in components))
        breakdown = {
            "S5m": components.get("5m", math.nan),
            "S30m": components.get("30m", math.nan),
            "S1d": components.get("1d", math.nan),
            "weights": normalized,
            "final_score": final_score,
        }
        return breakdown, final_score


def _coerce_float(value: object) -> Optional[float]:
    """Return ``value`` as a finite float or ``None``."""

    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _option_field(contract: object, *names: str) -> object:
    """Return the first non-null attribute or mapping field from ``names``."""

    for name in names:
        if hasattr(contract, name):
            value = getattr(contract, name)
            if value is not None:
                return value
        if isinstance(contract, dict) and name in contract:
            value = contract[name]
            if value is not None:
                return value
    return None


def _fetch_iv_rank(ticker: str) -> Optional[float]:
    """Best-effort retrieval of the current IV rank for ``ticker``."""

    try:
        from services import options_provider
    except Exception:  # pragma: no cover - optional dependency failure
        return None

    try:
        chain = options_provider.get_chain(ticker.upper())
    except Exception:  # pragma: no cover - defensive
        return None

    best: Optional[Tuple[Tuple[float, float, float], float]] = None
    for contract in chain or []:
        iv_rank_raw = _option_field(
            contract, "iv_rank", "ivRank", "ivr", "ivRankPercentile"
        )
        iv_rank_val = _coerce_float(iv_rank_raw)
        if iv_rank_val is None:
            continue

        delta_val = _coerce_float(_option_field(contract, "delta"))
        distance = abs(abs(delta_val) - 0.5) if delta_val is not None else 0.6

        dte_val = _coerce_float(
            _option_field(contract, "dte", "days_to_expiration", "daysToExpiration")
        )
        dte_rank = abs(dte_val) if dte_val is not None else 999.0

        oi_val = _coerce_float(
            _option_field(contract, "open_interest", "openInterest")
        )
        oi_rank = -(oi_val or 0.0)

        score = (distance, dte_rank, oi_rank)
        if best is None or score < best[0]:
            best = (score, iv_rank_val)

    return best[1] if best else None


def _iv_rank_hint(iv_rank: Optional[float]) -> str:
    """Translate ``iv_rank`` to a qualitative label."""

    if iv_rank is None:
        return "neutral"
    if iv_rank < 30:
        return "cheap"
    if iv_rank > 70:
        return "rich"
    return "neutral"


def _safe_bias(median_close: Optional[float]) -> str:
    """Return directional bias from ``median_close``."""

    if median_close is None or math.isnan(median_close):
        return "Up"
    return "Up" if median_close >= 0 else "Down"


@dataclass
class MatchCandidate:
    timestamp: dt.datetime
    similarity: float
    frame_scores: Dict[str, float]
    breakdown: Dict[str, object]
    outcomes: Dict[str, float]
    source: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _trading_days(end: dt.date, lookback_days: int) -> List[dt.date]:
    start = end - dt.timedelta(days=max(1, lookback_days))
    day = start
    days: List[dt.date] = []
    while day < end:
        midday = dt.datetime.combine(day, dt.time(12, 0), tzinfo=TZ)
        if market_is_open(midday):
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def _offset_sequence(tolerance: int) -> Iterable[int]:
    yield 0
    for delta in range(1, tolerance + 1):
        yield delta
        yield -delta


def _target_time(day: dt.date, minutes: int, offset: int) -> Optional[dt.datetime]:
    minute_offset = minutes + offset
    if minute_offset < 0 or minute_offset > 390:
        return None
    base = dt.datetime.combine(day, OPEN_TIME, tzinfo=TZ)
    candidate = base + dt.timedelta(minutes=minute_offset)
    close_dt = base + dt.timedelta(hours=6, minutes=30)
    if candidate >= close_dt:
        return None
    return candidate.astimezone(dt.timezone.utc)


def _vector(state: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    vec = np.array(state.get("vec", []), dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    layout = state.get("layout") or {}
    return vec, layout


def _frame_slice(
    layout: Dict[str, Tuple[int, int]], frame: str, vector: np.ndarray
) -> Optional[np.ndarray]:
    bounds = layout.get(frame)
    if not bounds:
        return None
    start, end = bounds
    if end <= start:
        return None
    return vector[start:end]


def _winsorize(values: Sequence[float], lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    low = np.nanpercentile(arr, lower)
    high = np.nanpercentile(arr, upper)
    return np.clip(arr, low, high)


def _compute_outcomes(ticker: str, timestamp: dt.datetime) -> Tuple[Dict[str, float], str]:
    target_date = timestamp.astimezone(TZ).date()
    start = dt.datetime.combine(target_date, OPEN_TIME, tzinfo=TZ).astimezone(dt.timezone.utc)
    end = start + dt.timedelta(hours=6, minutes=30)
    minute_bars, source = load_price_frame(
        ticker, start, end + dt.timedelta(minutes=1), "1m"
    )
    if minute_bars is None or minute_bars.empty:
        return {}, source
    session = minute_bars.loc[(minute_bars.index >= start) & (minute_bars.index <= end)]
    if session.empty:
        return {}, source
    high = float(session["High"].max())
    low = float(session["Low"].min())
    close_price = float(session["Close"].iloc[-1])
    prior_start = start - dt.timedelta(days=7)
    daily_bars, _ = load_price_frame(ticker, prior_start, start, "1d")
    prior_close = None
    if daily_bars is not None and not daily_bars.empty:
        prev = daily_bars.loc[daily_bars.index < start]
        if not prev.empty:
            prior_close = float(prev["Close"].iloc[-1])
    if not prior_close:
        prev_minute, _ = load_price_frame(
            ticker, start - dt.timedelta(hours=6, minutes=30), start, "1m"
        )
        if prev_minute is not None and not prev_minute.empty:
            prior_close = float(prev_minute["Close"].iloc[-1])
    if not prior_close or prior_close == 0:
        return {}, source
    close_pct = (close_price / prior_close - 1.0) * 100.0
    high_pct = (high / prior_close - 1.0) * 100.0
    low_pct = (low / prior_close - 1.0) * 100.0
    return {
        "close_pct": close_pct,
        "high_pct": high_pct,
        "low_pct": low_pct,
    }, source


def find_similar_days(
    ticker: str,
    state: Optional[Dict[str, object]],
    asof: dt.datetime,
    lookback_days: int = LOOKBACK_DAYS,
    k: int = 100,
    per_frame_min: float = 0.85,
    overall_min: float = 0.90,
    tod_tolerance_min: int = 10,
) -> Dict[str, object]:
    """Find historical sessions with feature vectors similar to ``state``."""

    asof_utc = ensure_utc(asof)
    base_state = state or build_state(ticker, asof_utc)
    base_vec, layout = _vector(base_state)
    base_frames = base_state.get("frames") if isinstance(base_state, dict) else {}
    scorer = SimilarityScorer()
    frame_counts_zero = False
    if isinstance(base_frames, dict):
        for frame_data in base_frames.values():
            if not isinstance(frame_data, dict):
                continue
            count_val = frame_data.get("count")
            if count_val is None:
                count_val = frame_data.get("n")
            if count_val in (0, "0"):
                frame_counts_zero = True
                break
    if base_vec.size == 0 or frame_counts_zero:
        sources_payload = base_state.get("sources") if isinstance(base_state, dict) else {}
        if not isinstance(sources_payload, dict):
            sources_payload = {"bars": str(sources_payload)}
        bars_source = sources_payload.get("bars", "none") if isinstance(sources_payload, dict) else "none"
        minimal_sources = sources_payload or {"bars": bars_source}
        return {
            "ticker": ticker.upper(),
            "asof": base_state.get("asof", asof_utc.isoformat()),
            "n": 0,
            "low_sample": True,
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "summary": {},
            "matches": [],
            "sources": minimal_sources,
        }

    tod_minute_val = int(base_state.get("tod_minute", tod_minute(asof_utc)))
    target_day = asof_utc.astimezone(TZ).date()
    trading_days = _trading_days(target_day, lookback_days)
    if trading_days and target_day in trading_days:
        trading_days.remove(target_day)
    if trading_days:
        trading_days = [day for day in trading_days if day < target_day]
    recent_cut = set(trading_days[-10:])
    trading_days = [day for day in trading_days if day not in recent_cut]
    candidates: List[MatchCandidate] = []
    base_sources_raw = base_state.get("sources", {}).get("bars", "none")
    base_sources = set(filter(None, (base_sources_raw or "").split("|")))
    sources_used = base_sources or {"none"}

    for day in reversed(trading_days):
        best: Optional[MatchCandidate] = None
        for offset in _offset_sequence(tod_tolerance_min):
            candidate_ts = _target_time(day, tod_minute_val, offset)
            if candidate_ts is None or candidate_ts >= asof_utc:
                continue
            try:
                cand_state = build_state(ticker, candidate_ts)
            except Exception:
                continue
            cand_vec, cand_layout = _vector(cand_state)
            if cand_vec.size != base_vec.size:
                continue
            frame_scores: Dict[str, float] = {}
            passes = True
            for frame, _, _ in FRAME_ORDER:
                base_slice = _frame_slice(layout, frame, base_vec)
                cand_slice = _frame_slice(cand_layout, frame, cand_vec)
                if base_slice is None or cand_slice is None:
                    continue
                score = _cosine(base_slice, cand_slice)
                frame_scores[frame] = score
                if score < per_frame_min:
                    passes = False
                    break
            if not frame_scores or not passes:
                continue
            breakdown, overall = scorer.score(frame_scores)
            if overall < overall_min:
                continue
            outcomes, source = _compute_outcomes(ticker, candidate_ts)
            if not outcomes:
                continue
            for part in (source or "none").split("|"):
                sources_used.add(part)
            match = MatchCandidate(
                candidate_ts, overall, frame_scores, breakdown, outcomes, source
            )
            if best is None or match.similarity > best.similarity:
                best = match
        if best is not None:
            candidates.append(best)

    candidates.sort(key=lambda m: m.similarity, reverse=True)
    matches = candidates[:k]
    n = len(matches)
    low_sample = n < 8

    close_vals = [m.outcomes["close_pct"] for m in matches]
    high_vals = [m.outcomes["high_pct"] for m in matches]
    low_vals = [m.outcomes["low_pct"] for m in matches]
    winsor_close = _winsorize(close_vals)
    winsor_high = _winsorize(high_vals)
    winsor_low = _winsorize(low_vals)

    summary = {}
    if n:
        summary = {
            "median_close_pct": float(np.nanmedian(winsor_close)) if winsor_close.size else math.nan,
            "iqr_close_pct": [
                float(np.nanpercentile(winsor_close, 25)) if winsor_close.size else math.nan,
                float(np.nanpercentile(winsor_close, 75)) if winsor_close.size else math.nan,
            ],
            "p5_p95_close_pct": [
                float(np.nanpercentile(winsor_close, 5)) if winsor_close.size else math.nan,
                float(np.nanpercentile(winsor_close, 95)) if winsor_close.size else math.nan,
            ],
            "median_high_pct": float(np.nanmedian(winsor_high)) if winsor_high.size else math.nan,
            "median_low_pct": float(np.nanmedian(winsor_low)) if winsor_low.size else math.nan,
        }

    clean_iqr: List[float] = []
    iqr_vals = summary.get("iqr_close_pct") if summary else None
    if isinstance(iqr_vals, Sequence) and len(iqr_vals) == 2:
        for val in iqr_vals:
            coerced = _coerce_float(val)
            clean_iqr.append(coerced if coerced is not None else math.nan)
    else:
        clean_iqr = [math.nan, math.nan]
    summary["iqr_close_pct"] = clean_iqr

    median_close = _coerce_float(summary.get("median_close_pct") if summary else None)
    median_high = _coerce_float(summary.get("median_high_pct") if summary else None)
    median_low = _coerce_float(summary.get("median_low_pct") if summary else None)

    if summary is None or not summary:
        summary = {}

    summary["median_close_pct"] = median_close if median_close is not None else math.nan
    summary["median_high_pct"] = median_high if median_high is not None else math.nan
    summary["median_low_pct"] = median_low if median_low is not None else math.nan
    summary["expected_move_iqr"] = list(clean_iqr)

    iv_rank_val = _fetch_iv_rank(ticker)
    summary["iv_rank_hint"] = _iv_rank_hint(iv_rank_val)

    population_scores = np.array([cand.similarity for cand in candidates], dtype=float)
    top_score = matches[0].similarity if matches else 0.0
    if population_scores.size:
        percentile = float((population_scores <= top_score + 1e-9).sum()) / float(
            population_scores.size
        )
    else:
        percentile = 0.0
    confidence = max(0.0, min(percentile, 1.0))
    confidence_pct = round(confidence * 100.0, 1)
    bias = _safe_bias(median_close)
    summary["bias"] = bias
    summary["confidence_pct"] = confidence_pct

    if matches:
        top_breakdown = matches[0].breakdown
        logger.info(
            "forecast similarity ticker=%s S5m=%.4f S30m=%.4f S1d=%.4f weights=%s final=%.4f",
            ticker.upper(),
            float(top_breakdown.get("S5m", math.nan)),
            float(top_breakdown.get("S30m", math.nan)),
            float(top_breakdown.get("S1d", math.nan)),
            top_breakdown.get("weights", {}),
            float(top_breakdown.get("final_score", 0.0)),
        )

    match_payload = [
        {
            "date": match.timestamp.astimezone(TZ).date().isoformat(),
            "similarity": round(match.similarity, 4),
            "time_of_match": match.timestamp.astimezone(TZ).strftime("%H:%M"),
            "close_pct": round(match.outcomes["close_pct"], 3),
            "high_pct": round(match.outcomes["high_pct"], 3),
            "low_pct": round(match.outcomes["low_pct"], 3),
            "timestamp": match.timestamp.isoformat(),
            "similarity_breakdown": match.breakdown,
        }
        for match in matches
    ]

    sources_used.discard("none")
    sources_used.discard("")
    sources_str = "|".join(sorted(sources_used)) if sources_used else "none"

    return {
        "ticker": ticker.upper(),
        "asof": asof_utc.isoformat(),
        "n": n,
        "low_sample": low_sample,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "bias": bias,
        "summary": summary,
        "matches": match_payload,
        "sources": {"bars": sources_str},
    }


__all__ = ["find_similar_days", "SimilarityScorer"]


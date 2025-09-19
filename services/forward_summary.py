from __future__ import annotations

import math
import statistics
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

from config import settings
from services.telemetry import log as log_telemetry
from utils import now_et

SUMMARY_CACHE_TTL_SECONDS = 60.0


_summary_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _normalize_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        num = float(value)
        if math.isnan(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _parse_timestamp(raw: Any, target_tz) -> datetime | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        stamp = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    stamp = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    if target_tz is not None:
        try:
            return stamp.astimezone(target_tz)
        except Exception:
            return stamp
    return stamp


def _normalize_weight(value: float) -> float | int:
    if not math.isfinite(value):
        return 0.0
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    filtered: list[tuple[float, float]] = [
        (float(val), float(weight))
        for val, weight in pairs
        if weight is not None and weight > 0
    ]
    if not filtered:
        return None
    filtered.sort(key=lambda item: item[0])
    total = sum(weight for _, weight in filtered)
    if not total or not math.isfinite(total):
        return None
    half = total / 2.0
    running = 0.0
    for value, weight in filtered:
        running += weight
        if running >= half:
            return value
    return filtered[-1][0]


def _sanitize_percent(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def _sanitize_optional(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _wilson_lb95(hits: float | int, count: float | int) -> float:
    try:
        from routes import _wilson_lb95 as _route_lb95  # type: ignore

        return float(_route_lb95(int(hits), int(count)))
    except Exception:
        # Fallback Wilson approximation (z=1.96) if import fails.
        n = max(1.0, float(count))
        p = max(0.0, min(1.0, float(hits) / n))
        z = 1.96
        denom = 1 + (z**2) / n
        center = p + (z**2) / (2 * n)
        margin = z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
        return max(0.0, (center - margin) / denom)


def compute_forward_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "n": 0,
        "hits": 0,
        "hit_rate": 0.0,
        "hit_lb95": 0.0,
        "avg_roi": None,
        "median_tt_bars": None,
        "avg_dd": None,
    }

    reference = now_et()
    recency_mode_raw = getattr(
        settings, "forward_recency_mode", getattr(settings, "FORWARD_RECENCY_MODE", "off")
    )
    recency_mode = str(recency_mode_raw or "off").strip().lower()
    if recency_mode not in {"off", "exp"}:
        recency_mode = "off"
    try:
        half_life = float(
            getattr(
                settings,
                "forward_recency_halflife_days",
                getattr(settings, "FORWARD_RECENCY_HALFLIFE_DAYS", 30.0),
            )
        )
    except (TypeError, ValueError):
        half_life = 30.0
    if half_life <= 0:
        half_life = 30.0

    completed: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        outcome_raw = row.get("outcome")
        if outcome_raw is None:
            continue
        outcome = str(outcome_raw).strip().lower()
        if not outcome:
            continue
        completed.append((row, outcome))

    def compute_unweighted() -> dict[str, Any]:
        metrics = dict(base)
        count = len(completed)
        metrics["n"] = count
        hits = sum(1 for _, outcome in completed if outcome == "hit")
        metrics["hits"] = hits
        if not count:
            return metrics
        metrics["hit_rate"] = _sanitize_percent(hits / count)
        metrics["hit_lb95"] = _sanitize_percent(_wilson_lb95(hits, count))

        roi_values: list[float] = []
        tt_values: list[float] = []
        dd_values: list[float] = []
        for run, _ in completed:
            roi_val = _coerce_float(run.get("roi"))
            if roi_val is not None:
                roi_values.append(roi_val)
            tt_val = _coerce_float(run.get("tt_bars"))
            if tt_val is not None:
                tt_values.append(tt_val)
            dd_val = _coerce_float(run.get("dd"))
            if dd_val is not None:
                dd_values.append(dd_val)
        if roi_values:
            try:
                metrics["avg_roi"] = float(statistics.mean(roi_values))
            except statistics.StatisticsError:
                metrics["avg_roi"] = None
        if tt_values:
            try:
                med = float(statistics.median(tt_values))
                metrics["median_tt_bars"] = int(med) if float(med).is_integer() else med
            except statistics.StatisticsError:
                metrics["median_tt_bars"] = None
        if dd_values:
            try:
                metrics["avg_dd"] = float(statistics.mean(dd_values))
            except statistics.StatisticsError:
                metrics["avg_dd"] = None
        return metrics

    def compute_weighted(unweighted: dict[str, Any]) -> dict[str, Any]:
        if recency_mode != "exp":
            return dict(unweighted)
        if not completed:
            return dict(base)
        metrics = dict(base)
        target_tz = reference.tzinfo
        half_life_days = half_life
        weights: list[float] = []
        roi_total = 0.0
        roi_observed = False
        dd_total = 0.0
        dd_observed = False
        tt_pairs: list[tuple[float, float]] = []

        for run, outcome in completed:
            weight = 1.0
            stamp = _parse_timestamp(run.get("exit_ts"), target_tz)
            if stamp is None:
                stamp = _parse_timestamp(run.get("entry_ts"), target_tz)
            if stamp is not None:
                delta = reference - stamp
                age_days = max(0.0, delta.total_seconds() / 86400.0)
                try:
                    weight = 0.5 ** (age_days / half_life_days)
                except Exception:
                    weight = 0.0
            weights.append(weight)
            roi_val = _coerce_float(run.get("roi"))
            if roi_val is not None:
                roi_total += weight * roi_val
                roi_observed = True
            tt_val = _coerce_float(run.get("tt_bars"))
            if tt_val is not None:
                tt_pairs.append((tt_val, weight))
            dd_val = _coerce_float(run.get("dd"))
            if dd_val is not None:
                dd_total += weight * dd_val
                dd_observed = True
            if outcome == "hit":
                continue
        weight_sum = float(sum(weights))
        metrics["n"] = _normalize_weight(weight_sum)
        hits_weighted = float(
            sum(weight for (run, outcome), weight in zip(completed, weights) if outcome == "hit")
        )
        metrics["hits"] = _normalize_weight(hits_weighted)
        if weight_sum <= 0 or not math.isfinite(weight_sum):
            return metrics
        metrics["hit_rate"] = _sanitize_percent(hits_weighted / weight_sum)
        n_eff = int(round(weight_sum))
        hits_eff = int(round(hits_weighted))
        metrics["hit_lb95"] = _sanitize_percent(_wilson_lb95(hits_eff, max(1, n_eff)))
        if roi_observed and weight_sum:
            metrics["avg_roi"] = roi_total / weight_sum
        if tt_pairs:
            med = _weighted_median(tt_pairs)
            if med is not None:
                metrics["median_tt_bars"] = int(med) if float(med).is_integer() else med
        if dd_observed and weight_sum:
            metrics["avg_dd"] = dd_total / weight_sum
        return metrics

    unweighted = compute_unweighted()
    weighted = compute_weighted(unweighted)

    # Sanitize optional outputs (avoid NaN propagation)
    for bucket in (unweighted, weighted):
        bucket["hit_rate"] = _sanitize_percent(bucket.get("hit_rate"))
        bucket["hit_lb95"] = _sanitize_percent(bucket.get("hit_lb95"))
        bucket["avg_roi"] = _sanitize_optional(bucket.get("avg_roi"))
        bucket["avg_dd"] = _sanitize_optional(bucket.get("avg_dd"))
        med = bucket.get("median_tt_bars")
        if isinstance(med, float) and not math.isfinite(med):
            bucket["median_tt_bars"] = None

    summary = dict(base)
    summary.update(unweighted)
    summary["mode"] = recency_mode
    summary["half_life_days"] = half_life
    summary["unweighted"] = dict(unweighted)
    summary["weighted"] = dict(weighted)
    return summary


def get_cached_forward_summary(
    favorite_id: Any, rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    key = _normalize_id(favorite_id)
    materialized_rows = list(rows)
    if key:
        entry = _summary_cache.get(key)
        now_ts = time.monotonic()
        if entry and entry[0] > now_ts:
            log_telemetry(
                {
                    "event": "forward_summary_cached",
                    "favorite_id": key,
                    "rows": len(materialized_rows),
                }
            )
            return deepcopy(entry[1])

    summary = compute_forward_summary(materialized_rows)
    if not key:
        return summary

    expires = time.monotonic() + SUMMARY_CACHE_TTL_SECONDS
    _summary_cache[key] = (expires, deepcopy(summary))
    log_telemetry(
        {
            "event": "forward_summary_refresh",
            "favorite_id": key,
            "rows": len(materialized_rows),
        }
    )
    return summary


def invalidate_forward_summary(favorite_id: Any) -> None:
    key = _normalize_id(favorite_id)
    if not key:
        return
    if key in _summary_cache:
        _summary_cache.pop(key, None)


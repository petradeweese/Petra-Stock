"""Utilities for running scheduled scan batches."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Mapping, Tuple

from db import DB_PATH, row_to_dict
from scanner import compute_scan_for_ticker, preload_prices

logger = logging.getLogger(__name__)


def _coerce(value: Any, caster, default: Any) -> Any:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            if not value.strip():
                return default
            return caster(value)
        return caster(value)
    except Exception:
        return default


def _favorite_to_params(fav: Mapping[str, Any]) -> Dict[str, Any]:
    params = {
        "interval": str(fav.get("interval") or "15m"),
        "direction": str(fav.get("direction") or "UP").upper(),
        "target_pct": _coerce(fav.get("target_pct"), float, 1.0),
        "stop_pct": _coerce(fav.get("stop_pct"), float, 0.5),
        "window_value": _coerce(fav.get("window_value"), float, 4.0),
        "window_unit": str(fav.get("window_unit") or "Hours"),
        "lookback_years": _coerce(fav.get("lookback_years"), float, 2.0),
        "max_tt_bars": _coerce(fav.get("max_tt_bars"), int, 12),
        "min_support": _coerce(fav.get("min_support"), int, 20),
        "delta_assumed": _coerce(fav.get("delta"), float, 0.40),
        "theta_per_day_pct": _coerce(fav.get("theta_day"), float, 0.20),
        "atrz_gate": _coerce(fav.get("atrz"), float, 0.10),
        "slope_gate_pct": _coerce(fav.get("slope"), float, 0.02),
        "use_regime": _coerce(fav.get("use_regime"), int, 0),
        "regime_trend_only": _coerce(fav.get("trend_only"), int, 0),
        "vix_z_max": _coerce(fav.get("vix_z_max"), float, 3.0),
        "slippage_bps": _coerce(fav.get("slippage_bps"), float, 7.0),
        "vega_scale": _coerce(fav.get("vega_scale"), float, 0.03),
    }
    cooldown = fav.get("cooldown_bars")
    if cooldown is not None:
        params["cooldown_bars"] = _coerce(cooldown, int, params["max_tt_bars"])
    return params


def _load_favorites() -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM favorites ORDER BY id DESC")
            return [row_to_dict(row, cur) for row in cur.fetchall()]
    except sqlite3.Error:
        logger.exception("failed to load favorites for autoscan")
        return []


def _preload(groups: Mapping[Tuple[str, float], List[str]]) -> None:
    for (interval, lookback), tickers in groups.items():
        if not tickers:
            continue
        try:
            preload_prices(tickers, interval, lookback)
        except Exception:
            logger.debug(
                "autoscan preload failed interval=%s tickers=%d",
                interval,
                len(tickers),
            )


def run_autoscan_batch() -> List[Dict[str, Any]]:
    """Execute the autoscan pipeline for all saved favorites."""

    favorites = _load_favorites()
    if not favorites:
        return []

    grouped: Dict[Tuple[str, float], List[str]] = {}
    tasks: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []
    for fav in favorites:
        ticker = str(fav.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        params = _favorite_to_params(fav)
        interval = params.get("interval", "15m")
        lookback = float(params.get("lookback_years", 2.0))
        grouped.setdefault((interval, lookback), []).append(ticker)
        tasks.append((fav, ticker, params))

    _preload(grouped)

    results: List[Dict[str, Any]] = []
    for _fav, ticker, params in tasks:
        try:
            row = compute_scan_for_ticker(ticker, params) or {}
        except Exception:
            logger.exception("autoscan compute failed symbol=%s", ticker)
            continue
        if isinstance(row, dict) and row:
            results.append(row)
    return results


__all__ = ["run_autoscan_batch"]

# ruff: noqa: E501
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from db import row_to_dict
from services.market_data import (
    expected_bar_count,
    fetch_prices,
    window_from_lookback,
)
from services.price_utils import DataUnavailableError

# Adapter to the original ROI engine
_real_scan_single: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None
logger = logging.getLogger(__name__)


def _install_real_engine_adapter():
    """
    Integrate with pattern_finder_app. We support:
      - scan_parallel_threaded(tickers, cfg, max_workers=None) -> DataFrame
      - scan_parallel(tickers, cfg, max_workers=None) -> DataFrame
      - analyze_roi_mode(...) -> (model, df_te, forward)
    We normalize to: {ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule}
    """
    global _real_scan_single
    try:
        import importlib

        mod = importlib.import_module("pattern_finder_app")

        fn = None
        mode = None
        if hasattr(mod, "scan_parallel_threaded"):
            fn = getattr(mod, "scan_parallel_threaded")
            mode = "threaded"
        elif hasattr(mod, "scan_parallel"):
            fn = getattr(mod, "scan_parallel")
            mode = "parallel"
        elif hasattr(mod, "analyze_roi_mode"):
            fn = getattr(mod, "analyze_roi_mode")
            mode = "single"
        else:
            logger.warning("pattern_finder_app found, but no known scan function.")
            _real_scan_single = None
            return

        def _row_to_dict(row: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            out = row_to_dict(row)

            def get(*keys, default=None):
                for k in keys:
                    if k in out and out[k] is not None:
                        return out[k]
                return default

            def fnum(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            def to_pct(x):
                x = fnum(x)
                # If looks like a fraction (<=1), convert to %
                if abs(x) <= 1.0:
                    return x * 100.0
                return x

            roi = get("avg_roi_pct", "avg_roi", default=None)
            hit = get("hit_pct", "hit_rate", default=None)
            dd = get("avg_dd_pct", "avg_dd", default=0.0)
            supp = get("support", "n", "count", default=0)
            tt = get("avg_tt", default=0.0)
            stab = get("stability", default=0.0)
            sharpe = get("sharpe", "sharpe_ratio", default=0.0)
            rule = get("rule", "rule_str", default="")
            direct = get("direction", default=params.get("direction", "UP"))
            tkr = get("ticker", default=params.get("ticker", "?"))

            roi_pct = to_pct(roi) if roi is not None else None
            hit_pct = to_pct(hit) if hit is not None else None
            dd_pct = to_pct(dd) if dd is not None else 0.0

            if roi_pct is None or hit_pct is None:
                return {}

            stab_pct = to_pct(stab)

            return {
                "ticker": str(tkr),
                "direction": str(direct).upper(),
                "avg_roi_pct": float(roi_pct),
                "hit_pct": float(hit_pct),
                "support": int(supp or 0),
                "avg_tt": fnum(tt),
                "avg_dd_pct": float(dd_pct),
                "stability": fnum(stab_pct),
                "sharpe": fnum(sharpe),
                "rule": str(rule or ""),
            }

        if mode in ("threaded", "parallel"):

            def wrapper(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    df = fn([ticker], params)
                    try:
                        import pandas as pd

                        if isinstance(df, pd.DataFrame) and not df.empty:
                            row = df.iloc[0].to_dict()
                            return _row_to_dict(row, params)
                    except Exception:
                        pass
                    if isinstance(df, list) and df:
                        return _row_to_dict(df[0], params)
                    if isinstance(df, dict) and df:
                        return _row_to_dict(df, params)
                    return {}
                except DataUnavailableError:
                    raise
                except Exception as e:
                    logger.error("scan_* error for %s: %r", ticker, e)
                    return {}

        else:

            def wrapper(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    model, df_te, _ = fn(
                        ticker=ticker,
                        interval=params.get("interval", "15m"),
                        direction=params.get("direction", "UP"),
                        target_pct=params.get("target_pct", 1.0),
                        stop_pct=params.get("stop_pct", 0.5),
                        window_value=params.get("window_value", 4.0),
                        window_unit=params.get("window_unit", "Hours"),
                        lookback_years=params.get("lookback_years", 2.0),
                        max_tt_bars=params.get("max_tt_bars", 12),
                        min_support=params.get("min_support", 20),
                        delta_assumed=params.get("delta_assumed", 0.40),
                        theta_per_day_pct=params.get("theta_per_day_pct", 0.20),
                        atrz_gate=params.get("atrz_gate", 0.10),
                        slope_gate_pct=params.get("slope_gate_pct", 0.02),
                        use_regime=bool(params.get("use_regime", 0)),
                        regime_trend_only=bool(params.get("regime_trend_only", 0)),
                        vix_z_max=params.get("vix_z_max", 3.0),
                        event_mask=None,
                        slippage_bps=params.get("slippage_bps", 7.0),
                        vega_scale=params.get("vega_scale", 0.03),
                        cooldown_bars=params.get("cooldown_bars"),
                    )
                    if df_te is None or getattr(df_te, "empty", True):
                        return {}
                    df_te = df_te.sort_values(
                        ["avg_roi", "hit_rate", "support", "stability"],
                        ascending=[False, False, False, False],
                    )
                    row = df_te.iloc[0].to_dict()
                    mapped = {
                        "ticker": ticker,
                        "direction": row.get(
                            "direction", params.get("direction", "UP")
                        ),
                        "avg_roi": row.get("avg_roi", 0.0),
                        "hit_rate": row.get("hit_rate", 0.0),
                        "support": row.get("support", 0),
                        "avg_tt": row.get("avg_tt", 0.0),
                        "avg_dd": row.get("avg_dd", 0.0),
                        "stability": row.get("stability", 0.0),
                        "rule": row.get("rule", ""),
                    }
                    return _row_to_dict(mapped, params)
                except DataUnavailableError:
                    raise
                except Exception as e:
                    logger.error("analyze_roi_mode error for %s: %r", ticker, e)
                    return {}

        _real_scan_single = wrapper
        logger.info("Using REAL engine from pattern_finder_app (%s).", mode)
    except Exception as e:
        logger.warning("pattern_finder_app not available or failed to import: %r", e)
        _real_scan_single = None


_PRICE_DATA: Dict[Tuple[str, str, float], pd.DataFrame] = {}


def _ensure_coverage(ticker: str, interval: str, lookback_years: float) -> bool:
    """Return ``True`` if price data coverage is >=95% for the window."""
    start, end = window_from_lookback(lookback_years)
    key = (ticker, interval, lookback_years)
    df = _PRICE_DATA.get(key)
    if df is None:
        df = fetch_prices([ticker], interval, lookback_years).get(
            ticker, pd.DataFrame()
        )
        _PRICE_DATA[key] = df

    expected = expected_bar_count(start, end, interval)
    bars = len(df) if df is not None else 0
    coverage = bars / expected if expected else 0.0
    if bars > 0 and coverage >= 0.95:
        return True

    df = fetch_prices([ticker], interval, lookback_years).get(ticker, pd.DataFrame())
    _PRICE_DATA[key] = df
    bars = len(df) if df is not None else 0
    expected = expected_bar_count(start, end, interval)
    coverage = bars / expected if expected else 0.0
    if bars > 0 and coverage >= 0.95:
        return True

    logger.info(
        "skip_no_data symbol=%s window=%s:%s bars=%d coverage=%.2f",
        ticker,
        start.date(),
        end.date(),
        bars,
        coverage,
    )
    raise DataUnavailableError(f"{ticker} coverage {coverage:.2%} <95%")


def preload_prices(tickers: List[str], interval: str, lookback_years: float) -> None:
    """Preload price data for a batch of tickers."""
    try:
        fetched = fetch_prices(tickers, interval, lookback_years)
        for t, df in fetched.items():
            _PRICE_DATA[(t, interval, lookback_years)] = df
    except Exception as e:  # pragma: no cover - network failures
        logger.error("prefetch failed: %r", e)


_install_real_engine_adapter()

try:
    import pattern_finder_app as _pfa
except Exception:
    _pfa = None

if _pfa is not None:

    def _price_lookup(
        ticker: str, interval: str, lookback_years: float
    ) -> pd.DataFrame:
        key = (ticker, interval, lookback_years)
        df = _PRICE_DATA.get(key)
        if df is not None and not df.empty:
            return df.copy()
        data = fetch_prices([ticker], interval, lookback_years).get(ticker)
        if data is None:
            data = pd.DataFrame()
        _PRICE_DATA[key] = data
        return data.copy()

    _pfa._download_prices = _price_lookup


def _desktop_like_single(ticker: str, params: dict) -> dict:
    """Match pattern_finder_app._scan_worker for a SINGLE ticker+direction."""
    if _pfa is None:
        logger.warning("pattern_finder_app is not available")
        return {}
    try:
        # Pull parameters into locals once so the tight loops in the engine
        # don't need repeated dict lookups. Missing values fall back to the
        # same defaults used by the desktop scanner.

        def _get_str(key: str, default: str) -> str:
            value = params.get(key, default)
            if value in (None, ""):
                return default
            return str(value)

        def _get_float(key: str, default: float) -> float:
            value = params.get(key, default)
            try:
                if value in (None, ""):
                    return float(default)
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _get_int(key: str, default: int) -> int:
            value = params.get(key, default)
            try:
                if value in (None, ""):
                    return int(default)
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        interval = _get_str("interval", "15m")
        direction = _get_str("direction", "UP").upper()
        target_pct = _get_float("target_pct", 1.0)
        stop_pct = _get_float("stop_pct", 0.5)
        window_value = _get_float("window_value", 4.0)
        window_unit = _get_str("window_unit", "Hours")
        lookback_years = _get_float("lookback_years", 2.0)
        max_tt_bars = _get_int("max_tt_bars", 12)
        min_support = _get_int("min_support", 20)
        delta_assumed = _get_float("delta_assumed", 0.40)
        theta_per_day_pct = _get_float("theta_per_day_pct", 0.20)
        atrz_gate = _get_float("atrz_gate", 0.10)
        slope_gate_pct = _get_float("slope_gate_pct", 0.02)
        use_regime = bool(_get_int("use_regime", 0))
        regime_trend_only = bool(_get_int("regime_trend_only", 0))
        vix_z_max = _get_float("vix_z_max", 3.0)
        slippage_bps = _get_float("slippage_bps", 7.0)
        vega_scale = _get_float("vega_scale", 0.03)
        scan_min_hit = _get_float("scan_min_hit", 50.0)
        scan_max_dd = _get_float("scan_max_dd", 50.0)

        try:
            cooldown_default = int(_pfa._bars_for_window(window_value, window_unit, interval))
        except Exception:
            cooldown_default = max_tt_bars
        cooldown_param = params.get("cooldown_bars", cooldown_default)
        try:
            cooldown_bars = int(round(float(cooldown_param)))
        except (TypeError, ValueError):
            cooldown_bars = cooldown_default
        if cooldown_bars < 0:
            cooldown_bars = cooldown_default

        model, df, _ = _pfa.analyze_roi_mode(
            ticker=ticker,
            interval=interval,
            direction=direction,
            target_pct=target_pct,
            stop_pct=stop_pct,
            window_value=window_value,
            window_unit=window_unit,
            lookback_years=lookback_years,
            max_tt_bars=max_tt_bars,
            min_support=min_support,
            delta_assumed=delta_assumed,
            theta_per_day_pct=theta_per_day_pct,
            atrz_gate=atrz_gate,
            slope_gate_pct=slope_gate_pct,
            use_regime=use_regime,
            regime_trend_only=regime_trend_only,
            vix_z_max=vix_z_max,
            event_mask=None,
            slippage_bps=slippage_bps,
            vega_scale=vega_scale,
            cooldown_bars=cooldown_bars,
        )
        if df is None or df.empty:
            return {}
        df = df[
            (df["hit_rate"] * 100.0 >= scan_min_hit)
            & (df["avg_dd"] * 100.0 <= scan_max_dd)
        ]
        if df.empty:
            return {}
        r = df.sort_values(
            ["sharpe", "avg_roi", "hit_rate", "support", "stability"],
            ascending=[False, False, False, False, False],
        ).iloc[0]

        def _coerce_float(value: Any, default: float = 0.0) -> float:
            try:
                if value is None or pd.isna(value):
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        def _compute_confidence() -> tuple[int, str]:
            try:
                if hasattr(_pfa, "_conf_v1"):
                    score, label = _pfa._conf_v1(
                        r.get("hit_lb95"), r.get("avg_roi"), r.get("support")
                    )
                    try:
                        score_int = int(round(float(score)))
                    except Exception:
                        score_int = 0
                    return max(score_int, 0), str(label or "")
            except Exception:
                pass
            return 0, ""

        stab = _coerce_float(r.get("stability", 0.0))
        stab_pct = stab * 100.0 if abs(stab) <= 1.0 else stab
        hit_lb95 = _coerce_float(r.get("hit_lb95", 0.0))
        stop_fraction = _coerce_float(r.get("stop_pct", 0.0))
        timeout_fraction = _coerce_float(r.get("timeout_pct", 0.0))

        confidence_raw = r.get("confidence")
        if confidence_raw is None or pd.isna(confidence_raw):
            confidence_score, confidence_label = _compute_confidence()
        else:
            try:
                confidence_score = int(round(float(confidence_raw)))
            except (TypeError, ValueError):
                confidence_score, confidence_label = _compute_confidence()
            else:
                confidence_label = str(r.get("confidence_label") or "")
                if not confidence_label:
                    _, confidence_label = _compute_confidence()

        recent3 = r.get("recent3")
        if isinstance(recent3, float) and pd.isna(recent3):
            recent3 = []
        if recent3 is None:
            recent3 = []
        elif not isinstance(recent3, list):
            try:
                recent3 = list(recent3)
            except Exception:
                recent3 = []

        return {
            "ticker": ticker,
            "direction": r.get("direction", direction),
            "avg_roi_pct": float(r["avg_roi"]) * 100.0,
            "hit_pct": float(r["hit_rate"]) * 100.0,
            "support": int(r["support"]),
            "avg_tt": float(r["avg_tt"]) if pd.notna(r["avg_tt"]) else 0.0,
            "avg_dd_pct": float(r["avg_dd"]) * 100.0,
            "stability": stab_pct,
            "sharpe": float(r.get("sharpe", 0.0)),
            "rule": str(r["rule"]),
            "hit_lb95": hit_lb95,
            "stop_pct": stop_fraction,
            "timeout_pct": timeout_fraction,
            "confidence": confidence_score,
            "confidence_label": confidence_label,
            "recent3": recent3,
        }
    except Exception:
        logger.exception("scan computation failed for %s", ticker)
        return {}


def compute_scan_for_ticker(
    ticker: str, params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Final override: delegate to _desktop_like_single so the web API matches the desktop scanner.
    Handles BOTH by evaluating UP and DOWN and returning the better-scoring row.
    Returns ``None`` when price data is unavailable.
    """
    try:
        dirn = str(params.get("direction", "UP")).upper()
    except Exception:
        dirn = "UP"

    interval = params.get("interval", "15m")
    lookback_years = float(params.get("lookback_years", 2.0))
    try:
        if not _ensure_coverage(ticker, interval, lookback_years):
            return None
    except DataUnavailableError as e:
        logger.info("skip_no_data symbol=%s reason=%s", ticker, e)
        return None

    try:
        if dirn == "BOTH":
            a = dict(params)
            a["direction"] = "UP"
            b = dict(params)
            b["direction"] = "DOWN"
            ra = _desktop_like_single(ticker, a)
            rb = _desktop_like_single(ticker, b)
            picks = [r for r in (ra, rb) if isinstance(r, dict) and r]
            if not picks:
                return {}
            return sorted(
                picks,
                key=lambda r: (
                    r.get("sharpe", 0.0),
                    r.get("avg_roi_pct", 0.0),
                    r.get("hit_pct", 0.0),
                    r.get("support", 0),
                    r.get("stability", 0.0),
                ),
                reverse=True,
            )[0]
        else:
            return _desktop_like_single(ticker, params)
    except DataUnavailableError:
        start, end = window_from_lookback(lookback_years)
        key = (ticker, interval, lookback_years)
        df = _PRICE_DATA.get(key)
        bars = len(df) if df is not None else 0
        logger.info(
            "skip_no_data symbol=%s window=%s:%s bars=%d",
            ticker,
            start.date(),
            end.date(),
            bars,
        )
        return None

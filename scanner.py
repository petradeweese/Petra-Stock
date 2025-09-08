from typing import Dict, Any, Optional, Callable

# Adapter to the original ROI engine
_real_scan_single: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None


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
            print("[adapter] pattern_finder_app found, but no known scan function.")
            _real_scan_single = None
            return

        def _row_to_dict(row: dict, params: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)

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
            rule = get("rule", "rule_str", default="")
            direct = get("direction", default=params.get("direction", "UP"))
            tkr = get("ticker", default=params.get("ticker", "?"))

            roi_pct = to_pct(roi) if roi is not None else None
            hit_pct = to_pct(hit) if hit is not None else None
            dd_pct = to_pct(dd) if dd is not None else 0.0

            if roi_pct is None or hit_pct is None:
                return {}

            return {
                "ticker": str(tkr),
                "direction": str(direct).upper(),
                "avg_roi_pct": float(roi_pct),
                "hit_pct": float(hit_pct),
                "support": int(supp or 0),
                "avg_tt": fnum(tt),
                "avg_dd_pct": float(dd_pct),
                "stability": fnum(stab),
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
                except Exception as e:
                    print(f"[adapter] scan_* error for {ticker}: {e!r}")
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
                        "direction": row.get("direction", params.get("direction", "UP")),
                        "avg_roi": row.get("avg_roi", 0.0),
                        "hit_rate": row.get("hit_rate", 0.0),
                        "support": row.get("support", 0),
                        "avg_tt": row.get("avg_tt", 0.0),
                        "avg_dd": row.get("avg_dd", 0.0),
                        "stability": row.get("stability", 0.0),
                        "rule": row.get("rule", ""),
                    }
                    return _row_to_dict(mapped, params)
                except Exception as e:
                    print(f"[adapter] analyze_roi_mode error for {ticker}: {e!r}")
                    return {}

        _real_scan_single = wrapper
        print(f"[adapter] Using REAL engine from pattern_finder_app ({mode}).")
    except Exception as e:
        print("[adapter] pattern_finder_app not available or failed to import:", repr(e))
        _real_scan_single = None


_install_real_engine_adapter()

try:
    import pattern_finder_app as _pfa
    import pandas as pd
except Exception:
    _pfa = None
    import pandas as pd  # ensure pd is available for type hints


def _desktop_like_single(ticker: str, params: dict) -> dict:
    """Match pattern_finder_app._scan_worker for a SINGLE ticker+direction."""
    if _pfa is None:
        return {}
    try:
        px = _pfa._download_prices(ticker, params["interval"], params["lookback_years"])
        ev = _pfa.build_event_mask(px.index, set())
        model, df, _ = _pfa.analyze_roi_mode(
            ticker=ticker,
            interval=params["interval"],
            direction=params["direction"],
            target_pct=params["target_pct"],
            stop_pct=params["stop_pct"],
            window_value=params["window_value"],
            window_unit=params["window_unit"],
            lookback_years=params["lookback_years"],
            max_tt_bars=params["max_tt_bars"],
            min_support=params["min_support"],
            delta_assumed=params["delta_assumed"],
            theta_per_day_pct=params["theta_per_day_pct"],
            atrz_gate=params["atrz_gate"],
            slope_gate_pct=params["slope_gate_pct"],
            use_regime=params["use_regime"],
            regime_trend_only=params["regime_trend_only"],
            vix_z_max=params["vix_z_max"],
            event_mask=ev,
            slippage_bps=params["slippage_bps"],
            vega_scale=params["vega_scale"],
        )
        if df is None or df.empty:
            return {}
        df = df[(df["hit_rate"] * 100.0 >= params["scan_min_hit"]) &
                (df["avg_dd"] * 100.0 <= params["scan_max_dd"])]
        if df.empty:
            return {}
        r = df.sort_values(
            ["avg_roi", "hit_rate", "support", "stability"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return {
            "ticker": ticker,
            "direction": r.get("direction", params["direction"]),
            "avg_roi_pct": float(r["avg_roi"]) * 100.0,
            "hit_pct": float(r["hit_rate"]) * 100.0,
            "support": int(r["support"]),
            "avg_tt": float(r["avg_tt"]) if pd.notna(r["avg_tt"]) else 0.0,
            "avg_dd_pct": float(r["avg_dd"]) * 100.0,
            "stability": float(r.get("stability", 0.0)),
            "rule": str(r["rule"]),
        }
    except Exception:
        return {}


def compute_scan_for_ticker(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final override: delegate to _desktop_like_single so the web API matches the desktop scanner.
    Handles BOTH by evaluating UP and DOWN and returning the better-scoring row.
    """
    try:
        dirn = str(params.get("direction", "UP")).upper()
    except Exception:
        dirn = "UP"

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
                r.get("avg_roi_pct", 0.0),
                r.get("hit_pct", 0.0),
                r.get("support", 0),
                r.get("stability", 0.0),
            ),
            reverse=True,
        )[0]
    else:
        return _desktop_like_single(ticker, params)

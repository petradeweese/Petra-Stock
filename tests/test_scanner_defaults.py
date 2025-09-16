from types import SimpleNamespace

import pandas as pd

import scanner


def test_desktop_like_single_uses_defaults(monkeypatch):
    calls = {}

    def fake_analyze_roi_mode(**kwargs):
        calls.update(kwargs)
        df = pd.DataFrame(
            [
                {
                    "avg_roi": 0.8,
                    "hit_rate": 0.6,
                    "support": 25,
                    "avg_tt": 5.0,
                    "avg_dd": 0.4,
                    "stability": 0.1,
                    "rule": "demo",
                    "direction": kwargs.get("direction", "UP"),
                    "sharpe": 1.5,
                }
            ]
        )
        return object(), df, None

    monkeypatch.setattr(
        scanner,
        "_pfa",
        SimpleNamespace(analyze_roi_mode=fake_analyze_roi_mode),
    )

    result = scanner._desktop_like_single("AAPL", {})

    assert result == {
        "ticker": "AAPL",
        "direction": "UP",
        "avg_roi_pct": 80.0,
        "hit_pct": 60.0,
        "support": 25,
        "avg_tt": 5.0,
        "avg_dd_pct": 40.0,
        "stability": 10.0,
        "sharpe": 1.5,
        "rule": "demo",
    }

    assert calls == {
        "ticker": "AAPL",
        "interval": "15m",
        "direction": "UP",
        "target_pct": 1.0,
        "stop_pct": 0.5,
        "window_value": 4.0,
        "window_unit": "Hours",
        "lookback_years": 2.0,
        "max_tt_bars": 12,
        "min_support": 20,
        "delta_assumed": 0.40,
        "theta_per_day_pct": 0.20,
        "atrz_gate": 0.10,
        "slope_gate_pct": 0.02,
        "use_regime": False,
        "regime_trend_only": False,
        "vix_z_max": 3.0,
        "event_mask": None,
        "slippage_bps": 7.0,
        "vega_scale": 0.03,
    }

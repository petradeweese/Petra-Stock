import pandas as pd

from routes import check_alert_filters

def _base_option():
    return {"open_interest": 1000, "volume": 200, "bid": 1.0, "ask": 1.05}

def _base_price_sma():
    return (10.0, 9.0)

def test_liquidity_flags():
    allowed, flags = check_alert_filters(
        "AAA",
        "UP",
        get_adv=lambda t: 500_000,
        get_option=lambda t: {"open_interest": 400, "volume": 80, "bid": 1.0, "ask": 1.3},
        get_price_sma=_base_price_sma,
        get_earnings=lambda t: None,
    )
    assert not allowed
    assert {"low_adv", "low_option_oi", "low_option_volume", "wide_spread"}.issubset(set(flags))

def test_trend_confirmation():
    allowed, flags = check_alert_filters(
        "AAA",
        "UP",
        get_adv=lambda t: 2_000_000,
        get_option=_base_option,
        get_price_sma=lambda t: (9.0, 10.0),
        get_earnings=lambda t: None,
    )
    assert not allowed
    assert "trend" in flags

def test_earnings_blackout(monkeypatch):
    today = pd.Timestamp("2024-01-02", tz="America/New_York")
    monkeypatch.setattr("routes.now_et", lambda: today)
    allowed, flags = check_alert_filters(
        "AAA",
        "UP",
        get_adv=lambda t: 2_000_000,
        get_option=_base_option,
        get_price_sma=_base_price_sma,
        get_earnings=lambda t: today + pd.Timedelta(days=5),
    )
    assert not allowed
    assert "earnings" in flags


def test_zero_mid_price(monkeypatch):
    allowed, flags = check_alert_filters(
        "AAA",
        "UP",
        get_adv=lambda t: 2_000_000,
        get_option=lambda t: {
            "open_interest": 1000,
            "volume": 200,
            "bid": 1.0,
            "ask": 1.2,
            "mid": 0.0,
        },
        get_price_sma=_base_price_sma,
        get_earnings=lambda t: None,
    )
    assert not allowed
    assert "wide_spread" in flags

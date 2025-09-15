from datetime import date, timedelta

from services.favorites_alerts import evaluate_contract
from services.options_provider import OptionContract


def make_contract(**kw):
    today = date(2024, 1, 1)
    defaults = dict(
        occ="AAPL",
        side="call",
        strike=100.0,
        expiry=today + timedelta(days=10),
        bid=1.0,
        ask=2.0,
        mid=1.5,
        last=1.5,
        open_interest=300,
        volume=100,
        delta=0.5,
        gamma=0.1,
        theta=-0.05,
        vega=0.2,
        spread_pct=4.0,
        dte=10,
        iv_rank=50.0,
    )
    defaults.update(kw)
    return OptionContract(**defaults)


def test_evaluator_pass_and_fail():
    contract = make_contract(delta=0.9, open_interest=50, iv_rank=90)
    profile = {
        "delta_max": 0.7,
        "delta_min": 0.3,
        "min_open_interest": 200,
        "iv_rank_max": 80,
        "dte_min": 5,
        "dte_max": 30,
    }
    checks = evaluate_contract(contract, profile)
    out = {c.name: c for c in checks}
    assert not out["Delta"].passed and "Delta too high" in out["Delta"].explanation
    assert not out["Open Interest"].passed and "Open interest too low" in out["Open Interest"].explanation
    assert not out["IV Rank"].passed and "IV Rank high" in out["IV Rank"].explanation
    assert out["DTE"].passed

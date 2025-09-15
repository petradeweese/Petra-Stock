from datetime import date, timedelta

from services.favorites_alerts import select_contract
from services.options_provider import OptionContract


def _contract(**kw):
    today = date(2024, 1, 1)
    defaults = dict(
        occ="AAPL2401C100",
        side="call",
        strike=100.0,
        expiry=today + timedelta(days=10),
        bid=1.0,
        ask=2.0,
        mid=1.5,
        last=1.5,
        open_interest=200,
        volume=100,
        delta=0.5,
        gamma=0.1,
        theta=-0.05,
        vega=0.2,
        spread_pct=5.0,
        dte=10,
        iv_rank=50.0,
    )
    defaults.update(kw)
    return OptionContract(**defaults)


def test_selector_tie_breakers(monkeypatch):
    chain = [
        _contract(occ="c1", delta=0.52, spread_pct=5, volume=100),
        _contract(occ="c2", delta=0.48, spread_pct=3, volume=80),
        _contract(occ="c3", delta=0.52, spread_pct=3, volume=70),
    ]
    monkeypatch.setattr("services.options_provider.get_chain", lambda t: chain)
    profile = {
        "target_delta": 0.5,
        "dte_min": 5,
        "dte_max": 20,
        "min_open_interest": 100,
        "min_volume": 50,
        "max_spread_pct": 6,
    }
    res = select_contract("AAPL", "call", profile)
    assert res.contract.occ == "c2"  # lower spread then higher volume


def test_selector_no_liquid(monkeypatch):
    chain = [
        _contract(occ="c1", open_interest=10, volume=10, spread_pct=10),
        _contract(occ="c2", open_interest=20, volume=5, spread_pct=12),
        _contract(occ="c3", open_interest=30, volume=1, spread_pct=15),
    ]
    monkeypatch.setattr("services.options_provider.get_chain", lambda t: chain)
    profile = {
        "target_delta": 0.5,
        "dte_min": 5,
        "dte_max": 20,
        "min_open_interest": 100,
        "min_volume": 50,
        "max_spread_pct": 6,
    }
    res = select_contract("AAPL", "call", profile)
    assert res.contract is None
    assert [c.occ for c in res.alternatives] == ["c1", "c2"]
    assert res.note == "no liquid match; best alternatives shown"

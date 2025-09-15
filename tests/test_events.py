from datetime import date, timedelta

from services.favorites_alerts import select_contract
from services.options_provider import OptionContract


def contract(expiry_days, occ):
    today = date(2024, 1, 1)
    return OptionContract(
        occ=occ,
        side="call",
        strike=100.0,
        expiry=today + timedelta(days=expiry_days),
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
        dte=expiry_days,
        iv_rank=50.0,
    )


def test_event_avoidance(monkeypatch):
    chain = [contract(5, "c1"), contract(10, "c2")]
    monkeypatch.setattr("services.options_provider.get_chain", lambda t: chain)
    event_date = date(2024, 1, 3)  # within 7d of c1 expiry only
    monkeypatch.setattr(
        "services.events_provider.next_events",
        lambda t: [{"type": "earnings", "date": event_date.isoformat()}],
    )
    profile = {
        "target_delta": 0.5,
        "dte_min": 1,
        "dte_max": 30,
        "min_open_interest": 100,
        "min_volume": 50,
        "max_spread_pct": 6,
        "avoid_event_days": 7,
    }
    res = select_contract("AAPL", "call", profile)
    assert res.contract.occ == "c2"  # first contract skipped due to event
    assert res.event_note and "avoided" in res.event_note

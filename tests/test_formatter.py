from datetime import date, timedelta

from services.favorites_alerts import FavoriteHitStub, format_mms, Check
from services.options_provider import OptionContract


def sample_contract():
    today = date(2024, 1, 1)
    return OptionContract(
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


def test_formatter_verbose_vs_compact():
    hit = FavoriteHitStub(ticker="AAPL", direction="UP", pattern="Test")
    contract = sample_contract()
    checks = [
        Check("Delta", "Δ", 0.5, True),
        Check("Open Interest", "OI", 100, False, "Open interest too low"),
    ]
    profile_verbose = {"compact_mms": False, "include_symbols_in_alerts": True}
    body_verbose = format_mms(hit, contract, checks, profile_verbose)
    assert "Δ" in body_verbose and "OI" in body_verbose
    assert "Open interest too low" in body_verbose

    profile_compact = {"compact_mms": True, "include_symbols_in_alerts": True}
    body_compact = format_mms(hit, contract, checks, profile_compact)
    assert "Delta" not in body_compact  # passed check removed
    assert "OI" in body_compact and "Open interest too low" in body_compact

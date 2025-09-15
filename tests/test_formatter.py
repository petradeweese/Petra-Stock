from datetime import date, timedelta

from services.favorites_alerts import format_favorites_alert, Check
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


def test_formatter_mms_verbose_vs_compact():
    contract = sample_contract()
    checks = [
        Check("Delta", "Δ", 0.5, True, "balanced delta."),
        Check(
            "Open Interest",
            "OI",
            100,
            False,
            "Open interest too low — Not enough contracts.",
        ),
    ]
    targets = {"target": 185.0, "stop": 194.0, "hit": 70, "roi": 12, "dd": 8}
    body_verbose = format_favorites_alert(
        "AAPL",
        "UP",
        contract,
        checks,
        targets,
        compact=False,
        channel="mms",
        pattern="Test",
    )
    assert "Greeks & IV:" in body_verbose
    assert "• Delta (0.5)" in body_verbose
    assert "balanced delta" in body_verbose
    assert "Open interest too low" in body_verbose
    assert "Targets: 185" in body_verbose

    body_compact = format_favorites_alert(
        "AAPL",
        "UP",
        contract,
        checks,
        targets,
        compact=True,
        channel="mms",
        pattern="Test",
    )
    assert "• Delta" not in body_compact
    assert "Open interest too low" in body_compact
    assert "Feedback:" in body_compact
    assert "Open interest too low" in body_compact


def test_formatter_email_keeps_summary():
    contract = sample_contract()
    checks = [
        Check("Delta", "Δ", 0.5, True),
        Check("Open Interest", "OI", 100, False, "Open interest too low"),
    ]
    body_email = format_favorites_alert(
        "AAPL",
        "UP",
        contract,
        checks,
        {"target": None},
        compact=False,
        channel="email",
        pattern="Test",
    )
    assert "AAPL UP Test" in body_email
    assert "Contract AAPL" in body_email
    assert "Delta (Δ):" in body_email
    assert "Why this contract" in body_email

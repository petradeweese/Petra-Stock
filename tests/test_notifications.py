import pytest

from services.notifications import AlertDetails, build_recipient, format_mms, format_sms


def test_build_recipient_supported_carriers_sms_and_mms():
    number = "5551234567"
    # Mint Mobile (uses tmomail.net for both SMS and MMS)
    assert build_recipient(number, "mint") == f"{number}@tmomail.net"
    assert build_recipient(number, "mint", mms=True) == f"{number}@tmomail.net"
    # AT&T (separate domains for SMS and MMS)
    assert build_recipient(number, "att") == f"{number}@txt.att.net"
    assert build_recipient(number, "att", mms=True) == f"{number}@mms.att.net"
    # Verizon fallback
    assert build_recipient(number, "verizon") == f"{number}@vtext.com"
    assert build_recipient(number, "verizon", mms=True) == f"{number}@vzwpix.com"


def test_build_recipient_custom_domain():
    assert (
        build_recipient("5551234567", "custom", custom_domain="example.com")
        == "5551234567@example.com"
    )
    with pytest.raises(ValueError):
        build_recipient("555", "custom")


def test_format_sms_call_and_put():
    call = AlertDetails(
        ticker="AAPL",
        direction="UP",
        hit=188.20,
        target=191.5,
        stop=186.8,
        expiry="09/19",
        strike="190C",
        hit_pct=87,
    )
    msg = format_sms(call)
    assert msg == "AAPL UP hit 188.20 | T:191.5 S:186.8 | Exp 09/19 190C | Hit%:87"
    assert len(msg) <= 160

    put = AlertDetails(
        ticker="MSFT",
        direction="DOWN",
        hit=328.40,
        target=320.0,
        stop=332.5,
        expiry="09/19",
        strike="325P",
        hit_pct=72,
    )
    msg = format_sms(put)
    assert msg == "MSFT DOWN hit 328.40 | T:320.0 S:332.5 | Exp 09/19 325P | Hit%:72"
    assert len(msg) <= 160


def test_format_mms_with_support():
    details = AlertDetails(
        ticker="AAPL",
        direction="UP",
        hit=188.20,
        target=191.5,
        stop=186.8,
        expiry="09/19",
        strike="190C",
        hit_pct=87,
        support=12,
    )
    subject, body = format_mms(details)
    assert subject == "Pattern Alert: AAPL UP"
    assert body == (
        "AAPL UP hit 188.20\n"
        "Target 191.5 | Stop 186.8\n"
        "Expiry 09/19 190C\n"
        "Hit% 87 | Support 12"
    )


def test_format_mms_put_without_support():
    details = AlertDetails(
        ticker="MSFT",
        direction="DOWN",
        hit=328.40,
        target=320.0,
        stop=332.5,
        expiry="09/19",
        strike="325P",
        hit_pct=72,
    )
    subject, body = format_mms(details)
    assert subject == "Pattern Alert: MSFT DOWN"
    assert body == (
        "MSFT DOWN hit 328.40\n"
        "Target 320.0 | Stop 332.5\n"
        "Expiry 09/19 325P\n"
        "Hit% 72"
    )


def test_format_sms_raises_when_over_limit():
    details = AlertDetails(
        ticker="TOOLONG",
        direction="UP",
        hit=1.0,
        target=2.0,
        stop=0.5,
        expiry="09/19",
        strike="1C",
        hit_pct=100,
    )
    details.ticker = "A" * 200  # make the message exceed 160 chars
    with pytest.raises(ValueError):
        format_sms(details)

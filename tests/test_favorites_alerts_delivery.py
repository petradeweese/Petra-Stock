from datetime import datetime, timezone

import pytest

from services import favorites_alerts


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(favorites_alerts, "_SENT_ALERTS", {})
    monkeypatch.setattr(favorites_alerts, "_SQLITE_CONN", None)
    monkeypatch.setattr(favorites_alerts, "_SQLITE_PATH", tmp_path / "alerts.sqlite")
    monkeypatch.setattr(favorites_alerts, "_REDIS_CLIENT", None)
    monkeypatch.setattr(favorites_alerts, "_REDIS_READY", None)


@pytest.mark.parametrize(
    "channel,outcome",
    [
        ("email", "hit"),
        ("email", "all"),
        ("mms", "hit"),
        ("mms", "all"),
        ("sms", "hit"),
        ("sms", "all"),
    ],
)
def test_deliver_preview_matrix(monkeypatch, channel, outcome):
    email_calls = []
    sms_calls = []

    def fake_email(host, port, user, password, mail_from, to, subject, body, *, context=None):
        email_calls.append((subject, body, tuple(to), context))
        return {"ok": True, "provider": "smtp", "message_id": "<id>"}

    monkeypatch.setattr(favorites_alerts, "send_email_smtp", fake_email)
    monkeypatch.setattr(favorites_alerts.twilio_client, "is_enabled", lambda: True)
    monkeypatch.setattr(
        favorites_alerts.twilio_client,
        "send_mms",
        lambda number, body, *, context=None: sms_calls.append((number, body, context)) or True,
    )
    monkeypatch.setattr(
        favorites_alerts.sms_consent,
        "active_destinations",
        lambda: [{"phone_e164": "+18005550001", "user_id": "user-1"}],
    )
    monkeypatch.setattr(
        favorites_alerts.sms_consent,
        "allow_sending",
        lambda number: (True, {"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        favorites_alerts.sms_consent,
        "record_delivery",
        lambda *args, **kwargs: None,
    )

    subject = "Preview"
    bodies = {"email": "Email", "mms": "MMS", "sms": "SMS"}
    bar_time = datetime(2024, 1, 2, 14, 45, tzinfo=timezone.utc).isoformat()
    dedupe_key = f"fav1|15m|{bar_time}|{channel}|{outcome}"

    recipients = ["alerts@example.com"] if channel == "email" else None
    smtp_config = {
        "host": "smtp.gmail.com",
        "port": 587,
        "user": "alerts@example.com",
        "password": "app-pass",
        "mail_from": "Alerts <alerts@example.com>",
    }

    response = favorites_alerts.deliver_preview_alert(
        subject,
        bodies,
        channel=channel,
        favorite_id="fav1",
        bar_time=bar_time,
        interval="15m",
        dedupe_key=dedupe_key,
        simulated=True,
        outcome=outcome,
        symbol="AAPL",
        direction="UP",
        recipients=recipients,
        smtp_config=smtp_config,
    )

    assert response["ok"] is True
    assert response["channel"] == (channel if channel in {"email", "sms"} else "mms")
    assert response["reason"] == "sent"

    if channel == "email":
        assert email_calls, "email send expected"
        assert sms_calls == []
    else:
        assert sms_calls, "twilio send expected"

    throttled = favorites_alerts.deliver_preview_alert(
        subject,
        bodies,
        channel=channel,
        favorite_id="fav1",
        bar_time=bar_time,
        interval="15m",
        dedupe_key=dedupe_key,
        simulated=True,
        outcome=outcome,
        symbol="AAPL",
        direction="UP",
        recipients=recipients,
        smtp_config=smtp_config,
    )

    assert throttled["ok"] is False
    assert throttled["reason"] == "throttled"


def test_deliver_preview_fallback_to_email(monkeypatch):
    email_calls = []

    def fake_email(host, port, user, password, mail_from, to, subject, body, *, context=None):
        email_calls.append((subject, body, tuple(to), context))
        return {"ok": True, "provider": "smtp", "message_id": "<fallback>"}

    monkeypatch.setattr(favorites_alerts, "send_email_smtp", fake_email)
    monkeypatch.setattr(favorites_alerts.twilio_client, "is_enabled", lambda: False)

    subject = "Preview"
    bodies = {"email": "Email", "mms": "MMS", "sms": "SMS"}
    bar_time = datetime(2024, 1, 2, 14, 45, tzinfo=timezone.utc).isoformat()

    smtp_config = {
        "host": "smtp.gmail.com",
        "port": 587,
        "user": "alerts@example.com",
        "password": "app-pass",
        "mail_from": "Alerts <alerts@example.com>",
    }

    response = favorites_alerts.deliver_preview_alert(
        subject,
        bodies,
        channel="mms",
        favorite_id="fav2",
        bar_time=bar_time,
        interval="15m",
        dedupe_key="fav2|15m|fallback",
        simulated=True,
        outcome="hit",
        symbol="MSFT",
        direction="DOWN",
        recipients=["alerts@example.com"],
        smtp_config=smtp_config,
    )

    assert response["ok"] is True
    assert response["channel"] == "email"
    assert response["reason"] == "sent"
    assert email_calls and "[Sent via Email" in email_calls[0][1]

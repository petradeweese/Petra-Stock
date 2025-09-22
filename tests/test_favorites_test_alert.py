import sqlite3
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes


def setup_app(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    events = []

    def record(event):
        events.append(event)

    monkeypatch.setattr(routes, "log_telemetry", record)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    return client, events


def configure_settings(
    monkeypatch,
    *,
    channel: str,
    outcomes: str = "hit",
    sms_to: tuple[str, ...] | None = ("18005551212",),
):
    monkeypatch.setattr(routes.settings, "alert_channel", channel)
    monkeypatch.setattr(routes.settings, "ALERT_CHANNEL", channel)
    monkeypatch.setattr(routes.settings, "alert_outcomes", outcomes)
    monkeypatch.setattr(routes.settings, "ALERT_OUTCOMES", outcomes)
    if sms_to is not None:
        monkeypatch.setattr(routes.settings, "alert_sms_to", sms_to)


def stub_sms_consent(monkeypatch, numbers: tuple[str, ...]):
    records = [
        {"phone_e164": num, "user_id": f"user-{idx + 1}"}
        for idx, num in enumerate(numbers)
    ]

    monkeypatch.setattr(
        routes.sms_consent,
        "active_destinations",
        lambda **kwargs: list(records),
    )
    monkeypatch.setattr(
        routes.sms_consent,
        "allow_sending",
        lambda number, **kwargs: (True, {"user_id": "user-1", "phone_e164": number}),
    )
    monkeypatch.setattr(routes.sms_consent, "record_delivery", lambda *args, **kwargs: None)


def test_favorites_test_alert_mms(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="MMS", outcomes="all", sms_to=("18005550100",))
    stub_sms_consent(monkeypatch, ("18005550100",))

    send_calls: list[tuple[str, str, dict | None]] = []

    def fake_twilio(number, body, *, context=None):
        send_calls.append((number, body, context))
        return True

    monkeypatch.setattr(routes.twilio_client, "send_mms", fake_twilio)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: True)

    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        called.update(
            {
                "symbol": symbol,
                "channel": channel,
                "compact": compact,
                "outcomes": outcomes,
            }
        )
        return True, {"subject": "Test Subject", "body": "Alert body"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)

    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post("/favorites/test_alert", json={"symbol": "AAPL"})
    assert res.status_code == 200
    data = res.json()
    assert data["channel"] == "MMS"
    assert data["outcomes"] == "all"
    assert data["subject"] == "Test Subject"
    assert called == {
        "symbol": "AAPL",
        "channel": "mms",
        "compact": False,
        "outcomes": "all",
    }
    assert send_calls[0][0] == "+18005550100"
    assert send_calls[0][2]["channel"] == "mms"
    assert send_calls[0][1].endswith("STOP=opt-out, HELP=help")
    assert events[-2] == {
        "type": "favorites_test_alert_send",
        "channel": "mms",
        "provider": "twilio",
        "ok": True,
        "outcomes": "all",
    }
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "mms",
        "outcomes": "all",
        "ok": True,
    }


def test_settings_test_alert_overrides(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="Email", outcomes="hit", sms_to=("18005550100",))
    stub_sms_consent(monkeypatch, ("18005550100",))

    sent: list[tuple[str, str, dict | None]] = []

    def fake_twilio(number, body, *, context=None):
        sent.append((number, body, context))
        return True

    monkeypatch.setattr(routes.twilio_client, "send_mms", fake_twilio)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: True)

    captured: dict[str, object] = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        captured.update(
            {
                "symbol": symbol,
                "channel": channel,
                "compact": compact,
                "outcomes": outcomes,
            }
        )
        return True, {"subject": "Alt Subject", "body": "Message"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)

    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post(
        "/settings/test-alert",
        json={"symbol": "TSLA", "channel": "MMS", "outcomes": "all"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["channel"] == "MMS"
    assert data["outcomes"] == "all"
    assert captured == {
        "symbol": "TSLA",
        "channel": "mms",
        "compact": False,
        "outcomes": "all",
    }
    assert sent and sent[0][0] == "+18005550100"
    assert sent[0][1].endswith("STOP=opt-out, HELP=help")
    assert events[-1]["outcomes"] == "all"


def test_test_alert_email_success_returns_message_id(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="Email", outcomes="hit", sms_to=())

    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        called.update(
            {
                "symbol": symbol,
                "channel": channel,
                "compact": compact,
                "outcomes": outcomes,
            }
        )
        return True, {"subject": "Email Subject", "body": "Line 1\nLine 2"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)

    sent_call = {}

    def fake_send(host, port, user, password, mail_from, to, subject, body, *, context=None):
        sent_call.update(
            {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "mail_from": mail_from,
                "to": to,
                "subject": subject,
                "body": body,
                "context": context,
            }
        )
        return {"ok": True, "provider": "smtp", "message_id": "<msg-123>"}

    monkeypatch.setattr(routes, "send_email_smtp", fake_send)

    client, events = setup_app(tmp_path, monkeypatch)

    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(
        """
        UPDATE settings
           SET smtp_host=?, smtp_port=?, smtp_user=?, smtp_pass=?, mail_from=?, recipients=?
         WHERE id=1
        """,
        (
            "smtp.gmail.com",
            587,
            "alerts@gmail.com",
            "app-pass",
            "Petra Alerts <alerts@gmail.com>",
            "test@example.com",
        ),
    )
    conn.commit()
    conn.close()

    res = client.post("/favorites/test_alert", json={"symbol": "AAPL"})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["message_id"] == "<msg-123>"
    assert data["channel"] == "Email"
    assert data["outcomes"] == "hit"
    assert called == {
        "symbol": "AAPL",
        "channel": "email",
        "compact": False,
        "outcomes": "hit",
    }
    assert sent_call["to"] == ["test@example.com"]
    assert sent_call["context"]["channel"] == "email"
    assert events[-2] == {
        "type": "favorites_test_alert_send",
        "channel": "email",
        "provider": "smtp",
        "ok": True,
        "message_id": "<msg-123>",
        "outcomes": "hit",
    }
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "email",
        "outcomes": "hit",
        "ok": True,
    }


def test_test_alert_email_missing_config_400(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="Email", outcomes="hit", sms_to=())

    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        return True, {"subject": "Email Subject", "body": "Preview"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, events = setup_app(tmp_path, monkeypatch)

    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(
        """
        UPDATE settings
           SET smtp_port=?, smtp_user=?, smtp_pass=?, mail_from=?, recipients=?
         WHERE id=1
        """,
        (
            587,
            "alerts@gmail.com",
            "app-pass",
            "Petra Alerts <alerts@gmail.com>",
            "test@example.com",
        ),
    )
    conn.commit()
    conn.close()

    res = client.post("/favorites/test_alert", json={"symbol": "AAPL"})
    assert res.status_code == 400
    data = res.json()
    assert data["ok"] is False
    assert data["channel"] == "Email"
    assert data["outcomes"] == "hit"
    assert "SMTP not configured" in data["error"]
    assert events[-2] == {
        "type": "favorites_test_alert_send",
        "channel": "email",
        "provider": "smtp",
        "ok": False,
        "error": data["error"],
        "outcomes": "hit",
    }
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "email",
        "outcomes": "hit",
        "ok": False,
    }


def test_preview_returns_body_subject(tmp_path, monkeypatch):
    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        return True, {"subject": f"Preview {symbol}", "body": "Example body"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post(
        "/favorites/test_alert/preview",
        json={"symbol": "MSFT", "channel": "email", "compact": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data == {
        "ok": True,
        "symbol": "MSFT",
        "channel": "email",
        "compact": True,
        "outcomes": "hit",
        "subject": "Preview MSFT",
        "body": "Example body",
    }
    assert events == []


def test_favorites_test_alert_mms_real_layout(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="MMS", outcomes="hit", sms_to=("18005550100",))
    stub_sms_consent(monkeypatch, ("18005550100",))

    def fake_send(number, body, *, context=None):
        return True

    monkeypatch.setattr(routes.twilio_client, "send_mms", fake_send)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: True)

    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post("/favorites/test_alert", json={"symbol": "AAPL"})
    assert res.status_code == 200
    data = res.json()
    body = data["body"]
    assert data["channel"] == "MMS"
    assert data["outcomes"] == "hit"
    assert "AAPL UP | Picked:" in body
    assert "Greeks & IV:" in body
    assert "• Theta" in body
    assert "✅" in body
    assert events[-2]["ok"] is True
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "mms",
        "outcomes": "hit",
        "ok": True,
    }


def test_favorites_test_alert_mms_compact_only_failures(tmp_path, monkeypatch):
    configure_settings(monkeypatch, channel="MMS", outcomes="hit", sms_to=("18005550100",))
    stub_sms_consent(monkeypatch, ("18005550100",))

    def fake_send(number, body, *, context=None):
        return True

    monkeypatch.setattr(routes.twilio_client, "send_mms", fake_send)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: True)

    def fake_enrich(symbol, direction, channel="mms", compact=False, outcomes="hit"):
        return True, {
            "subject": "Compact",
            "body": "• Delta (0.50) ❌ — too high; demo",
        }

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post("/favorites/test_alert", json={"symbol": "AAPL"})
    assert res.status_code == 200
    body = res.json()["body"]
    assert "• Delta" in body
    assert events[-1]["outcomes"] == "hit"

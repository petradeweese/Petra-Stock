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


def test_favorites_test_alert_mms(tmp_path, monkeypatch):
    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False):
        called.update({"symbol": symbol, "channel": channel, "compact": compact})
        return True, {"subject": "Test Subject", "body": f"{symbol} {direction}\nWhy this contract: Δ 0.00"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)
    client, events = setup_app(tmp_path, monkeypatch)
    res = client.post(
        "/favorites/test_alert",
        json={"symbol": "AAPL", "channel": "mms", "compact": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert "Why this contract" in data["body"]
    assert data["subject"] == "Test Subject"
    assert called["channel"] == "mms"
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "mms",
        "compact": False,
        "ok": True,
    }


def test_test_alert_email_success_returns_message_id(tmp_path, monkeypatch):
    called = {}

    def fake_enrich(symbol, direction, channel="mms", compact=False):
        called.update({"symbol": symbol, "channel": channel, "compact": compact})
        return True, {"subject": "Email Subject", "body": "Line 1\nLine 2"}

    monkeypatch.setattr(routes.favorites_alerts, "enrich_and_send_test", fake_enrich)

    sent_call = {}

    def fake_send(host, port, user, password, mail_from, to, subject, body):
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

    res = client.post(
        "/favorites/test_alert",
        json={"symbol": "AAPL", "channel": "email", "compact": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["message_id"] == "<msg-123>"
    assert data["subject"] == "Email Subject"
    assert called["channel"] == "email"
    assert called["compact"] is True
    assert sent_call == {
        "host": "smtp.gmail.com",
        "port": 587,
        "user": "alerts@gmail.com",
        "password": "app-pass",
        "mail_from": "Petra Alerts <alerts@gmail.com>",
        "to": ["test@example.com"],
        "subject": "Email Subject",
        "body": "Line 1\nLine 2",
    }
    assert events[-2] == {
        "type": "favorites_test_alert_send",
        "channel": "email",
        "provider": "smtp",
        "ok": True,
        "message_id": "<msg-123>",
    }
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "email",
        "compact": True,
        "ok": True,
    }


def test_test_alert_email_missing_config_400(tmp_path, monkeypatch):
    def fake_enrich(symbol, direction, channel="mms", compact=False):
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

    res = client.post(
        "/favorites/test_alert",
        json={"symbol": "AAPL", "channel": "email", "compact": False},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["ok"] is False
    assert "SMTP not configured" in data["error"]
    assert "SMTP host" in data["error"]
    assert events[-2] == {
        "type": "favorites_test_alert_send",
        "channel": "email",
        "provider": "smtp",
        "ok": False,
        "error": data["error"],
    }
    assert events[-1] == {
        "type": "favorites_test_alert",
        "symbol": "AAPL",
        "channel": "email",
        "compact": False,
        "ok": False,
    }


def test_preview_returns_body_subject(tmp_path, monkeypatch):
    def fake_enrich(symbol, direction, channel="mms", compact=False):
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
        "subject": "Preview MSFT",
        "body": "Example body",
    }
    assert events == []

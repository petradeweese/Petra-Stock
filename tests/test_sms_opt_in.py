import html
import sqlite3
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes
from services import sms_consent


def build_client(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "sms.db")
    db.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_sms_verify_flow_records_consent(tmp_path, monkeypatch):
    client = build_client(tmp_path, monkeypatch)

    monkeypatch.setattr(routes.twilio_client, "start_verification", lambda phone: "ver-start")
    monkeypatch.setattr(
        routes.twilio_client,
        "check_verification",
        lambda phone, code: (code == "123456", "ver-check"),
    )

    res = client.post(
        "/api/sms/verify/start",
        json={
            "phone": "(800) 555-1212",
            "consent": True,
            "consent_text": "I agree.",
        },
        headers={"user-agent": "pytest"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["ok"]
    assert data["sent"] is True

    res = client.post(
        "/api/sms/verify/check",
        json={
            "phone": "800-555-1212",
            "code": "123456",
            "consent_text": "I agree.",
        },
        headers={"user-agent": "pytest"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    consent = payload["consent"]
    assert consent["phone_e164"] == "+18005551212"
    assert consent["verification_id"] == "ver-check"
    assert consent["method"] == "settings"

    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute(
        "SELECT phone_e164, consent_text, method, revoked_at FROM sms_consent"
    ).fetchone()
    conn.close()
    assert row[0] == "+18005551212"
    assert row[1] == "I agree."
    assert row[2] == "settings"
    assert row[3] is None


def test_sms_reverify_creates_history(tmp_path, monkeypatch):
    client = build_client(tmp_path, monkeypatch)

    monkeypatch.setattr(routes.twilio_client, "start_verification", lambda phone: "sid-1")
    monkeypatch.setattr(
        routes.twilio_client,
        "check_verification",
        lambda phone, code: (True, f"ver-{phone[-4:]}")
    )

    client.post(
        "/api/sms/verify/start",
        json={"phone": "+1 (800) 555-1212", "consent": True, "consent_text": "Consent"},
    )
    client.post(
        "/api/sms/verify/check",
        json={"phone": "+18005551212", "code": "000000", "consent_text": "Consent"},
    )

    client.post(
        "/api/sms/verify/start",
        json={"phone": "800-555-1333", "consent": True, "consent_text": "Consent"},
    )
    client.post(
        "/api/sms/verify/check",
        json={"phone": "8005551333", "code": "000000", "consent_text": "Consent"},
    )

    conn = sqlite3.connect(db.DB_PATH)
    rows = conn.execute(
        "SELECT phone_e164, revoked_at FROM sms_consent ORDER BY consent_at"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0][0] == "+18005551212"
    assert rows[0][1] is not None  # first entry revoked when new number added
    assert rows[1][0] == "+18005551333"
    assert rows[1][1] is None


def test_inbound_keywords_and_codes(tmp_path, monkeypatch):
    client = build_client(tmp_path, monkeypatch)

    start_calls: list[str] = []

    def fake_start(phone: str) -> str:
        start_calls.append(phone)
        return "start-sid"

    def fake_check(phone: str, code: str) -> tuple[bool, str | None]:
        return (code == "123456", "verify-sid")

    monkeypatch.setattr(routes.twilio_client, "start_verification", fake_start)
    monkeypatch.setattr(routes.twilio_client, "check_verification", fake_check)

    sms_consent.record_consent("user-x", "+18005551212", "Consent text")

    res = client.post(
        "/twilio/inbound-sms",
        data={"From": "+18005551212", "Body": "STOP"},
    )
    assert res.status_code == 200
    assert "opted out" in res.text

    conn = sqlite3.connect(db.DB_PATH)
    revoked = conn.execute(
        "SELECT revoked_at FROM sms_consent ORDER BY consent_at DESC LIMIT 1"
    ).fetchone()[0]
    conn.close()
    assert revoked is not None

    res = client.post(
        "/twilio/inbound-sms",
        data={"From": "+18005551212", "Body": "START"},
    )
    assert res.status_code == 200
    assert "verification code" in res.text
    assert start_calls == ["+18005551212"]

    res = client.post(
        "/twilio/inbound-sms",
        data={"From": "+18005551212", "Body": "123456"},
    )
    assert res.status_code == 200
    assert "opted in" in res.text

    rows = sms_consent.history_for_user("user-x")
    assert len(rows) == 2
    assert rows[0]["method"] == "sms-keyword"
    assert rows[0]["verification_id"] == "verify-sid"

    res = client.post(
        "/twilio/inbound-sms",
        data={"From": "+18005551212", "Body": "HELP"},
    )
    assert res.status_code == 200
    assert "Msg&data rates may apply" in html.unescape(res.text)


def test_allow_sending_rate_limit(tmp_path):
    db.DB_PATH = str(tmp_path / "rate.db")
    db.init_db()
    sms_consent.record_consent("rate-user", "+18005550000", "Consent")

    ok, _ = sms_consent.allow_sending("+18005550000", limit_per_day=1)
    assert ok is True
    sms_consent.record_delivery("+18005550000", "rate-user", "body")
    ok, _ = sms_consent.allow_sending("+18005550000", limit_per_day=1)
    assert ok is False

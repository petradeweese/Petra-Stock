import html

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes
from services import sms_consent


def _client(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "hooks.db")
    db.init_db()
    start_calls: list[str] = []

    def fake_start(phone: str) -> str:
        start_calls.append(phone)
        return "verify-start"

    def fake_check(phone: str, code: str):
        return (code == "123456", "verify-ok")

    monkeypatch.setattr(routes.twilio_client, "start_verification", fake_start)
    monkeypatch.setattr(routes.twilio_client, "check_verification", fake_check)

    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    client = TestClient(app)
    return client, start_calls


def test_stop_help_start_keywords(tmp_path, monkeypatch):
    client, start_calls = _client(tmp_path, monkeypatch)
    sms_consent.record_consent("user-123", "+18005551234", "Consent")

    stop = client.post("/twilio/inbound-sms", data={"From": "+18005551234", "Body": "STOP"})
    assert stop.status_code == 200
    assert "opted out" in html.unescape(stop.text).lower()

    help_res = client.post("/twilio/inbound-sms", data={"From": "+18005551234", "Body": "HELP"})
    assert help_res.status_code == 200
    help_text = html.unescape(help_res.text)
    assert "Msg & data rates may apply" in help_text
    assert "support@petrastock.com" in help_text

    start = client.post("/twilio/inbound-sms", data={"From": "+18005551234", "Body": "START"})
    assert start.status_code == 200
    start_text = html.unescape(start.text)
    assert "opted in" in start_text.lower()
    assert start_calls == []

    history = sms_consent.history_for_user("user-123")
    assert len(history) == 2
    latest = history[0]
    assert latest["method"] == "sms-keyword"
    assert latest["verification_id"] is None
    assert latest["revoked_at"] is None

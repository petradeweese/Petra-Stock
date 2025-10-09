import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _client(tmp_path, monkeypatch, *, enabled: bool, verify_enabled: bool) -> TestClient:
    db.DB_PATH = str(tmp_path / "settings-ui.db")
    db.init_db()
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: enabled)
    monkeypatch.setattr(routes.twilio_client, "is_verify_enabled", lambda: verify_enabled)
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return TestClient(app)


def _find_verify_button(html: str) -> str:
    match = re.search(r"<button[^>]*id=\"sms-verify-btn\"[^>]*>", html)
    return match.group(0) if match else ""


def test_settings_shows_unavailable_when_twilio_disabled(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=False, verify_enabled=False)
    res = client.get("/settings")
    assert res.status_code == 200
    text = res.text
    assert "Verification is unavailable" in text
    button_html = _find_verify_button(text)
    assert "disabled" in button_html
    assert "Reply <strong>STOP</strong> to opt out" in text or "Reply STOP" in text
    assert "+1 4705584503" in text
    assert "Example Street" not in text


def test_settings_allows_opt_in_when_twilio_enabled(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=True, verify_enabled=True)
    res = client.get("/settings")
    assert res.status_code == 200
    text = res.text
    match = re.search(r"<p class=\"note\" id=\"sms-unavailable-note\"[^>]*>(.*?)</p>", text, re.S)
    assert match is not None
    assert "Verification is unavailable" not in (match.group(1) or "")
    button_html = _find_verify_button(text)
    assert "disabled" not in button_html
    assert "No more than" in text

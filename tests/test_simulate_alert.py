import json
import sqlite3
from pathlib import Path

import db
import routes
import scanner
from fastapi import FastAPI
from services import config as services_config
from services import favorites_alerts
from starlette.testclient import TestClient


def setup_sim_app(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "sim.db")
    db.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    monkeypatch.setattr(routes, "DEBUG_SIMULATION", True)
    monkeypatch.setattr(scanner, "DEBUG_SIMULATION", True)
    monkeypatch.setattr(services_config, "DEBUG_SIMULATION", True)
    return client


def test_simulate_alert_flow(tmp_path, monkeypatch):
    client = setup_sim_app(tmp_path, monkeypatch)
    alerts_db = Path("alerts.sqlite")
    if alerts_db.exists():
        alerts_db.unlink()

    sent_payloads = []

    def fake_send_email(host, port, user, password, mail_from, recipients, subject, body, context=None):
        sent_payloads.append({
            "host": host,
            "port": port,
            "mail_from": mail_from,
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "context": context,
        })
        return {"ok": True}

    monkeypatch.setattr(favorites_alerts, "send_email_smtp", fake_send_email)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: False)

    payload = {
        "symbol": "AAPL",
        "direction": "DOWN",
        "channel": "mms",
        "outcome": "hit",
        "outcomes_mode": "hit",
        "bar_ts": "2025-09-17T13:45:00+00:00",
        "recipients": ["alerts@example.com"],
        "smtp": {
            "host": "smtp.test",
            "port": 2525,
            "user": "alerts",
            "password": "secret",
            "mail_from": "alerts@example.com",
        },
    }

    res = client.post("/debug/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["was_sent"] is False
    assert data["delivery"]["channel"] == "email"
    assert data["delivery"]["reason"] == "sent"
    assert sent_payloads and "[Sent via Email — MMS unavailable]" in sent_payloads[-1]["body"]

    res_again = client.post("/debug/simulate", json=payload)
    assert res_again.status_code == 200
    data_again = res_again.json()
    assert data_again["was_sent"] is True
    assert "delivery" not in data_again

    csv_path = Path("sim_export.csv")
    json_path = Path("sim_export.json")
    assert csv_path.exists()
    assert json_path.exists()
    export_rows = json.loads(json_path.read_text())
    assert export_rows and export_rows[0]["symbol"] == "AAPL"

    conn = sqlite3.connect(db.DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute("SELECT favorite_id, simulated FROM forward_runs")
        rows = cur.fetchall()
    finally:
        conn.close()
    assert rows and rows[0][1] == 1

    if alerts_db.exists():
        conn = sqlite3.connect(alerts_db, check_same_thread=False)
        try:
            cur = conn.cursor()
            cur.execute("SELECT simulated FROM sent_alerts")
            alert_rows = cur.fetchall()
        finally:
            conn.close()
        assert any(row[0] == 1 for row in alert_rows)


def test_simulate_sms_channel(tmp_path, monkeypatch):
    client = setup_sim_app(tmp_path, monkeypatch)
    monkeypatch.setattr(routes.twilio_client, "is_enabled", lambda: True)

    sent_numbers: list[str] = []

    def fake_send_mms(number, body, *, context=None):
        sent_numbers.append(number)
        return True

    monkeypatch.setattr(routes.twilio_client, "send_mms", fake_send_mms)

    payload = {
        "symbol": "MSFT",
        "direction": "UP",
        "channel": "sms",
        "outcome": "hit",
        "outcomes_mode": "hit",
        "bar_ts": "2025-09-17T14:00:00+00:00",
        "recipients": ["alerts@example.com"],
        "smtp": {
            "host": "smtp.test",
            "port": 2525,
            "user": "alerts",
            "password": "secret",
            "mail_from": "alerts@example.com",
        },
    }

    res = client.post("/debug/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["delivery"]["channel"] in {"sms", "mms"}
    if sent_numbers:
        assert sent_numbers[0].startswith("+") or sent_numbers[0].isdigit()

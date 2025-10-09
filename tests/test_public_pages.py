from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import db
import routes


def _build_client(tmp_path):
    db.DB_PATH = str(tmp_path / "public.db")
    db.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return TestClient(app)


def test_public_pages_render(tmp_path):
    client = _build_client(tmp_path)

    home = client.get("/")
    assert home.status_code == 200
    home_text = home.text
    assert "Enter Scanner" in home_text
    assert "No more than" in home_text
    assert "Msg &amp; data rates may apply" in home_text or "Msg & data rates may apply" in home_text

    about = client.get("/about")
    assert about.status_code == 200
    assert "pattern alerts" in about.text.lower()

    contact = client.get("/contact")
    assert contact.status_code == 200
    assert "support@petrastock.com" in contact.text
    assert "privacy@petrastock.com" in contact.text
    assert "Petra Stock, LLC" in contact.text
    assert "+1 4705584503" in contact.text
    assert "Address" not in contact.text

    privacy = client.get("/privacy")
    assert privacy.status_code == 200
    assert "Msg &amp; data rates may apply" in privacy.text or "Msg & data rates may apply" in privacy.text
    assert "We do not sell or share your phone number or personal information" in privacy.text
    assert "Example Street" not in privacy.text

    terms = client.get("/terms")
    assert terms.status_code == 200
    assert "Alerts are informational only and not financial advice" in terms.text
    assert "No more than" in terms.text

    consent = client.get("/sms-consent")
    assert consent.status_code == 200
    assert "Send Verification Code" in consent.text
    assert "No more than" in consent.text
    assert "Reply <strong>STOP</strong>" in consent.text or "Reply STOP" in consent.text

    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "Sitemap:" in robots.text

    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    assert "https://petrastock.com/" in sitemap.text

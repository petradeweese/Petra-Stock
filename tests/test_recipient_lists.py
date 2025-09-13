import smtplib

from routes import _send_email


def _dummy_smtp(sent):
    class DummySMTP:
        def __init__(self, *args, **kwargs):
            pass

        def login(self, user, pwd):
            pass

        def send_message(self, msg):
            sent.append(msg)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    return DummySMTP


def test_scanner_recipients_ignore_sms(monkeypatch):
    sent = []
    dummy = _dummy_smtp(sent)
    monkeypatch.setattr(smtplib, "SMTP_SSL", dummy)
    monkeypatch.setattr(smtplib, "SMTP", dummy)
    st = {
        "smtp_user": "u@example.com",
        "smtp_pass": "pw",
        "scanner_recipients": "user@example.com, 5551234567@txt.att.net",
    }
    _send_email(st, "sub", "body", list_field="scanner_recipients", allow_sms=False)
    assert len(sent) == 1
    assert sent[0]["To"] == "user@example.com"


def test_favorites_recipients_allow_sms(monkeypatch):
    sent = []
    dummy = _dummy_smtp(sent)
    monkeypatch.setattr(smtplib, "SMTP_SSL", dummy)
    monkeypatch.setattr(smtplib, "SMTP", dummy)
    st = {
        "smtp_user": "u@example.com",
        "smtp_pass": "pw",
        "recipients": "user@example.com, 5551234567@txt.att.net",
    }
    _send_email(st, "sub", "body")
    assert len(sent) == 1
    assert sent[0]["To"] == "user@example.com, 5551234567@txt.att.net"

import routes


def test_scanner_recipients_ignore_sms(monkeypatch):
    sent = []

    def fake_send(host, port, user, password, mail_from, to, subject, body):
        sent.append({
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "mail_from": mail_from,
            "to": to,
            "subject": subject,
            "body": body,
        })
        return {"ok": True, "provider": "smtp", "message_id": "<id>"}

    monkeypatch.setattr(routes, "send_email_smtp", fake_send)
    st = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "u@example.com",
        "smtp_pass": "pw",
        "mail_from": "u@example.com",
        "scanner_recipients": "user@example.com, 5551234567@txt.att.net",
    }
    routes._send_email(st, "sub", "body", list_field="scanner_recipients", allow_sms=False)
    assert len(sent) == 1
    assert sent[0]["to"] == ["user@example.com"]


def test_favorites_recipients_allow_sms(monkeypatch):
    sent = []

    def fake_send(host, port, user, password, mail_from, to, subject, body):
        sent.append({
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "mail_from": mail_from,
            "to": to,
            "subject": subject,
            "body": body,
        })
        return {"ok": True, "provider": "smtp", "message_id": "<id>"}

    monkeypatch.setattr(routes, "send_email_smtp", fake_send)
    st = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "u@example.com",
        "smtp_pass": "pw",
        "mail_from": "u@example.com",
        "recipients": "user@example.com, 5551234567@txt.att.net",
    }
    routes._send_email(st, "sub", "body")
    assert len(sent) == 1
    assert sent[0]["to"] == ["user@example.com", "5551234567@txt.att.net"]

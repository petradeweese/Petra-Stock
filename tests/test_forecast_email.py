import datetime as dt

from jobs import forecast_email


def test_run_and_email_sends_when_forecast_available(monkeypatch):
    asof = dt.datetime(2024, 1, 2, 15, 30, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(forecast_email, "is_trading_day", lambda value: True)
    monkeypatch.setattr(
        forecast_email,
        "load_smtp_settings",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "alerts@example.com",
            "smtp_pass": "pw",
            "mail_from": "alerts@example.com",
            "forecast_recipients": "alice@example.com, bob@example.com",
        },
    )

    def fake_select(asof_arg, *, include_metadata=False):
        assert asof_arg == asof
        data = [
            {
                "ticker": "AAPL",
                "confidence": 0.9,
                "n": 12,
                "median_close_pct": 0.12,
                "iqr_close_pct": [0.05, 0.2],
                "median_high_pct": 0.3,
                "median_low_pct": -0.1,
                "implied_eod_move_pct": 0.15,
                "edge": 0.02,
                "bias": "Long",
                "final_score": 0.42,
                "similarity_breakdown": {
                    "S5m": 0.98,
                    "S30m": 0.95,
                    "S1d": 0.9,
                    "weights": {"S5m": 0.5, "S30m": 0.3, "S1d": 0.2},
                },
                "options_hint": {
                    "bias": "up",
                    "exp_move_pct": 0.02,
                    "suggested_delta": 0.3,
                    "suggested_expiry": "2024-01-05",
                },
            }
        ]
        if include_metadata:
            return data, {"processed": 10, "selected": len(data)}
        return data

    sent_call: dict[str, object] = {}

    def fake_send(
        host,
        port,
        user,
        password,
        mail_from,
        recipients,
        subject,
        html,
        *,
        context=None,
        raise_exceptions=False,
    ):
        sent_call.update(
            {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "mail_from": mail_from,
                "recipients": recipients,
                "subject": subject,
                "html": html,
                "context": context,
                "raise_exceptions": raise_exceptions,
            }
        )
        return {"ok": True, "provider": "smtp"}

    monkeypatch.setattr(forecast_email, "select_forecast_top5", fake_select)
    monkeypatch.setattr(forecast_email, "send_email_smtp", fake_send)

    forecast_email.run_and_email(asof, "test-run")

    assert sent_call["host"] == "smtp.gmail.com"
    assert sent_call["port"] == 587
    assert sent_call["mail_from"] == "alerts@example.com"
    assert sent_call["recipients"] == ["alice@example.com", "bob@example.com"]
    assert "Top 5 Opportunities" in sent_call["subject"]
    assert "Similarity:" in sent_call["html"]
    assert sent_call["raise_exceptions"] is True
    assert "AAPL" in sent_call["html"]
    assert sent_call["context"] == {
        "job": "forecast_email",
        "run_label": "test-run",
        "asof": asof.isoformat(),
        "tickers": ["AAPL"],
    }


def test_run_and_email_scanner_fallback(monkeypatch):
    asof = dt.datetime(2024, 1, 2, 15, 30, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(forecast_email, "is_trading_day", lambda value: True)
    monkeypatch.setattr(
        forecast_email,
        "load_smtp_settings",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "alerts@example.com",
            "smtp_pass": "pw",
            "mail_from": "alerts@example.com",
            "forecast_recipients": "",
            "scanner_recipients": "alice@example.com, Alice@example.com, bob@example.com",
        },
    )

    def fake_select(asof_arg, *, include_metadata=False):
        assert asof_arg == asof
        data = [
            {
                "ticker": "AAPL",
                "confidence": 0.9,
                "n": 12,
                "median_close_pct": 0.12,
                "iqr_close_pct": [0.05, 0.2],
                "median_high_pct": 0.3,
                "median_low_pct": -0.1,
                "implied_eod_move_pct": 0.15,
                "edge": 0.02,
                "bias": "Long",
                "final_score": 0.42,
                "similarity_breakdown": {"S5m": 0.98, "S30m": 0.95, "S1d": 0.9},
            }
        ]
        if include_metadata:
            return data, {"processed": 3, "selected": len(data)}
        return data

    sent_call: dict[str, object] = {}

    def fake_send(
        host,
        port,
        user,
        password,
        mail_from,
        recipients,
        subject,
        html,
        *,
        context=None,
        raise_exceptions=False,
    ):
        sent_call.update({
            "recipients": recipients,
            "subject": subject,
            "raise_exceptions": raise_exceptions,
        })
        return {"ok": True, "provider": "smtp"}

    monkeypatch.setattr(forecast_email, "select_forecast_top5", fake_select)
    monkeypatch.setattr(forecast_email, "send_email_smtp", fake_send)

    forecast_email.run_and_email(asof, "test-run")

    assert sent_call["recipients"] == ["alice@example.com", "bob@example.com"]
    assert sent_call["raise_exceptions"] is True

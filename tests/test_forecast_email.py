import datetime as dt

from jobs import forecast_email


def test_run_and_email_sends_when_forecast_available(monkeypatch):
    asof = dt.datetime(2024, 1, 2, 15, 30, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(forecast_email, "is_trading_day", lambda value: True)
    monkeypatch.setenv("FORECAST_EMAIL_RECIPIENTS", "alice@example.com, bob@example.com")
    monkeypatch.setenv("ALERTS_FROM_EMAIL", "alerts@example.com")

    def fake_select(asof_arg):
        assert asof_arg == asof
        return [
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
            }
        ]

    sent_call: dict[str, object] = {}

    def fake_send(sender, recipients, subject, html, *, context=None):
        sent_call.update(
            {
                "sender": sender,
                "recipients": recipients,
                "subject": subject,
                "html": html,
                "context": context,
            }
        )
        return {"ok": True, "provider": "smtp"}

    monkeypatch.setattr(forecast_email, "select_forecast_top5", fake_select)
    monkeypatch.setattr(forecast_email, "send_email", fake_send)

    forecast_email.run_and_email(asof, "test-run")

    assert sent_call["sender"] == "alerts@example.com"
    assert sent_call["recipients"] == ["alice@example.com", "bob@example.com"]
    assert "test-run" in sent_call["subject"]
    assert "AAPL" in sent_call["html"]
    assert sent_call["context"] == {
        "job": "forecast_email",
        "run_label": "test-run",
        "asof": asof.isoformat(),
        "tickers": ["AAPL"],
    }

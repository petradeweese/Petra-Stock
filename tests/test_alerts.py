from datetime import datetime, timedelta

from services.alerts import alert_due, in_earnings_blackout


def test_alert_due_cooldown_and_bar():
    now = datetime(2024, 1, 1, 10, 30)
    fav = {
        "cooldown_minutes": 30,
        "last_notified_ts": (now - timedelta(minutes=10)).isoformat(),
        "last_signal_bar": "2024-01-01T10:15:00",
    }
    bar_time = datetime.fromisoformat("2024-01-01T10:15:00")
    assert not alert_due(fav, bar_time, now)

    bar_time = datetime.fromisoformat("2024-01-01T10:30:00")
    fav["last_notified_ts"] = (now - timedelta(minutes=20)).isoformat()
    assert not alert_due(fav, bar_time, now)

    fav["last_notified_ts"] = (now - timedelta(minutes=31)).isoformat()
    assert alert_due(fav, bar_time, now)


def test_in_earnings_blackout():
    now = datetime(2024, 1, 10)
    dates = [datetime(2024, 1, 15)]
    assert in_earnings_blackout(dates, now)
    far = [datetime(2024, 2, 1)]
    assert not in_earnings_blackout(far, now)

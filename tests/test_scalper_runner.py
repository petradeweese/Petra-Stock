from datetime import datetime, timedelta, timezone

from services.scalper.runner import _determine_lookback


def test_determine_lookback_default():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    lookback = _determine_lookback(45, now=now, open_rows=[], max_minutes=300)
    assert lookback == 45


def test_determine_lookback_extends_for_open_trade():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    entry = now - timedelta(minutes=120)
    row = {"entry_time": entry.isoformat()}
    lookback = _determine_lookback(45, now=now, open_rows=[row], max_minutes=300)
    assert lookback >= 125
    assert lookback <= 300


def test_determine_lookback_ignores_invalid_entries():
    now = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)
    rows = [
        {"entry_time": None},
        {"entry_time": "invalid"},
    ]
    lookback = _determine_lookback(60, now=now, open_rows=rows, max_minutes=120)
    assert lookback == 60

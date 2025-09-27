from datetime import datetime, timezone

import pytest

from services import favorites_alerts


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    monkeypatch.setattr(favorites_alerts, "_SENT_ALERTS", {})
    monkeypatch.setattr(favorites_alerts, "_SQLITE_CONN", None)
    monkeypatch.setattr(favorites_alerts, "_SQLITE_PATH", tmp_path / "alerts.sqlite")
    monkeypatch.setattr(favorites_alerts, "_REDIS_CLIENT", None)
    monkeypatch.setattr(favorites_alerts, "_REDIS_READY", None)


def test_dedupe_ttl_expiry(monkeypatch):
    base_ts = 1_700_000_000

    def fake_time():
        return fake_time.current

    fake_time.current = float(base_ts)

    monkeypatch.setattr(favorites_alerts, "_now", lambda: fake_time())
    monkeypatch.setattr(favorites_alerts, "_REDIS_TTL_SECONDS", 60)

    key = favorites_alerts._dedupe_key("fav-1", "15m", datetime.now(timezone.utc))
    assert key

    assert favorites_alerts.was_sent_key("fav-1", "now", interval="15m", dedupe_key=key) is False
    favorites_alerts.mark_sent_key("fav-1", "now", interval="15m", dedupe_key=key)

    assert favorites_alerts.was_sent_key("fav-1", "now", interval="15m", dedupe_key=key) is True

    fake_time.current = base_ts + 61
    assert favorites_alerts.was_sent_key("fav-1", "now", interval="15m", dedupe_key=key) is False

    favorites_alerts.mark_sent_key("fav-1", "later", interval="15m", dedupe_key=key)

    fake_time.current = base_ts + 120
    assert favorites_alerts.was_sent_key("fav-1", "later", interval="15m", dedupe_key=key) is True

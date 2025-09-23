import datetime as dt

import pytest

import scheduler
from scanner import _ensure_coverage
from services import market_data
from services.price_store import clear_cache
from services.price_utils import DataUnavailableError


def test_gap_fill_enqueued(monkeypatch):
    called = {}

    def fake_enqueue(key, fn):
        called["key"] = key

    monkeypatch.setattr(scheduler.work_queue, "enqueue", fake_enqueue)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    clear_cache()
    monkeypatch.setattr(market_data, "DEFAULT_PROVIDER", "db")
    market_data.get_prices(["AAPL"], "15m", start, end)
    assert "gap:AAPL" in called.get("key", "")


def test_ensure_coverage_refuses(monkeypatch):
    monkeypatch.setattr(scheduler.work_queue, "enqueue", lambda *a, **k: None)
    clear_cache()
    with pytest.raises(DataUnavailableError):
        _ensure_coverage("AAPL", "15m", 0.01)

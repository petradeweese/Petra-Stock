from datetime import datetime, timezone

import pytest

from services import scheduler


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    monkeypatch.setattr(scheduler, "_LAST_TICK_BOUNDARY", None)


def test_tick_runs_once_per_bar(monkeypatch):
    calls = []

    monkeypatch.setattr(scheduler.market_calendar, "is_open", lambda now: True)
    monkeypatch.setattr(
        scheduler.scans,
        "run_autoscan_batch",
        lambda: calls.append("run"),
    )

    base = datetime(2024, 1, 2, 14, 45, tzinfo=timezone.utc)

    scheduler._tick(base.replace(second=4))
    scheduler._tick(base.replace(second=11))
    scheduler._tick(base.replace(second=30))
    scheduler._tick(base.replace(second=50))

    next_bar = base.replace(minute=0, hour=15)
    scheduler._tick(next_bar.replace(second=9))
    scheduler._tick(next_bar.replace(second=12))

    assert calls == ["run", "run"]

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils
from utils import TZ, clamp_market_closed


@pytest.fixture(autouse=True)
def no_calendar(monkeypatch):
    monkeypatch.setattr(utils, "mcal", None)
    monkeypatch.setattr(utils, "_XNYS", None)


def test_clamp_noop_during_market():
    start = dt.datetime(2023, 1, 3, 14, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 3, 15, tzinfo=dt.timezone.utc)
    new_end, clamped = clamp_market_closed(start, end)
    assert new_end == end
    assert clamped is False


def test_clamp_after_hours():
    start = dt.datetime(2023, 1, 3, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 3, 22, tzinfo=dt.timezone.utc)
    new_end, clamped = clamp_market_closed(start, end)
    assert clamped is True
    assert new_end == dt.datetime(2023, 1, 3, 21, tzinfo=dt.timezone.utc)


def test_clamp_weekend():
    start = dt.datetime(2023, 1, 6, 19, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 7, 16, tzinfo=dt.timezone.utc)
    new_end, clamped = clamp_market_closed(start, end)
    assert clamped is True
    assert new_end == dt.datetime(2023, 1, 6, 21, tzinfo=dt.timezone.utc)


def test_clamp_holiday(monkeypatch):
    original = utils.market_is_open

    def fake_market_is_open(ts=None):
        ts = ts or dt.datetime.now(tz=TZ)
        if ts.date() == dt.date(2023, 7, 4):
            return False
        return original(ts)

    monkeypatch.setattr(utils, "market_is_open", fake_market_is_open)
    start = dt.datetime(2023, 7, 3, 12, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 7, 4, 16, tzinfo=dt.timezone.utc)
    new_end, clamped = clamp_market_closed(start, end)
    assert clamped is True
    assert new_end == dt.datetime(2023, 7, 3, 20, tzinfo=dt.timezone.utc)


def test_friday_close_to_monday_open():
    start = dt.datetime(2023, 1, 6, 21, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 9, 13, tzinfo=dt.timezone.utc)
    new_end, clamped = clamp_market_closed(start, end)
    assert clamped is True
    assert new_end == start

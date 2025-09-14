import datetime
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import TZ, market_is_open


def test_market_is_closed_on_weekend():
    ts = datetime.datetime(2023, 1, 1, 12, tzinfo=TZ)  # Sunday
    assert market_is_open(ts) is False


def test_market_is_closed_on_holiday():
    pytest.importorskip("pandas_market_calendars")
    ts = datetime.datetime(2023, 7, 4, 12, tzinfo=TZ)
    assert market_is_open(ts) is False


def test_market_is_closed_on_future_date():
    """Ensure future dates outside calendar range don't raise errors."""
    pytest.importorskip("pandas_market_calendars")
    ts = datetime.datetime(2035, 1, 1, 12, tzinfo=TZ)
    assert market_is_open(ts) is False

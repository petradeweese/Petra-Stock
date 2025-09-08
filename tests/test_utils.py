import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import market_is_open, TZ


def test_market_is_closed_on_weekend():
    ts = datetime.datetime(2023, 1, 1, 12, tzinfo=TZ)  # Sunday
    assert market_is_open(ts) is False

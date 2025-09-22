import datetime as dt

import pandas as pd

from services import data_provider


def test_normalize_window_handles_dst():
    start = dt.datetime(2024, 9, 3, 12, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    _, _, start_ms, _ = data_provider._normalize_window(start, end)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    assert start_dt == pd.Timestamp(start)

    start = dt.datetime(2024, 1, 3, 12, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    _, _, start_ms, _ = data_provider._normalize_window(start, end)
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    assert start_dt == pd.Timestamp(start)


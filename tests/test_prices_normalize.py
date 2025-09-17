import datetime as dt

import pandas as pd
import pytest

from services.market_data import filter_session, normalize_corporate_actions
from utils import CLOSE_TIME, OPEN_TIME, TZ


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def test_normalize_corporate_actions_split_adjusts_prices_and_volume():
    index = pd.to_datetime(
        [
            "2024-01-02T14:30:00Z",
            "2024-01-03T14:30:00Z",
            "2024-01-04T14:30:00Z",
        ]
    )
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0],
            "High": [101.0, 103.0, 105.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [100.5, 102.5, 104.5],
            "Volume": [1_000.0, 1_100.0, 1_200.0],
        },
        index=index,
    )
    actions = pd.DataFrame(
        [
            {
                "type": "split",
                "effective": dt.datetime(2024, 1, 4, 0, 0, tzinfo=dt.timezone.utc),
                "ratio": 2,
            }
        ]
    )

    adjusted = normalize_corporate_actions(df, actions)

    pre_split = adjusted.loc[adjusted.index < _utc("2024-01-04T00:00:00Z")]
    pd.testing.assert_series_equal(
        pre_split["Close"],
        df.loc[df.index < _utc("2024-01-04T00:00:00Z"), "Close"] / 2,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pre_split["Volume"],
        df.loc[df.index < _utc("2024-01-04T00:00:00Z"), "Volume"] * 2,
        check_names=False,
    )
    # Post-split bars remain unchanged
    post_split = adjusted.loc[adjusted.index >= _utc("2024-01-04T00:00:00Z")]
    pd.testing.assert_series_equal(
        post_split["Close"],
        df.loc[df.index >= _utc("2024-01-04T00:00:00Z"), "Close"],
        check_names=False,
    )


@pytest.mark.parametrize("freq", ["15min", "1h"])
def test_filter_session_rth_limits_to_regular_hours(freq: str):
    periods = 40 if freq == "15min" else 24
    start = pd.Timestamp("2024-03-18T11:00:00Z")
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "Open": range(periods),
            "High": range(periods),
            "Low": range(periods),
            "Close": range(periods),
            "Volume": [1_000] * periods,
        },
        index=index,
    )

    filtered = filter_session(df, "RTH")
    converted = df.index.tz_convert(TZ)
    expected = [
        ts for ts in converted if OPEN_TIME <= ts.time() <= CLOSE_TIME
    ]

    assert len(filtered) == len(expected)
    if expected:
        filtered_times = filtered.index.tz_convert(TZ)
        assert all(OPEN_TIME <= ts.time() <= CLOSE_TIME for ts in filtered_times)
    else:
        assert filtered.empty

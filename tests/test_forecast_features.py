import datetime as dt

import pandas as pd
import pandas.testing as pdt

from services import price_store
from services import forecast_features


def test_load_price_frame_allows_unreadable_token(monkeypatch):
    symbol = "AAPL"
    start = dt.datetime(2024, 1, 1, 14, 0, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=5)

    idx = pd.date_range(start=start, periods=5, freq="1min", tz="UTC")
    populated = pd.DataFrame(
        {
            "Open": [100 + i for i in range(5)],
            "High": [101 + i for i in range(5)],
            "Low": [99 + i for i in range(5)],
            "Close": [100.5 + i for i in range(5)],
            "Adj Close": [100.5 + i for i in range(5)],
            "Volume": [1_000 + i for i in range(5)],
        },
        index=idx,
    )
    fetched = populated.copy()
    fetched.attrs["provider"] = "schwab"
    empty = pd.DataFrame(columns=populated.columns)

    call_state = {"count": 0}

    def fake_get_prices_from_db(symbols, start_arg, end_arg, *, interval):
        assert symbols == [symbol]
        assert interval == "1m"
        call_state["count"] += 1
        if call_state["count"] == 1:
            return {symbol: empty}
        return {symbol: populated}

    def fake_detect_gaps(*args, **kwargs):
        return []

    def fake_fetch(symbols, interval, start_arg, end_arg):
        assert symbols == [symbol]
        return {symbol: fetched}

    monkeypatch.setattr(price_store, "get_prices_from_db", fake_get_prices_from_db)
    monkeypatch.setattr(price_store, "detect_gaps", fake_detect_gaps)
    monkeypatch.setattr(price_store, "clear_cache", lambda: None)
    monkeypatch.setattr(forecast_features, "_TOKEN_STATUS_LOGGED", False)
    monkeypatch.setattr(
        forecast_features,
        "_token_path_status",
        lambda: (False, "unreadable", "/tmp/token"),
    )

    async def _noop_token():
        return ""

    monkeypatch.setattr(
        forecast_features.schwab_client._client,
        "_ensure_token",
        lambda: _noop_token(),
    )
    monkeypatch.setattr(
        forecast_features.schwab_client,
        "disabled_state",
        lambda: (False, None, None, None),
    )
    monkeypatch.setattr(forecast_features, "fetch_bars", fake_fetch)

    frame, source = forecast_features.load_price_frame(symbol, start, end, "1m")

    assert call_state["count"] == 2
    pdt.assert_frame_equal(frame, populated)
    assert source == "schwab"

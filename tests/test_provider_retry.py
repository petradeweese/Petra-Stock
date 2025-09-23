import asyncio
import datetime as dt
import asyncio
import datetime as dt
from types import SimpleNamespace

import pandas as pd

from services import data_provider
from services.schwab_client import SchwabAPIError


def _sample_frame(ts: dt.datetime) -> pd.DataFrame:
    ts = ts.replace(hour=14, minute=30)
    idx = pd.DatetimeIndex([ts], tz="UTC")
    return pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1],
        },
        index=idx,
    )


def test_retry_success(monkeypatch, caplog):
    attempts = 0

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise SchwabAPIError("boom")
        return _sample_frame(start)

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
        ),
    )
    monkeypatch.setattr(data_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        data_provider,
        "settings",
        SimpleNamespace(fetch_retry_max=4, fetch_retry_base_ms=1, fetch_retry_cap_ms=2),
    )

    start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)
    with caplog.at_level("WARNING"):
        df, provider = asyncio.run(
            data_provider._fetch_single("SPY", start, end, interval="15m")
        )
    assert attempts == 3
    assert not df.empty
    assert provider == "schwab"
    msgs = [r.message for r in caplog.records if "schwab_retry" in r.message]
    assert len(msgs) == 2


def test_retry_exhaust_falls_back(monkeypatch):
    attempts = 0
    fallback_called = False

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        nonlocal attempts
        attempts += 1
        raise SchwabAPIError("boom")

    async def fake_sleep(_):
        return None

    async def fake_yfinance(symbol, start, end, interval):
        nonlocal fallback_called
        fallback_called = True
        return _sample_frame(start)

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
        ),
    )
    monkeypatch.setattr(data_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        data_provider,
        "settings",
        SimpleNamespace(fetch_retry_max=2, fetch_retry_base_ms=1, fetch_retry_cap_ms=2),
    )
    monkeypatch.setattr(data_provider, "_fetch_yfinance", fake_yfinance)

    start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)
    df, provider = asyncio.run(
        data_provider._fetch_single("SPY", start, end, interval="15m")
    )
    assert attempts == 2
    assert fallback_called
    assert not df.empty
    assert provider == "yfinance"

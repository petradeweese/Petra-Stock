import asyncio
import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from services import data_provider
from services.schwab_client import SchwabAPIError


def test_fetch_bars_uses_schwab(monkeypatch):
    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        idx = pd.date_range(start, periods=2, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Adj Close": [1.2, 2.2],
                "Volume": [100, 200],
            },
            index=idx,
        )

    async def fake_quote(symbol, timeout_ctx=None):
        return {
            "symbol": symbol,
            "price": 10.0,
            "timestamp": dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc),
            "source": "schwab",
        }

    fake_client = SimpleNamespace(
        get_price_history=fake_history,
        get_quote=fake_quote,
        last_status=lambda: 200,
        disabled_state=lambda: (False, None, None, None),
        disable=lambda **_: None,
    )

    async def fail_yf(*args, **kwargs):
        pytest.fail("unexpected fallback")

    async def fail_yf_quote(symbol):
        pytest.fail("unexpected quote fallback")

    monkeypatch.setattr(data_provider, "schwab_client", fake_client)
    monkeypatch.setattr(data_provider, "_fetch_yfinance", fail_yf)
    monkeypatch.setattr(data_provider, "_fetch_yfinance_quote", fail_yf_quote)

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(minutes=30)
    end = now
    bars = asyncio.run(data_provider.fetch_bars_async(["AAPL"], "15m", start, end))
    df = bars["AAPL"]
    assert len(df) == 2
    assert df.attrs["provider"] == "schwab"

    quote = asyncio.run(data_provider.get_quote_async("AAPL"))
    assert quote["source"] == "schwab"
    assert quote["price"] == 10.0


def test_fetch_bars_fallbacks_to_yfinance(monkeypatch, caplog):
    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        raise SchwabAPIError("boom", status_code=429)

    async def fake_yf(symbol, start, end, interval):
        idx = pd.date_range(start, periods=1, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "Open": [3.0],
                "High": [3.5],
                "Low": [2.5],
                "Close": [3.2],
                "Adj Close": [3.2],
                "Volume": [300],
            },
            index=idx,
        )

    async def fake_yf_quote(symbol):
        return {"symbol": symbol, "price": 9.0, "timestamp": None, "source": "yfinance"}

    async def failing_quote(symbol, timeout_ctx=None):
        raise SchwabAPIError("boom", status_code=429)

    fake_client = SimpleNamespace(
        get_price_history=fake_history,
        get_quote=failing_quote,
        last_status=lambda: 429,
        disabled_state=lambda: (False, None, None, None),
        disable=lambda **_: None,
    )

    monkeypatch.setattr(data_provider, "schwab_client", fake_client)
    monkeypatch.setattr(data_provider, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(data_provider, "_fetch_yfinance_quote", fake_yf_quote)

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(minutes=15)
    end = now
    with caplog.at_level("INFO"):
        bars = asyncio.run(
            data_provider.fetch_bars_async(["MSFT"], "15m", start, end)
        )
    df = bars["MSFT"]
    assert df.attrs["provider"] == "yfinance"
    assert any("yfinance_fetch" in record.message for record in caplog.records)

    quote = asyncio.run(data_provider.get_quote_async("MSFT"))
    assert quote["source"] == "yfinance"
    assert quote["price"] == 9.0


def test_fetch_bars_returns_empty_when_all_fail(monkeypatch, caplog):
    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        raise SchwabAPIError("boom", status_code=500)

    async def fake_yf(symbol, start, end, interval):
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

    async def failing_quote(symbol, timeout_ctx=None):
        raise SchwabAPIError("boom", status_code=500)

    fake_client = SimpleNamespace(
        get_price_history=fake_history,
        get_quote=failing_quote,
        last_status=lambda: 500,
        disabled_state=lambda: (False, None, None, None),
        disable=lambda **_: None,
    )

    async def fake_yf_quote(symbol):
        return {}

    monkeypatch.setattr(data_provider, "schwab_client", fake_client)
    monkeypatch.setattr(data_provider, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(data_provider, "_fetch_yfinance_quote", fake_yf_quote)

    start = dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=15)
    with caplog.at_level("INFO"):
        result = asyncio.run(
            data_provider._fetch_single("QQQ", start, end, interval="15m")
        )
    df, provider = result
    assert provider == "none"
    assert df.empty
    assert any("yfinance_fetch" in record.message for record in caplog.records)

    quote = asyncio.run(data_provider.get_quote_async("QQQ"))
    assert quote == {}

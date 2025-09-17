import datetime as dt
from typing import List

import pandas as pd
import pytest

from services.errors import DataUnavailableError
import services.market_data as market_data
from services.market_data import _fetch_with_fallback, get_options_chain, normalize_options_df
from services.providers import schwab, yahoo, schwab_options, yahoo_options
from services.providers.schwab_options import OptionsUnavailableError


@pytest.fixture(autouse=True)
def _schwab_env(monkeypatch):
    monkeypatch.setenv("SCHWAB_BARS_PATH", "/v1/bars")
    monkeypatch.setenv("SCHWAB_API_BASE", "https://example.com")
    monkeypatch.setenv("SCHWAB_OPTIONS_PATH", "/v1/options")
    yield


def _sample_rows() -> List[dict]:
    base = dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc)
    rows: List[dict] = []
    for i in range(4):
        ts = base + dt.timedelta(minutes=i)
        rows.append(
            {
                "datetime": ts.isoformat(),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.5 + i,
                "close": 100.5 + i,
                "volume": 1_000 + 10 * i,
            }
        )
    return rows


def _sample_option_payload() -> dict:
    return {
        "optionChain": {
            "underlying": {"symbol": "AAPL"},
            "callExpDateMap": {
                "2024-05-17:7": {
                    "150": [
                        {
                            "optionSymbol": "AAPL240517C00150000",
                            "bid": 1.25,
                            "ask": 1.3,
                            "last": 1.27,
                            "delta": 0.45,
                            "gamma": 0.05,
                            "theta": -0.02,
                            "vega": 0.12,
                            "openInterest": 1500,
                            "volume": 200,
                            "impliedVolatility": 0.28,
                            "quoteTime": "2024-04-01T15:45:00Z",
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2024-05-17:7": {
                    "150": [
                        {
                            "optionSymbol": "AAPL240517P00150000",
                            "bid": 1.4,
                            "ask": 1.45,
                            "last": 1.42,
                            "delta": -0.53,
                            "gamma": 0.06,
                            "theta": -0.03,
                            "vega": 0.14,
                            "openInterest": 1800,
                            "volume": 250,
                            "impliedVolatility": 32.0,
                            "quoteTime": "2024-04-01T15:46:00Z",
                        }
                    ]
                }
            },
        }
    }


def test_schwab_fetch_prices_normalizes_schema(monkeypatch):
    start = dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=3)

    monkeypatch.setattr(schwab, "get_session", lambda: object())
    monkeypatch.setattr(schwab, "_get_access_token", lambda _session: "token")
    monkeypatch.setattr(schwab, "_fetch_raw_bars", lambda *args, **kwargs: _sample_rows())

    df = schwab.fetch_prices("SPY", "1m", start, end)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert not df.empty
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing


def test_schwab_options_fetch_chain_normalizes(monkeypatch):
    payload = _sample_option_payload()

    monkeypatch.setattr(schwab_options, "get_session", lambda: object())
    monkeypatch.setattr(schwab_options.schwab, "get_access_token", lambda _s: "token")
    monkeypatch.setattr(
        schwab_options,
        "_fetch_raw_chain",
        lambda *args, **kwargs: payload,
    )

    df = schwab_options.fetch_chain("AAPL")
    assert list(df.columns) == market_data._OPTIONS_COLUMNS
    assert len(df) == 2
    assert set(df["type"]) == {"call", "put"}
    # IV percentage converted to fraction
    assert pytest.approx(df.loc[df["type"] == "put", "iv"].iloc[0]) == 0.32
    assert df["updated_at"].dt.tz is not None


def test_market_data_fallback_to_yahoo(monkeypatch):
    start = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)

    def _raise(*_args, **_kwargs):
        raise DataUnavailableError("schwab down")

    fallback_df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [1_000],
        },
        index=pd.DatetimeIndex([start], tz="UTC"),
    )

    calls = []

    def _yahoo(*args, **kwargs):
        calls.append(args[0])
        return fallback_df

    monkeypatch.setattr(schwab, "fetch_prices", _raise)
    monkeypatch.setattr(yahoo, "fetch_prices", _yahoo)

    df = _fetch_with_fallback("AAPL", "1d", start, end)
    assert calls == ["AAPL"]
    assert df.equals(fallback_df)


def test_get_options_chain_falls_back_to_yahoo(monkeypatch):
    def _raise_options(*_args, **_kwargs):
        raise OptionsUnavailableError("schwab unavailable")

    fallback = pd.DataFrame(
        {
            "contractSymbol": ["AAPL240517C00150000"],
            "strike": [150.0],
            "bid": [1.1],
            "ask": [1.25],
            "impliedVolatility": [25.0],
            "lastTradeDate": [pd.Timestamp("2024-04-01T15:45:00Z")],
            "type": ["call"],
            "expiry": ["2024-05-17"],
        }
    )

    monkeypatch.setattr(schwab_options, "fetch_chain", _raise_options)
    monkeypatch.setattr(yahoo_options, "fetch_chain", lambda *args, **kwargs: fallback)

    df = get_options_chain("AAPL")
    assert list(df.columns) == market_data._OPTIONS_COLUMNS
    assert len(df) == 1
    assert pytest.approx(df.loc[0, "iv"]) == 0.25
    assert df.loc[0, "updated_at"].tzinfo is not None


def test_normalize_options_df_requires_bid_and_ask():
    raw = pd.DataFrame(
        {
            "contractSymbol": ["AAPL240517C00150000"],
            "strike": [150.0],
            "bid": [None],
            "ask": [1.2],
            "impliedVolatility": [0.25],
            "lastTradeDate": [pd.Timestamp("2024-04-01T15:45:00Z")],
            "type": ["call"],
            "expiry": ["2024-05-17"],
        }
    )

    with pytest.raises(OptionsUnavailableError):
        normalize_options_df(raw, "AAPL")


@pytest.mark.parametrize("interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
def test_schwab_interval_resampling(monkeypatch, interval):
    monkeypatch.setattr(schwab, "get_session", lambda: object())
    monkeypatch.setattr(schwab, "_get_access_token", lambda _session: "token")

    def _fake_fetch_raw(_session, _token, _symbol, spec, start, end):
        rows: List[dict] = []
        if spec.granularity == "day":
            current = start
            step = dt.timedelta(days=1)
        else:
            current = start
            step = dt.timedelta(minutes=spec.multiplier)
        counter = 0
        while current <= end:
            rows.append(
                {
                    "datetime": current.isoformat(),
                    "open": 50 + counter,
                    "high": 51 + counter,
                    "low": 49 + counter,
                    "close": 50.5 + counter,
                    "volume": 500 + counter * 10,
                }
            )
            current += step
            counter += 1
        return rows

    monkeypatch.setattr(schwab, "_fetch_raw_bars", _fake_fetch_raw)

    if interval.endswith("d"):
        start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=2)
    else:
        start = dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(hours=3)

    df = schwab.fetch_prices("QQQ", interval, start, end)
    assert not df.empty
    assert df.index.is_monotonic_increasing
    assert df.index.tz is not None
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_retry_fetch_transient(monkeypatch):
    attempts = {"count": 0}

    def _fake_fetch():
        attempts["count"] += 1
        if attempts["count"] < 3:
            err = DataUnavailableError("try again")
            err.status = 429
            raise err
        return "ok"

    monkeypatch.setattr(market_data.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(market_data.time, "sleep", lambda *_: None)

    result = market_data._retry_fetch(_fake_fetch)
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_fetch_non_transient(monkeypatch):
    def _fake_fetch():
        err = DataUnavailableError("no retry")
        err.status = 404
        raise err

    monkeypatch.setattr(market_data.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(market_data.time, "sleep", lambda *_: None)

    with pytest.raises(DataUnavailableError):
        market_data._retry_fetch(_fake_fetch)

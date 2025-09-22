import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from services import data_provider
from services.price_store import clear_cache, get_prices_from_db, upsert_bars
import db


def _frame(timestamps):
    return pd.DataFrame(
        {
            "Open": range(1, len(timestamps) + 1),
            "High": range(1, len(timestamps) + 1),
            "Low": range(1, len(timestamps) + 1),
            "Close": range(1, len(timestamps) + 1),
            "Volume": range(1, len(timestamps) + 1),
        },
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
    )


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(
        data_provider,
        "settings",
        SimpleNamespace(fetch_retry_max=1, fetch_retry_base_ms=1, fetch_retry_cap_ms=1),
    )
    monkeypatch.delenv("SCHWAB_INCLUDE_PREPOST", raising=False)


def test_fetch_bars_filters_session(monkeypatch):
    pre = pd.Timestamp("2024-01-02 14:15", tz="UTC")
    regular = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    post = pd.Timestamp("2024-01-02 21:15", tz="UTC")
    frame = _frame([pre, regular, post])

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        return frame

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
        ),
    )
    start = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)
    data = data_provider.fetch_bars(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert list(df.index) == [regular]


def test_fetch_bars_include_prepost(monkeypatch):
    monkeypatch.setenv("SCHWAB_INCLUDE_PREPOST", "true")
    timestamps = [
        pd.Timestamp("2024-01-02 14:15", tz="UTC"),
        pd.Timestamp("2024-01-02 14:30", tz="UTC"),
        pd.Timestamp("2024-01-02 21:15", tz="UTC"),
    ]
    frame = _frame(timestamps)

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        return frame

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
        ),
    )
    start = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)
    data = data_provider.fetch_bars(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert list(df.index) == timestamps


def test_upsert_idempotent(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    ts0 = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 10:15", tz="UTC")
    df1 = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [100, 200],
        },
        index=[ts0, ts1],
    )
    upsert_bars("AAA", df1, "15m")
    df2 = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 3],
            "Volume": [100, 300],
        },
        index=[ts0, ts1],
    )
    upsert_bars("AAA", df2, "15m")
    res = get_prices_from_db(["AAA"], ts0.to_pydatetime(), ts1.to_pydatetime(), "15m")
    df = res["AAA"]
    assert len(df) == 2
    assert float(df.loc[ts1, "Close"]) == 3
    assert float(df.loc[ts1, "Volume"]) == 300
    clear_cache()

import datetime as dt
from types import SimpleNamespace

import pandas as pd
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
    base = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).replace(
        hour=14, minute=30, second=0, microsecond=0
    )
    regular = base
    pre = regular - pd.Timedelta(minutes=15)
    post = regular + pd.Timedelta(hours=6, minutes=45)
    frame = _frame([pre, regular, post])

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        return data_provider._align_to_session(frame.copy())

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
            disabled_state=lambda: (False, None, None, None),
            disable=lambda **_: None,
        ),
    )
    start = pre.to_pydatetime()
    end = (post + pd.Timedelta(minutes=15)).to_pydatetime()
    symbol = "AAA_SESSION"
    conn = db.get_engine().raw_connection()
    try:
        conn.execute("DELETE FROM bars WHERE symbol=?", (symbol,))
        conn.commit()
    finally:
        conn.close()
    data = data_provider.fetch_bars([symbol], "15m", start, end)
    df = data[symbol]
    assert regular in df.index
    assert post not in df.index


def test_fetch_bars_include_prepost(monkeypatch):
    monkeypatch.setenv("SCHWAB_INCLUDE_PREPOST", "true")
    base = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).replace(
        hour=14, minute=30, second=0, microsecond=0
    )
    timestamps = [
        base - pd.Timedelta(minutes=15),
        base,
        base + pd.Timedelta(hours=6, minutes=45),
    ]
    frame = _frame(timestamps)

    async def fake_history(symbol, start, end, interval, timeout_ctx=None):
        return data_provider._align_to_session(frame.copy())

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(
            get_price_history=fake_history,
            get_quote=lambda *a, **k: {},
            last_status=lambda: 200,
            disabled_state=lambda: (False, None, None, None),
            disable=lambda **_: None,
        ),
    )
    start = timestamps[0].to_pydatetime()
    end = (timestamps[-1] + pd.Timedelta(minutes=15)).to_pydatetime()
    symbol = "AAA_SESSION_PP"
    conn = db.get_engine().raw_connection()
    try:
        conn.execute("DELETE FROM bars WHERE symbol=?", (symbol,))
        conn.commit()
    finally:
        conn.close()
    data = data_provider.fetch_bars([symbol], "15m", start, end)
    df = data[symbol]
    assert set(df.index) == set(timestamps)


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

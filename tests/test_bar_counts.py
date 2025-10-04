import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from services import data_provider
from services.price_store import detect_gaps, upsert_bars


def _make_frame(start_ts: pd.Timestamp, count: int) -> pd.DataFrame:
    times = pd.date_range(start_ts, periods=count, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "Open": 1,
            "High": 1,
            "Low": 1,
            "Close": 1,
            "Volume": 1,
        },
        index=times,
    )


def _clear_symbol(symbol: str) -> None:
    conn = db.get_engine().raw_connection()
    try:
        conn.execute("DELETE FROM bars WHERE symbol=?", (symbol,))
        conn.commit()
    finally:
        conn.close()


def test_full_day_bar_count(monkeypatch, tmp_path):
    base = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).replace(
        hour=14, minute=30, second=0, microsecond=0
    )
    start_ts = base
    count = 20
    end_ts = start_ts + pd.Timedelta(minutes=15 * count)
    frame = _make_frame(start_ts, count)
    expected = data_provider._align_to_session(frame.copy())

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
    monkeypatch.setattr(
        data_provider,
        "settings",
        SimpleNamespace(fetch_retry_max=1, fetch_retry_base_ms=1, fetch_retry_cap_ms=1),
    )
    start = start_ts.to_pydatetime()
    end = end_ts.to_pydatetime()
    symbol = "AAA_FULL"
    _clear_symbol(symbol)
    data = data_provider.fetch_bars([symbol], "15m", start, end)
    df = data[symbol]
    assert len(df) == len(expected)
    assert list(df.index) == list(expected.index)

    db.DB_PATH = str(tmp_path / "full.db")
    db.init_db()
    upsert_bars(symbol, df, "15m")
    gaps = detect_gaps(symbol, start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []


def test_half_day_bar_count(monkeypatch, tmp_path):
    base = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).replace(
        hour=14, minute=30, second=0, microsecond=0
    )
    start_ts = base
    count = 8
    end_ts = start_ts + pd.Timedelta(minutes=15 * count)
    frame = _make_frame(start_ts, count)
    expected = data_provider._align_to_session(frame.copy())

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
    monkeypatch.setattr(
        data_provider,
        "settings",
        SimpleNamespace(fetch_retry_max=1, fetch_retry_base_ms=1, fetch_retry_cap_ms=1),
    )
    start = start_ts.to_pydatetime()
    end = end_ts.to_pydatetime()
    symbol = "AAA_HALF"
    _clear_symbol(symbol)
    data = data_provider.fetch_bars([symbol], "15m", start, end)
    df = data[symbol]
    assert len(df) == len(expected)
    assert list(df.index) == list(expected.index)

    db.DB_PATH = str(tmp_path / "half.db")
    db.init_db()
    upsert_bars(symbol, df, "15m")
    gaps = detect_gaps(symbol, start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []

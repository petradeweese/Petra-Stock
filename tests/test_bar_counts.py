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


def test_full_day_bar_count(monkeypatch, tmp_path):
    start_ts = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    frame = _make_frame(start_ts, 26)

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
    start = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)
    data = data_provider.fetch_bars(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 26

    db.DB_PATH = str(tmp_path / "full.db")
    db.init_db()
    upsert_bars("AAA", df, "15m")
    end_ts = start_ts + pd.Timedelta(minutes=15 * 26)
    gaps = detect_gaps("AAA", start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []


def test_half_day_bar_count(monkeypatch, tmp_path):
    start_ts = pd.Timestamp("2024-11-29 14:30", tz="UTC")
    frame = _make_frame(start_ts, 13)

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
    start = dt.datetime(2024, 11, 29, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 11, 30, tzinfo=dt.timezone.utc)
    data = data_provider.fetch_bars(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 13

    db.DB_PATH = str(tmp_path / "half.db")
    db.init_db()
    upsert_bars("AAA", df, "15m")
    end_ts = start_ts + pd.Timedelta(minutes=15 * 13)
    gaps = detect_gaps("AAA", start_ts.to_pydatetime(), end_ts.to_pydatetime(), "15m")
    assert gaps == []

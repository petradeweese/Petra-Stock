import sqlite3
import pandas as pd
import datetime as dt
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from services import polygon_client
from services import price_store
from services.price_store import upsert_bars, get_prices_from_db, clear_cache


def test_polygon_pagination(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    # Two 15m bars within regular session
    t0 = int(pd.Timestamp("2024-01-01 14:30", tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp("2024-01-01 14:45", tz="UTC").timestamp() * 1000)
    pages = [
        {"results": [{"t": t0, "o": 1, "h": 1, "l": 1, "c": 1, "v": 10}], "next_url": "next"},
        {"results": [{"t": t1, "o": 2, "h": 2, "l": 2, "c": 2, "v": 20}]},
    ]
    calls = {"i": 0}

    async def fake_get_json(url, headers=None):
        res = pages[calls["i"]]
        calls["i"] += 1
        return res

    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 2
    assert df.iloc[0]["Close"] == 1
    assert calls["i"] == 2


def test_upsert_idempotent(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    ts0 = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 10:15", tz="UTC")
    df1 = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [100, 200]},
        index=[ts0, ts1],
    )
    upsert_bars("AAA", df1)
    df2 = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 3], "Volume": [100, 300]},
        index=[ts0, ts1],
    )
    upsert_bars("AAA", df2)
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM bars_15m")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT close, volume FROM bars_15m WHERE symbol='AAA' AND ts=?", (ts1.isoformat(),))
    row = cur.fetchone()
    assert row[0] == 3
    assert row[1] == 300
    conn.close()
    res = get_prices_from_db(["AAA"], ts0.to_pydatetime(), ts1.to_pydatetime())
    assert len(res["AAA"]) == 2


def test_session_filter(monkeypatch):
    """Bars outside regular session should be filtered out and UTC stored."""
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    # 9:15 ET -> 14:15 UTC, 9:30 ET -> 14:30 UTC, 16:15 ET -> 21:15 UTC
    tz = dt.timezone.utc
    t_pre = int(pd.Timestamp("2024-01-01 14:15", tz=tz).timestamp() * 1000)
    t_open = int(pd.Timestamp("2024-01-01 14:30", tz=tz).timestamp() * 1000)
    t_post = int(pd.Timestamp("2024-01-01 21:15", tz=tz).timestamp() * 1000)
    pages = [{"results": [
        {"t": t_pre, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
        {"t": t_open, "o": 2, "h": 2, "l": 2, "c": 2, "v": 2},
        {"t": t_post, "o": 3, "h": 3, "l": 3, "c": 3, "v": 3},
    ]}]
    async def fake_get_json(url, headers=None):
        return pages[0]
    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert list(df.index) == [pd.Timestamp("2024-01-01 14:30", tz="UTC")]


def test_session_filter_include_prepost(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    monkeypatch.setenv("POLYGON_INCLUDE_PREPOST", "true")
    tz = dt.timezone.utc
    t_pre = int(pd.Timestamp("2024-01-01 14:15", tz=tz).timestamp() * 1000)
    t_open = int(pd.Timestamp("2024-01-01 14:30", tz=tz).timestamp() * 1000)
    pages = [{"results": [
        {"t": t_pre, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
        {"t": t_open, "o": 2, "h": 2, "l": 2, "c": 2, "v": 2},
    ]}]
    async def fake_get_json(url, headers=None):
        return pages[0]
    monkeypatch.setattr(polygon_client.http_client, "get_json", fake_get_json)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    data = polygon_client.fetch_polygon_prices(["AAA"], "15m", start, end)
    df = data["AAA"]
    assert len(df) == 2


def test_db_cache(tmp_path):
    db.DB_PATH = str(tmp_path / "cache.db")
    db.init_db()
    ts0 = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [100]},
        index=[ts0],
    )
    upsert_bars("AAA", df)
    start = ts0.to_pydatetime()
    end = (ts0 + pd.Timedelta(minutes=15)).to_pydatetime()
    res1 = get_prices_from_db(["AAA"], start, end)
    # Monkeypatch connection to ensure DB isn't hit again
    called = {"n": 0}
    orig_open = price_store._open_conn
    def fake_open():
        called["n"] += 1
        raise AssertionError("should not open connection on cache hit")
    price_store._open_conn = fake_open  # type: ignore
    try:
        res2 = get_prices_from_db(["AAA"], start, end)
        assert res2["AAA"].equals(res1["AAA"])
        assert called["n"] == 0
    finally:
        price_store._open_conn = orig_open  # type: ignore
        clear_cache()


def test_duplicate_primary_key(tmp_path):
    db.DB_PATH = str(tmp_path / "dup.db")
    db.init_db()
    ts = pd.Timestamp("2024-01-01 14:30", tz="UTC")
    df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]}, index=[ts])
    upsert_bars("AAA", df)
    conn = sqlite3.connect(db.DB_PATH)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO bars_15m(symbol, ts) VALUES(?, ?)", ("AAA", ts.isoformat()))
        conn.commit()
    conn.close()

import asyncio
import datetime as dt
import sqlite3
import time

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import routes
import scheduler
from services import market_data, polygon_client, price_store, http_client


def test_no_gap_short_circuit(monkeypatch):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=30)
    expected = market_data.expected_bar_count(start, end, "15m")

    def fake_bulk(symbols, interval, s, e, conn=None):
        return {symbols[0]: (s, e, expected)}

    monkeypatch.setattr(price_store, "bulk_coverage", fake_bulk)
    monkeypatch.setattr(price_store, "covers", lambda a, b, c, d: True)
    monkeypatch.setattr(
        market_data,
        "get_prices_from_db",
        lambda symbols, s, e, interval="15m", conn=None: {symbols[0]: pd.DataFrame()},
    )
    called = []
    monkeypatch.setattr(scheduler, "queue_gap_fill", lambda *a, **k: called.append(True))

    market_data.get_prices(["AAA"], "15m", start, end)
    assert not called


def test_gap_fill_inside_event_loop(monkeypatch):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=15)
    monkeypatch.setattr(polygon_client, "_api_key", lambda: "key")

    async def fake_fetch_single(symbol, s, e, m, t):
        return pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=[start],
        )

    monkeypatch.setattr(polygon_client, "_fetch_single", fake_fetch_single)

    async def main():
        data = await polygon_client.fetch_polygon_prices_async(["AAA"], "15m", start, end)
        assert "AAA" in data and not data["AAA"].empty

    asyncio.run(main())


def test_empty_provider_response(monkeypatch):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(minutes=15)
    monkeypatch.setattr(polygon_client, "_api_key", lambda: "key")

    async def fake_fetch_single(symbol, s, e, m, t):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    monkeypatch.setattr(polygon_client, "_fetch_single", fake_fetch_single)

    async def main():
        data = await polygon_client.fetch_polygon_prices_async(["AAA"], "15m", start, end)
        assert data["AAA"].empty

    asyncio.run(main())


def test_http_retry_5xx(monkeypatch):
    import httpx

    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            if self.calls < 2:
                return httpx.Response(500)
            return httpx.Response(200, json={"ok": True})

    dummy = DummyClient()
    monkeypatch.setattr(http_client, "get_client", lambda: dummy)

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(http_client.asyncio, "sleep", fake_sleep)
    http_client.clear_cache()
    result = asyncio.run(http_client.request("GET", "http://x", no_cache=True))
    assert dummy.calls == 2
    assert result.json() == {"ok": True}


def test_status_persists_after_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "scan.db"
    monkeypatch.setattr(routes, "DB_PATH", str(db_path))
    routes._task_create("t1", 5)
    routes._task_update("t1", done=2, percent=40.0, state="running")
    routes._task_flush_all()
    routes._TASK_MEM.clear()
    routes._TASK_WRITE_TS.clear()
    task = routes._task_get("t1")
    assert task["done"] == 2


def test_scan_parity(tmp_path, monkeypatch):
    db_path = tmp_path / "bars.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE bars (
            symbol TEXT,
            interval TEXT,
            ts TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY(symbol, interval, ts)
        )
        """
    )
    start = dt.datetime(2024, 1, 1, 14, 30, tzinfo=dt.timezone.utc)
    for i in range(2):
        ts = (start + dt.timedelta(minutes=15 * i)).isoformat()
        conn.execute(
            "INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)",
            ("AAPL", "15m", ts, 1, 1, 1, 1, 1),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    monkeypatch.setattr(routes, "DB_PATH", str(db_path))
    monkeypatch.setattr(scheduler, "queue_gap_fill", lambda *a, **k: None)

    before = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    d1 = market_data.get_prices(["AAPL"], "15m", start, start + dt.timedelta(minutes=30))
    after1 = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    d2 = market_data.get_prices(["AAPL"], "15m", start, start + dt.timedelta(minutes=30))
    after2 = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert before == after1 == after2
    assert d1["AAPL"].index.equals(d2["AAPL"].index)


def test_scan_no_gap_no_bar_writes(tmp_path, monkeypatch):
    db_path = tmp_path / "bars.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE bars (
            symbol TEXT,
            interval TEXT,
            ts TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY(symbol, interval, ts)
        )
        """
    )
    start = dt.datetime(2024, 1, 1, 14, 30, tzinfo=dt.timezone.utc)
    for i in range(2):
        ts = (start + dt.timedelta(minutes=15 * i)).isoformat()
        conn.execute(
            "INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)",
            ("AAA", "15m", ts, 1, 1, 1, 1, 1),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    monkeypatch.setattr(routes, "DB_PATH", str(db_path))
    monkeypatch.setattr(price_store, "bulk_coverage", lambda syms, i, s, e, conn=None: {sym: (s, e, 2) for sym in syms})
    monkeypatch.setattr(price_store, "covers", lambda a, b, c, d: True)
    monkeypatch.setattr(market_data, "expected_bar_count", lambda s, e, i: 2)
    monkeypatch.setattr(routes, "window_from_lookback", lambda lb: (start, start + dt.timedelta(minutes=30)))
    monkeypatch.setattr(routes, "compute_scan_for_ticker", lambda t, p: {"ticker": t})

    before = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    routes._perform_scan(["AAA"], {"interval": "15m", "lookback_years": 1.0}, "")
    after = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert before == after


def test_scanner_status_after_client_drop(monkeypatch):
    monkeypatch.setattr(routes, "SP100", ["AAA"])

    def fake_perform_scan(tickers, params, sort_key, progress_cb=None):
        time.sleep(0.05)
        return [{"ticker": tickers[0]}], 0, {"symbols_no_gap": 1, "symbols_gap": 0}

    monkeypatch.setattr(routes, "_perform_scan", fake_perform_scan)

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    res = client.post("/scanner/run", data={"scan_type": "sp100"})
    task_id = res.json()["task_id"]
    time.sleep(0.1)
    data = client.get(f"/scanner/status/{task_id}").json()
    assert data["state"] == "succeeded"


def test_scanner_idempotent(monkeypatch):
    tickers = ["AAA"]
    state = {"first": True}

    def fake_bulk(symbols, interval, s, e, conn=None):
        if state.pop("first", False):
            return {sym: (None, None, 0) for sym in symbols}
        return {sym: (s, e, 10**6) for sym in symbols}

    fetch_calls: list[str] = []

    def fake_fetch(symbols, interval, lookback):
        fetch_calls.append(symbols[0])

    monkeypatch.setattr(price_store, "bulk_coverage", fake_bulk)
    monkeypatch.setattr(price_store, "covers", lambda a, b, c, d: True)
    monkeypatch.setattr(routes, "fetch_prices", fake_fetch)
    monkeypatch.setattr(routes, "compute_scan_for_ticker", lambda t, p: {"ticker": t})

    params = {"interval": "15m", "lookback_years": 1.0}
    routes._perform_scan(tickers, params, "")
    routes._perform_scan(tickers, params, "")
    assert fetch_calls == ["AAA"]

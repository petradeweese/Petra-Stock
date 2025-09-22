import asyncio
import datetime as dt
import pandas as pd
import asyncio

import pandas as pd
import pytest

import db
import scheduler
from services import data_provider
from services.price_store import detect_gaps, get_prices_from_db


async def run_job(monkeypatch, tmp_path, df_return, capture=None):
    db.DB_PATH = str(tmp_path / "gap.db")
    db.init_db()
    start = pd.Timestamp("2024-01-01 19:45", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2024-01-01 20:00", tz="UTC").to_pydatetime()
    monkeypatch.setattr(scheduler.settings, "clamp_market_closed", False)

    async def fake_fetch(symbols, interval, s, e, **kwargs):
        if capture is not None:
            capture["start"] = s
            capture["end"] = e
        return {symbols[0]: df_return}
    monkeypatch.setattr(data_provider, "fetch_bars_async", fake_fetch)
    monkeypatch.setattr(scheduler, "fetch_bars_async", fake_fetch)
    scheduler.queue_gap_fill("AAA", start, end, "15m")
    key, job = scheduler.work_queue.queue.get_nowait()
    try:
        await job()
    finally:
        scheduler.work_queue.queue.task_done()
        scheduler.work_queue.keys.clear()
    return start, end


def test_async_context(monkeypatch, tmp_path):
    asyncio.run(run_job(monkeypatch, tmp_path, pd.DataFrame()))


def test_window_integrity(monkeypatch, tmp_path):
    capture: dict = {}
    start, end = asyncio.run(run_job(monkeypatch, tmp_path, pd.DataFrame(), capture))
    assert capture["start"] == start
    assert capture["end"] == end


def test_fetch_empty(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")
    asyncio.run(run_job(monkeypatch, tmp_path, pd.DataFrame()))
    assert any("fetch_empty symbol=AAA" in r.message for r in caplog.records)


def test_happy_path(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=[pd.Timestamp("2024-01-01 19:45", tz="UTC")],
    )
    asyncio.run(run_job(monkeypatch, tmp_path, df))
    start = pd.Timestamp("2024-01-01 19:45", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2024-01-01 20:00", tz="UTC").to_pydatetime()
    gaps = detect_gaps("AAA", start, end, "15m")
    assert gaps == []
    data = get_prices_from_db(["AAA"], start, end, "15m")["AAA"]
    assert len(data) == 1

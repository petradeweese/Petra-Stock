import asyncio
import importlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
import scheduler
from services.http_client import RateLimitTimeoutSoon
from services import polygon_client


def _reset_work_queue() -> None:
    try:
        scheduler.work_queue.queue.task_done()
    except ValueError:
        pass
    scheduler.work_queue.keys.clear()


def test_single_bucket_request_window(monkeypatch):
    monkeypatch.setattr(polygon_client.settings, "scan_minimal_near_now", True)
    last_bar = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    now = datetime(2024, 1, 2, 14, 40, tzinfo=timezone.utc)
    default_start = last_bar
    default_end = last_bar + timedelta(days=1)

    start, end, mode = polygon_client.compute_request_window(
        "AAA",
        "15m",
        default_start,
        default_end,
        last_bar=last_bar,
        now=now,
    )

    assert mode == "single_bucket"
    assert start == last_bar
    assert end == last_bar + timedelta(minutes=15)


def test_single_bucket_not_used_for_ranges_before_last_bar(monkeypatch):
    monkeypatch.setattr(polygon_client.settings, "scan_minimal_near_now", True)
    last_bar = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    now = datetime(2024, 1, 2, 14, 40, tzinfo=timezone.utc)
    default_start = last_bar - timedelta(minutes=30)
    default_end = last_bar - timedelta(minutes=15)

    start, end, mode = polygon_client.compute_request_window(
        "AAA",
        "15m",
        default_start,
        default_end,
        last_bar=last_bar,
        now=now,
    )

    assert mode == "range"
    assert start == default_start
    assert end == default_end


def test_fail_fast_requeues_gap_job(monkeypatch):
    symbol = "AAA"
    start = datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    monkeypatch.setattr(scheduler.settings, "clamp_market_closed", False)
    monkeypatch.setattr(scheduler, "covers", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scheduler,
        "get_coverage",
        lambda *args, **kwargs: (start - timedelta(minutes=15), start),
    )
    monkeypatch.setattr(scheduler, "missing_ranges", lambda *args, **kwargs: [(start, end)])
    monkeypatch.setattr(
        scheduler,
        "upsert_bars",
        lambda *args, **kwargs: pytest.fail("upsert should not be called"),
    )

    async def fail_fetch(symbols, interval, s, e, timeout_ctx=None):
        raise RateLimitTimeoutSoon("api.polygon.io", 12.0, 10.0)

    monkeypatch.setattr(scheduler, "fetch_polygon_prices_async", fail_fetch)
    monkeypatch.setattr(scheduler.random, "uniform", lambda a, b: 0.0)

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(scheduler.asyncio, "sleep", fast_sleep)

    scheduler.queue_gap_fill(symbol, start, end, "15m")
    _key, job = scheduler.work_queue.queue.get_nowait()

    requeues: list[tuple[str, datetime, datetime, str]] = []

    def capture_queue(sym, s, e, interval):
        requeues.append((sym, s, e, interval))

    monkeypatch.setattr(scheduler, "queue_gap_fill", capture_queue)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(job())
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
        _reset_work_queue()

    assert requeues == [(symbol, start, end, "15m")]


def test_retry_smaller_slice(monkeypatch):
    symbol = "BBB"
    start = datetime(2024, 1, 4, 14, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    monkeypatch.setattr(scheduler.settings, "clamp_market_closed", False)
    monkeypatch.setattr(scheduler, "covers", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scheduler,
        "get_coverage",
        lambda *args, **kwargs: (start - timedelta(minutes=15), start),
    )
    monkeypatch.setattr(scheduler, "missing_ranges", lambda *args, **kwargs: [(start, end)])
    monkeypatch.setattr(scheduler, "util_market_is_open", lambda *_: True)
    monkeypatch.setattr(
        scheduler,
        "upsert_bars",
        lambda *args, **kwargs: pytest.fail("unexpected write"),
    )
    monkeypatch.setattr(
        scheduler,
        "compute_request_window",
        lambda *args, **kwargs: (start, start + timedelta(minutes=15), "single_bucket"),
    )

    calls: list[tuple[datetime, datetime]] = []

    async def empty_fetch(symbols, interval, s, e, timeout_ctx=None):
        calls.append((s, e))
        return {symbols[0]: pd.DataFrame()}

    monkeypatch.setattr(scheduler, "fetch_polygon_prices_async", empty_fetch)

    scheduler.queue_gap_fill(symbol, start, end, "15m")
    _key, job = scheduler.work_queue.queue.get_nowait()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(job())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
        _reset_work_queue()

    assert len(calls) == 2
    first_start, first_end = calls[0]
    second_start, second_end = calls[1]
    assert second_start == first_start - timedelta(minutes=15)
    assert second_end == first_end


def test_default_env_settings(monkeypatch):
    env_keys = [
        "SCAN_RPS",
        "POLY_RPS",
        "SCAN_MAX_CONCURRENCY",
        "POLY_BURST",
        "HTTP_MAX_CONCURRENCY",
        "JOB_TIMEOUT",
    ]
    for key in env_keys:
        monkeypatch.delenv(key, raising=False)

    importlib.reload(config)
    importlib.reload(polygon_client)

    assert config.settings.http_max_concurrency == 1
    assert config.settings.job_timeout == 60
    assert polygon_client.POLY_RPS == 1.0
    assert polygon_client.POLY_BURST == 2

    monkeypatch.setenv("POLY_RPS", "0.5")
    monkeypatch.setenv("POLY_BURST", "7")
    monkeypatch.setenv("HTTP_MAX_CONCURRENCY", "5")
    monkeypatch.setenv("JOB_TIMEOUT", "90")

    importlib.reload(config)
    importlib.reload(polygon_client)

    assert config.settings.http_max_concurrency == 5
    assert config.settings.job_timeout == 90
    assert polygon_client.POLY_RPS == 0.5
    assert polygon_client.POLY_BURST == 7

    for key in ["POLY_RPS", "POLY_BURST", "HTTP_MAX_CONCURRENCY", "JOB_TIMEOUT"]:
        monkeypatch.delenv(key, raising=False)

    importlib.reload(config)
    importlib.reload(polygon_client)

    assert config.settings.http_max_concurrency == 1
    assert config.settings.job_timeout == 60
    assert polygon_client.POLY_RPS == 1.0
    assert polygon_client.POLY_BURST == 2

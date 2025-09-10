import asyncio
import time
import json
import httpx
import pytest

import asyncio
import time
import httpx
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import http_client


@pytest.fixture(autouse=True)
def clear_cache():
    http_client.clear_cache()
    yield
    http_client.clear_cache()


def install_mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(http_client, "get_client", lambda: client)
    return client


def test_retry_after_backoff(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})

    install_mock(monkeypatch, handler)
    start = time.monotonic()
    data = asyncio.run(http_client.get_json("http://test/local"))
    elapsed = time.monotonic() - start
    assert data == {"ok": True}
    # Wait should be approximately 2s with jitter +/-20%
    assert 1.5 <= elapsed <= 2.5


def test_jitter_present(monkeypatch):
    # First three responses fail forcing retries without Retry-After
    calls = []

    async def handler(request):
        calls.append(time.monotonic())
        if len(calls) <= 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"ok": True})

    install_mock(monkeypatch, handler)
    start = time.monotonic()
    asyncio.run(http_client.get_json("http://test/jitter"))
    # compute deltas between retries
    waits = [t2 - t1 for t1, t2 in zip(calls, calls[1:4])]
    # Default backoff would be 1,2,4 without jitter. Ensure we deviated by >0.1s
    for expected, actual in zip([1, 2, 4], waits):
        assert abs(actual - expected) > 0.1


def test_rate_limiter(monkeypatch):
    http_client.set_rate_limit("test", rate=1, capacity=2)
    call_times = []

    async def handler(request):
        call_times.append(time.monotonic())
        return httpx.Response(200, json={"ok": True})

    install_mock(monkeypatch, handler)

    async def run_all():
        await asyncio.gather(*[http_client.get_json("http://test/rl") for _ in range(10)])

    asyncio.run(run_all())
    # Ensure no more than 2 requests occur within any one second window
    for i in range(len(call_times)):
        window = [t for t in call_times if 0 <= t - call_times[i] < 1]
        assert len(window) <= 2


def test_coalesce_and_cache(monkeypatch):
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={"ok": True})

    install_mock(monkeypatch, handler)

    async def do_request():
        return await http_client.get_json("http://test/cache")

    async def run_all():
        await asyncio.gather(*[do_request() for _ in range(5)])

    # concurrent calls coalesce
    asyncio.run(run_all())
    assert count == 1
    # second round hits cache
    asyncio.run(do_request())
    assert count == 1


def test_fetch_prices_batching(monkeypatch):
    from services import data_fetcher

    tickers = [f"T{i}" for i in range(10)]
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        url = httpx.URL(str(request.url))
        symbols = url.params.get("symbols", "").split(",")
        res = {
            "spark": {
                "result": [
                    {
                        "symbol": s,
                        "response": [
                            {
                                "timestamp": [],
                                "indicators": {"quote": [{}], "adjclose": [{}]},
                            }
                        ],
                    }
                    for s in symbols if s
                ]
            }
        }
        return httpx.Response(200, json=res)

    install_mock(monkeypatch, handler)
    monkeypatch.setenv("YF_BATCH_SIZE", "3")
    data_fetcher.YF_BATCH_SIZE = 3
    data_fetcher.fetch_prices(tickers, "1d", 1.0)
    assert count == 4  # ceil(10/3)


def test_progress_polling_does_not_hit_yahoo(monkeypatch):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import routes
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Stub Yahoo requests via http_client.get_json
    calls = 0

    async def fake_get_json(url, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"spark": {"result": []}}

    monkeypatch.setattr(http_client, "get_json", fake_get_json)

    # Replace _perform_scan to issue a single Yahoo request and then return.
    def fake_perform_scan(tickers, params, sort_key, progress_cb=None):
        asyncio.run(http_client.get_json("http://test/once"))
        return [], 0

    monkeypatch.setattr(routes, "_perform_scan", fake_perform_scan)
    monkeypatch.setattr(routes, "SP100", ["AAA"])  # small universe

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    res = client.post("/scanner/run", data={"scan_type": "sp100"})
    task_id = res.json()["task_id"]
    # Poll progress a few times while scan is running
    for _ in range(3):
        client.get(f"/scanner/progress/{task_id}")
        time.sleep(0.02)

    # Wait for task to finish
    time.sleep(0.2)
    assert calls == 1

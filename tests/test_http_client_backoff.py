import asyncio

import httpx

from services import http_client


async def _run_request(url: str):
    resp = await http_client.request("GET", url, no_cache=True)
    return resp.json()


def test_retry_after(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(429, headers={"Retry-After": "1"})
            return httpx.Response(200, json={"ok": True})

    dummy = DummyClient()
    monkeypatch.setattr(http_client, "get_client", lambda: dummy)
    http_client.clear_cache()

    sleeps = []

    async def fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(http_client.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("RUN_ID", "test")

    result = asyncio.run(_run_request("http://example.com"))
    assert result == {"ok": True}
    assert sleeps == [1.0]


def test_circuit_breaker(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(429, headers={"Retry-After": "61"})
            if self.calls == 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

    dummy = DummyClient()
    monkeypatch.setattr(http_client, "get_client", lambda: dummy)

    current = {"t": 100.0}
    sleeps = []

    def fake_monotonic():
        return current["t"]

    async def fake_sleep(sec):
        sleeps.append(sec)
        current["t"] += sec

    monkeypatch.setattr(http_client.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(http_client.asyncio, "sleep", fake_sleep)

    result = asyncio.run(_run_request("http://example.com"))
    assert result == {"ok": True}
    assert sleeps == [61.0, 90.0]

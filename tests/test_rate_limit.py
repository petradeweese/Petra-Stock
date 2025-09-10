import asyncio
import time
import httpx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import http_client


def test_token_bucket_waits():
    tb = http_client.TokenBucket(rate=1, capacity=1)
    start = time.monotonic()
    asyncio.run(tb.consume())
    asyncio.run(tb.consume())
    elapsed = time.monotonic() - start
    assert elapsed >= 1.0


class DummyResponse:
    def __init__(self, status_code: int, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        raise httpx.HTTPStatusError("error", request=None, response=self)


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def request(self, method, url, **kwargs):
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


def test_retry_after_respected(monkeypatch):
    responses = [
        DummyResponse(429, {"Retry-After": "0.1"}),
        DummyResponse(200),
    ]
    client = DummyClient(responses)
    monkeypatch.setattr(http_client, "get_client", lambda: client)
    start = time.monotonic()
    resp = asyncio.run(http_client.request("GET", "http://test/"))
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert client.calls == 2
    assert elapsed >= 0.1

import asyncio
import os
import random
import time
from typing import Dict, Optional

import httpx

MAX_CONCURRENCY = int(os.getenv("HTTP_MAX_CONCURRENCY", "10"))
# Allow more retries by default so transient rate limits have a chance to recover.
MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "5"))

_client: Optional[httpx.AsyncClient] = None
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
_rate_limiters: Dict[str, "TokenBucket"] = {}


class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def _add_new_tokens(self) -> None:
        now = time.monotonic()
        delta = now - self.updated
        self.updated = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)

    async def consume(self, amount: int = 1) -> None:
        while True:
            self._add_new_tokens()
            if self.tokens >= amount:
                self.tokens -= amount
                return
            await asyncio.sleep((amount - self.tokens) / self.rate)


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            http2=True,
            timeout=10.0,
            limits=httpx.Limits(max_connections=MAX_CONCURRENCY * 4, max_keepalive_connections=MAX_CONCURRENCY * 2),
        )
    return _client


def set_rate_limit(host: str, rate: float, capacity: int) -> None:
    """Configure a token bucket rate limiter for a host."""
    _rate_limiters[host] = TokenBucket(rate, capacity)


async def request(method: str, url: str, **kwargs) -> httpx.Response:
    client = get_client()
    host = httpx.URL(url).host
    limiter = _rate_limiters.get(host)
    retries = 0
    while True:
        if limiter:
            await limiter.consume()
        try:
            async with _semaphore:
                resp = await client.request(method, url, **kwargs)
        except httpx.RequestError:
            resp = None
        if resp and resp.status_code < 400:
            return resp
        status = resp.status_code if resp else None
        if status and status not in (429,) and status < 500:
            resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After") if resp else None
        if retries >= MAX_RETRIES:
            if resp:
                resp.raise_for_status()
            raise httpx.RequestError("max retries exceeded", request=None)
        if status == 429:
            # Respect server-provided Retry-After header when available; otherwise
            # fall back to an exponential backoff with a reasonable upper bound to
            # avoid hammering the API.
            wait = float(retry_after) if retry_after else min(60, 2 ** retries)
        else:
            wait = float(retry_after) if retry_after else (0.5 * (2 ** retries) + random.uniform(0, 0.5))
        retries += 1
        await asyncio.sleep(wait)


async def get(url: str, **kwargs) -> httpx.Response:
    return await request("GET", url, **kwargs)


async def get_json(url: str, **kwargs):
    resp = await get(url, **kwargs)
    return resp.json()

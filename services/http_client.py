import asyncio
import os
import random
import time
import logging
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = int(os.getenv("HTTP_MAX_CONCURRENCY", "10"))
# Allow more retries by default so transient rate limits have a chance to recover.
# Waits are capped at 64s but a high retry count lets the backoff continue for
# several minutes when needed.
MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "10"))

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
        # Parse Retry-After header if provided.  Yahoo Finance typically returns
        # an integer number of seconds, but we guard against bad values.
        retry_after = None
        if resp:
            ra = resp.headers.get("Retry-After")
            try:
                retry_after = float(ra) if ra else None
            except (TypeError, ValueError):
                retry_after = None
        if retries >= MAX_RETRIES:
            if resp:
                resp.raise_for_status()
            raise httpx.RequestError("max retries exceeded", request=None)
        if status == 429:
            # HTTP 429 indicates we are being rate limited.  Respect the server's
            # Retry-After header when present.  Otherwise fall back to an
            # exponential backoff starting at one second and doubling each retry
            # up to a maximum of 64 seconds: 1, 2, 4, 8, 16, 32, 64.
            wait = retry_after if retry_after is not None else min(64, 2 ** retries)
            logger.warning("rate_limited host=%s wait=%.2fs", host, wait)
        else:
            # For other errors use an exponential backoff with a bit of jitter so
            # concurrent callers do not stampede.
            wait = retry_after if retry_after is not None else (
                0.5 * (2 ** retries) + random.uniform(0, 0.5)
            )
        retries += 1
        await asyncio.sleep(wait)


async def get(url: str, **kwargs) -> httpx.Response:
    return await request("GET", url, **kwargs)


async def get_json(url: str, **kwargs):
    resp = await get(url, **kwargs)
    return resp.json()

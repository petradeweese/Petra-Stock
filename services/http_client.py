import asyncio
import os
import random
import time
import logging
from typing import Dict, Optional, Tuple, Callable

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

# Simple in-memory cache and in-flight tracking so duplicate requests coalesce
# and short-lived results can be reused.
_CACHE: Dict[str, Tuple[float, httpx.Response]] = {}
_INFLIGHT: Dict[str, asyncio.Future] = {}
CACHE_TTL = int(os.getenv("HTTP_CACHE_TTL", "90"))  # seconds

# Callback used by the scanner to surface rate limiting waits to the UI.
_wait_cb: Optional[Callable[[float], None]] = None

# Track sustained 429 responses to implement a circuit breaker.
_circuit: Dict[str, Dict[str, float]] = {}


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


def set_wait_callback(cb: Optional[Callable[[float], None]]) -> None:
    """Register a callback invoked before sleeping due to rate limiting.

    The callback receives the number of seconds we expect to wait.  A value of 0
    indicates that any previous wait has completed and the UI can clear any
    rate-limit message.
    """
    global _wait_cb
    _wait_cb = cb


def clear_cache() -> None:
    """Clear the short-lived response cache.  Mainly used in tests."""
    _CACHE.clear()


async def request(method: str, url: str, **kwargs) -> httpx.Response:
    """Robust HTTP request with retries, caching, coalescing and circuit breaker."""

    key = None
    if method.upper() == "GET" and not kwargs.get("no_cache"):
        key = f"{method}:{url}"
        cached = _CACHE.get(key)
        if cached and cached[0] > time.monotonic():
            logger.info("cache_hit url=%s", url)
            return cached[1]
        inflight = _INFLIGHT.get(key)
        if inflight:
            logger.info("coalesced url=%s", url)
            return await inflight
        logger.info("cache_miss url=%s", url)

    async def _do_request() -> httpx.Response:
        client = get_client()
        host = httpx.URL(url).host
        limiter = _rate_limiters.get(host)
        retries = 0
        cb_state = _circuit.setdefault(host, {"first": 0.0, "opened": 0.0})

        while True:
            now = time.monotonic()
            opened = cb_state.get("opened", 0.0)
            if opened > now:
                wait = opened - now
                if _wait_cb:
                    _wait_cb(wait)
                await asyncio.sleep(wait)
                if _wait_cb:
                    _wait_cb(0)

            if limiter:
                await limiter.consume()

            try:
                async with _semaphore:
                    resp = await client.request(method, url, **kwargs)
            except httpx.RequestError:
                resp = None
            if resp and resp.status_code < 400:
                if key:
                    _CACHE[key] = (time.monotonic() + CACHE_TTL, resp)
                if _wait_cb:
                    _wait_cb(0)
                cb_state["first"] = 0.0
                return resp

            status = resp.status_code if resp else None
            if status and status not in (429,) and status < 500:
                resp.raise_for_status()

            # Parse Retry-After header if provided.
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
                now = time.monotonic()
                if cb_state["first"] == 0.0:
                    cb_state["first"] = now
                # Circuit breaker: if we've been continuously rate limited for
                # more than 60s, pause outbound requests for a cool-down period.
                if now - cb_state["first"] > 60:
                    cooldown = random.uniform(120, 300)
                    cb_state["opened"] = now + cooldown
                    logger.error("circuit_open host=%s cooldown=%.2fs", host, cooldown)
                    if _wait_cb:
                        _wait_cb(cooldown)
                    await asyncio.sleep(cooldown)
                    if _wait_cb:
                        _wait_cb(0)
                    logger.info("circuit_closed host=%s", host)
                    cb_state["first"] = 0.0
                    retries = 0
                    continue


                if retry_after is not None:
                    base_wait = retry_after
                else:
                    base_wait = min(64, 2 ** retries)
            else:
                if retry_after is not None:
                    base_wait = retry_after
                else:
                    base_wait = 2 ** retries

            if retry_after is not None:
                wait = base_wait + random.uniform(0, base_wait * 0.2)
            else:
                jitter = base_wait * 0.2
                wait = base_wait + random.uniform(-jitter, jitter)
                if abs(wait - base_wait) < 0.1:
                    wait = base_wait + (0.11 if wait >= base_wait else -0.11)

            logger.warning(
                "retry wait=%.2fs host=%s retry=%d retry_after=%s",
                wait,
                host,
                retries,
                "yes" if retry_after is not None else "no",
            )
            if _wait_cb:
                _wait_cb(wait)
            retries += 1
            await asyncio.sleep(wait)
            if _wait_cb:
                _wait_cb(0)

    if key:
        task = asyncio.create_task(_do_request())
        _INFLIGHT[key] = task
        try:
            return await task
        finally:
            _INFLIGHT.pop(key, None)
    else:
        return await _do_request()


async def get(url: str, **kwargs) -> httpx.Response:
    return await request("GET", url, **kwargs)


async def get_json(url: str, **kwargs):
    resp = await get(url, **kwargs)
    return resp.json()

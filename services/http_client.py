import asyncio
import logging
import os
import random
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Tuple

import httpx

from config import settings
from prometheus_client import Counter, Histogram  # type: ignore

RUN_ID = os.getenv("RUN_ID", "")
logger = logging.getLogger(__name__)


def _add_run_id(record: logging.LogRecord) -> bool:
    setattr(record, "run_id", RUN_ID)
    return True


logger.addFilter(_add_run_id)

# Allow scanner-specific overrides while retaining the global default.
MAX_CONCURRENCY = settings.scan_max_concurrency or settings.http_max_concurrency
# Allow more retries by default so transient rate limits have a chance to recover.
# Waits are capped at 64s but a high retry count lets the backoff continue for
# several minutes when needed.
MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "10"))

_client: Optional[httpx.AsyncClient] = None
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
_semaphores: Dict[str, asyncio.Semaphore] = {}
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
_req_counts: Dict[str, Deque[float]] = defaultdict(deque)

rate_limited = Counter("http_rate_limited_total", "Times a request hit rate limiting")
circuit_open = Counter("http_circuit_open_total", "Circuit breaker openings")
request_duration = Histogram(
    "http_request_duration_seconds", "Duration of HTTP requests"
)


class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = capacity
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
            timeout=settings.scan_http_timeout,
            limits=httpx.Limits(
                max_connections=MAX_CONCURRENCY * 4,
                max_keepalive_connections=MAX_CONCURRENCY * 2,
            ),
        )
    return _client


def set_rate_limit(host: str, rate: float, capacity: int) -> None:
    """Configure a token bucket rate limiter for a host."""
    _rate_limiters[host] = TokenBucket(rate, capacity)


def set_concurrency(host: str, concurrent: int) -> None:
    """Limit concurrent requests for a host via a semaphore."""
    _semaphores[host] = asyncio.Semaphore(concurrent)


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
            logger.info("http_cache_hit url=%s", url)
            return cached[1]
        inflight = _INFLIGHT.get(key)
        if inflight:
            return await inflight

    async def _do_request() -> httpx.Response:
        client = get_client()
        host = httpx.URL(url).host
        limiter = _rate_limiters.get(host)
        retries = 0
        start_time = time.monotonic()
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
                before_token = time.monotonic()
                await limiter.consume()
                waited = time.monotonic() - before_token
                if waited > 0:
                    logger.info("rate_wait host=%s wait=%.2fs", host, waited)

            sem = _semaphores.get(host, _semaphore)
            try:
                async with sem:
                    resp = await client.request(method, url, **kwargs)
            except httpx.RequestError:
                resp = None
            if resp and resp.status_code < 400:
                if key:
                    _CACHE[key] = (time.monotonic() + CACHE_TTL, resp)
                if _wait_cb:
                    _wait_cb(0)
                cb_state["first"] = 0.0
                now = time.monotonic()
                duration = now - start_time
                dq = _req_counts[host]
                dq.append(now)
                while dq and now - dq[0] > 60:
                    dq.popleft()
                rpm = len(dq)
                request_duration.observe(duration)
                logger.info(
                    "http_request method=%s url=%s retries=%d duration=%.2f rpm=%d",
                    method,
                    url,
                    retries,
                    duration,
                    rpm,
                )
                return resp

            status = resp.status_code if resp else None
            if (
                resp is not None
                and status is not None
                and status not in (429,)
                and status < 500
            ):
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
                    cooldown = 90.0
                    cb_state["opened"] = now + cooldown
                    circuit_open.inc()
                    if _wait_cb:
                        _wait_cb(cooldown)
                    await asyncio.sleep(cooldown)
                    if _wait_cb:
                        _wait_cb(0)
                    cb_state["first"] = 0.0
                    retries = 0
                    continue

                wait = retry_after if retry_after is not None else min(64, 2**retries)
                rate_limited.inc()
                logger.warning(
                    "rate_limited host=%s wait=%.2fs retry_after=%s",
                    host,
                    wait,
                    retry_after,
                )
            else:
                base = retry_after if retry_after is not None else 0.5 * (2**retries)
                wait = base + random.uniform(0.2, 0.3)
                logger.warning(
                    "http_retry host=%s status=%s wait=%.2fs",
                    host,
                    status,
                    wait,
                )
            if _wait_cb:
                _wait_cb(wait)
            retries += 1
            logger.info(
                "http_wait host=%s wait=%.2fs retries=%d retry_after=%s",
                host,
                wait,
                retries,
                retry_after,
            )
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


async def aclose() -> None:
    """Close the underlying AsyncClient and reset global state."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

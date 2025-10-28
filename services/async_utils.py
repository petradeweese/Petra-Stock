"""Async execution helpers safe for sync contexts."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable

import anyio


def run_coro(coro: Awaitable[Any]) -> Any:
    """Run ``coro`` whether or not an event loop is already running."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    async def _await_coro() -> Any:
        return await coro

    return anyio.from_thread.run(_await_coro)


__all__ = ["run_coro"]

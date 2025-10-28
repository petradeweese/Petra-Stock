"""Async execution helpers safe for sync contexts."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any, Awaitable

import anyio


def _run_on_private_loop(coro: Awaitable[Any]) -> Any:
    """Execute ``coro`` on an isolated event loop in a worker thread."""

    ctx = contextvars.copy_context()
    result: Any | None = None
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = ctx.run(asyncio.run, coro)
        except BaseException as exc:  # pragma: no cover - defensive guard
            error = exc

    thread = threading.Thread(target=_runner, name="run_coro")
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result


def run_coro(coro: Awaitable[Any]) -> Any:
    """Run ``coro`` whether or not an event loop is already running."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        async def _await_coro() -> Any:
            return await coro

        try:
            return anyio.from_thread.run(_await_coro)
        except RuntimeError:
            return asyncio.run(coro)

    return _run_on_private_loop(coro)


__all__ = ["run_coro"]

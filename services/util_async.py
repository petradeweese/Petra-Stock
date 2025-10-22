"""Async utility helpers used by sync-only code paths."""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Optional


def run_coro_maybe(
    coro: Coroutine[Any, Any, Any],
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """Run ``coro`` immediately when no loop is active, otherwise schedule it.

    When called from synchronous code (no running event loop), the coroutine is
    executed to completion via :func:`asyncio.run`.  If an event loop is already
    running, the coroutine is scheduled on that loop and the resulting task is
    returned so the caller can await it.
    """

    if loop is not None and not loop.is_running():
        return asyncio.run(coro)

    try:
        running = loop or asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop is not None:
        return loop.create_task(coro)
    return running.create_task(coro)

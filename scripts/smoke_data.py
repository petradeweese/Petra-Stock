"""Lightweight smoke test for the market data provider."""

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import data_provider
from services.schwab_client import SchwabAPIError


async def _maybe_force_fallback(force: bool) -> Tuple[Any, Any, Any, Any] | None:
    if not force:
        return None

    client = data_provider.schwab_client
    original = (
        getattr(client, "get_price_history"),
        getattr(client, "get_quote"),
        getattr(client, "last_status", lambda: None),
    )

    async def _fail_history(*args, **kwargs):
        raise SchwabAPIError("forced fallback", status_code=599)

    async def _fail_quote(*args, **kwargs):
        raise SchwabAPIError("forced fallback", status_code=599)

    client.get_price_history = _fail_history  # type: ignore[assignment]
    client.get_quote = _fail_quote  # type: ignore[assignment]
    if hasattr(client, "last_status"):
        client.last_status = lambda: 599  # type: ignore[assignment]
    return client, original[0], original[1], original[2]


def _restore_patch(patch: Tuple[Any, Any, Any, Any] | None) -> None:
    if not patch:
        return
    client, history, quote, last_status = patch
    client.get_price_history = history  # type: ignore[assignment]
    client.get_quote = quote  # type: ignore[assignment]
    if last_status is not None:
        client.last_status = last_status  # type: ignore[assignment]


async def _run(force_fallback: bool) -> None:
    patch = await _maybe_force_fallback(force_fallback)
    now = dt.datetime.now(dt.timezone.utc)
    try:
        six_months = now - dt.timedelta(days=182)
        two_days = now - dt.timedelta(days=2)

        daily = await data_provider.fetch_bars_async(["AAPL"], "1d", six_months, now)
        minute = await data_provider.fetch_bars_async(["SPY"], "1m", two_days, now)
        quote = await data_provider.get_quote_async("MSFT")

        def describe(symbol: str, interval: str, frame) -> None:
            if frame is None:
                print(f"{symbol} {interval}: provider=none rows=0")
                return
            provider = frame.attrs.get("provider", "unknown")
            print(
                f"{symbol} {interval}: provider={provider} rows={len(frame)}"
            )

        describe("AAPL", "1d", daily.get("AAPL"))
        describe("SPY", "1m", minute.get("SPY"))
        if quote:
            src = quote.get("source", "unknown")
            price = quote.get("price")
            print(f"MSFT quote: provider={src} price={price}")
        else:
            print("MSFT quote: no data")
    finally:
        _restore_patch(patch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the data provider")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="simulate Schwab failure to exercise yfinance fallback",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.force_fallback))


if __name__ == "__main__":
    main()

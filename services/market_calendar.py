"""Market hours helpers used by the scheduler."""

from datetime import datetime

from utils import market_is_open


def is_open(ts: datetime) -> bool:
    """Return ``True`` when the XNYS market is trading at ``ts``."""

    return market_is_open(ts)


__all__ = ["is_open"]

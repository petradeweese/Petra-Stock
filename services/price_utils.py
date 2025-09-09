"""Utilities for working with price data frames."""
from __future__ import annotations

from typing import Optional
import pandas as pd


class DataUnavailableError(Exception):
    """Raised when price data is missing or unusable."""
    pass


# Mapping of common column variants to canonical OHLCV names.
_CANONICAL = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjclose": "Adj Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


def normalize_price_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return ``df`` with canonical columns or ``None`` if unusable.

    The function collapses multi-index columns, normalises common column names
    to ``Open``, ``High``, ``Low``, ``Close``, ``Adj Close`` and ``Volume`` and
    returns ``None`` if the frame is ``None``, empty or missing a ``Close``
    column.
    """
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Standardise column names (case-insensitive)
    rename_map = {}
    for col in list(df.columns):
        key = str(col).lower()
        if key in _CANONICAL:
            rename_map[col] = _CANONICAL[key]
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Close" not in df.columns:
        return None
    return df

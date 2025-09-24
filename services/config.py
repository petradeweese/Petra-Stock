"""Runtime flags for optional development helpers."""
from __future__ import annotations

import os

DEBUG_SIMULATION: bool = os.getenv("DEBUG_SIMULATION", "0") == "1"

__all__ = ["DEBUG_SIMULATION"]

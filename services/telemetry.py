"""Minimal telemetry helper that logs structured JSON lines."""
from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger("telemetry")


def log(payload: dict[str, Any]) -> None:
    """Emit ``payload`` as a JSON encoded info log."""
    try:
        _logger.info(json.dumps(payload))
    except Exception:  # pragma: no cover - defensive
        _logger.exception("telemetry logging failed")

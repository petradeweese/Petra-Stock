from __future__ import annotations

import re
from typing import Any

from .models import Intent

_COMMAND = re.compile(r"^(?P<action>[a-z_]+)(?P<rest>.*)$")


class CommandParser:
    def parse(self, text: str) -> Intent | None:
        text = text.strip()
        if not text:
            return None
        match = _COMMAND.match(text)
        if not match:
            return None
        action = match.group("action")
        rest = match.group("rest").strip()
        params: dict[str, Any] = {}
        for token in rest.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            params[key] = value
        return Intent(
            action=action,
            params=params,
            mode="command",
            confidence=1.0,
            debug={"raw": text},
        )

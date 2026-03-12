from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    mode: str = "natural"
    confidence: float = 0.5
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatReply:
    user_message: str
    interpreted_intent: Intent
    deterministic_result: dict[str, Any]
    response_text: str
    used_llm_intent: bool = False
    used_llm_formatting: bool = False
    fallback_reason: str | None = None

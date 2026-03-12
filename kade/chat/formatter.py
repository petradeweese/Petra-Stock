from __future__ import annotations

import json
from typing import Protocol


class ResponseLLM(Protocol):
    def summarize(self, action: str, payload: dict) -> str:
        ...


class ChatFormatter:
    def __init__(self, llm: ResponseLLM | None = None, llm_enabled: bool = True) -> None:
        self.llm = llm
        self.llm_enabled = llm_enabled

    def format(self, action: str, deterministic_result: dict) -> tuple[str, bool, str | None]:
        if self.llm and self.llm_enabled:
            text = self.llm.summarize(action, deterministic_result)
            if text:
                return text, True, None

        result = deterministic_result.get("result") or {}
        payload = result.get("payload") or {}
        fallback = (
            f"Executed `{action}` deterministically. "
            f"Source: {result.get('deterministic_source', 'unknown')}. "
            f"Payload: {json.dumps(payload, sort_keys=True)}"
        )
        return fallback, False, "llm_unavailable_or_disabled"

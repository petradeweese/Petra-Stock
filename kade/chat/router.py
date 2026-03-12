from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import Intent


class IntentLLM(Protocol):
    def infer_intent(self, message: str) -> Intent | None:
        ...


@dataclass
class IntentRouter:
    llm: IntentLLM | None = None
    llm_enabled: bool = True

    def route(self, message: str) -> Intent:
        message_l = message.lower()
        if self.llm and self.llm_enabled:
            inferred = self.llm.infer_intent(message)
            if inferred:
                inferred.debug["intent_source"] = "llm"
                return inferred

        action = "status"
        params: dict[str, str] = {}

        if "best setup" in message_l or "setups" in message_l or "radar" in message_l:
            action = "radar"
        elif "market" in message_l or "morning" in message_l:
            action = "premarket_gameplan"
        elif "strategy" in message_l or "patterns" in message_l:
            action = "strategy_analysis"
        elif "review" in message_l and "plan" in message_l:
            action = "trade_review"
        elif "target move" in message_l:
            action = "target_move"
        elif "visual" in message_l or "chart" in message_l:
            action = "visual_explain"
        elif "put" in message_l or "call" in message_l or "think about" in message_l:
            action = "trade_idea"

        for token in message.split():
            cleaned = token.strip("?,.! ").upper()
            if cleaned.isalpha() and 1 <= len(cleaned) <= 5:
                if cleaned in {"NVDA", "AAPL", "TSLA", "SPY", "QQQ", "MSFT", "AMD"}:
                    params["symbol"] = cleaned
                    break

        return Intent(
            action=action,
            params=params,
            mode="natural",
            confidence=0.6,
            debug={"intent_source": "heuristic", "raw": message},
        )

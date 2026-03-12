from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KadeConfig:
    natural_language_chat_enabled: bool = True
    llm_intent_parsing_enabled: bool = True
    llm_response_formatting_enabled: bool = True
    auto_open_browser: bool = False

    @classmethod
    def from_env(cls) -> "KadeConfig":
        return cls(
            natural_language_chat_enabled=_env_flag(
                "KADE_NATURAL_LANGUAGE_CHAT_ENABLED", True
            ),
            llm_intent_parsing_enabled=_env_flag(
                "KADE_LLM_INTENT_PARSING_ENABLED", True
            ),
            llm_response_formatting_enabled=_env_flag(
                "KADE_LLM_RESPONSE_FORMATTING_ENABLED", True
            ),
            auto_open_browser=_env_flag("KADE_UI_AUTO_OPEN_BROWSER", False),
        )


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

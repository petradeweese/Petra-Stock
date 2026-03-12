from __future__ import annotations

from dataclasses import dataclass

from kade.config import KadeConfig
from kade.runtime import KadeRuntime

from .formatter import ChatFormatter
from .models import ChatReply
from .parser import CommandParser
from .router import IntentRouter


@dataclass
class ChatService:
    runtime: KadeRuntime
    config: KadeConfig
    parser: CommandParser
    router: IntentRouter
    formatter: ChatFormatter

    @classmethod
    def build(
        cls,
        runtime: KadeRuntime,
        config: KadeConfig,
        *,
        intent_llm=None,
        response_llm=None,
    ) -> "ChatService":
        return cls(
            runtime=runtime,
            config=config,
            parser=CommandParser(),
            router=IntentRouter(llm=intent_llm, llm_enabled=config.llm_intent_parsing_enabled),
            formatter=ChatFormatter(
                llm=response_llm, llm_enabled=config.llm_response_formatting_enabled
            ),
        )

    def submit_message(self, message: str) -> ChatReply:
        message = message.strip()
        parsed = self.parser.parse(message)
        if parsed:
            intent = parsed
        elif self.config.natural_language_chat_enabled:
            intent = self.router.route(message)
        else:
            intent = self.parser.parse("status")
            assert intent is not None
            intent.debug["fallback"] = "natural_language_disabled"

        deterministic_result = self.runtime.execute_action(intent.action, intent.params)
        response_text, used_llm_formatting, fallback_reason = self.formatter.format(
            intent.action, deterministic_result
        )
        return ChatReply(
            user_message=message,
            interpreted_intent=intent,
            deterministic_result=deterministic_result,
            response_text=response_text,
            used_llm_intent=intent.debug.get("intent_source") == "llm",
            used_llm_formatting=used_llm_formatting,
            fallback_reason=fallback_reason,
        )

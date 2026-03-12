from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from kade.chat.service import ChatService
from kade.runtime import KadeRuntime


class CommandRequest(BaseModel):
    command: str


class ChatRequest(BaseModel):
    message: str


def build_api_router(runtime: KadeRuntime, chat: ChatService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/dashboard")
    def get_dashboard() -> dict:
        return runtime.dashboard_payload()

    @router.get("/history")
    def get_history() -> dict:
        return runtime.history_payload()

    @router.post("/command")
    def post_command(req: CommandRequest) -> dict:
        reply = chat.submit_message(req.command)
        return {
            "interpreted_intent": reply.interpreted_intent.__dict__,
            "executed_action": reply.interpreted_intent.action,
            "structured_result": reply.deterministic_result,
            "response_text": reply.response_text,
        }

    @router.post("/chat")
    def post_chat(req: ChatRequest) -> dict:
        reply = chat.submit_message(req.message)
        return {
            "interpreted_intent": reply.interpreted_intent.__dict__,
            "executed_action": reply.interpreted_intent.action,
            "structured_result": reply.deterministic_result,
            "formatted_response": reply.response_text,
            "metadata": {
                "used_llm_intent": reply.used_llm_intent,
                "used_llm_formatting": reply.used_llm_formatting,
                "fallback_reason": reply.fallback_reason,
            },
        }

    return router

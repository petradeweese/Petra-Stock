from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from kade.chat.service import ChatService
from kade.runtime import KadeRuntime

from .view_models import dashboard_vm


def build_page_router(
    runtime: KadeRuntime, chat: ChatService, templates: Jinja2Templates
) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page(request: Request):
        vm = dashboard_vm(runtime.dashboard_payload())
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "dashboard": vm,
                "chat_history": runtime.history_payload().get("items", []),
                "chat_enabled": chat.config.natural_language_chat_enabled,
            },
        )

    return router

from __future__ import annotations

import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kade.chat.service import ChatService
from kade.config import KadeConfig
from kade.runtime import KadeRuntime

from .api import build_api_router
from .routes import build_page_router


def create_ui_app(config: KadeConfig | None = None) -> FastAPI:
    config = config or KadeConfig.from_env()
    runtime = KadeRuntime()
    chat = ChatService.build(runtime, config)

    app = FastAPI(title="Kade Operator UI")
    templates = Jinja2Templates(directory="kade/ui/templates")
    app.mount("/static", StaticFiles(directory="kade/ui/static"), name="static")
    app.include_router(build_page_router(runtime, chat, templates))
    app.include_router(build_api_router(runtime, chat))
    app.state.runtime = runtime
    app.state.chat = chat
    app.state.config = config
    return app


def main() -> None:
    app = create_ui_app()
    host, port = "127.0.0.1", 8010
    url = f"http://{host}:{port}"
    print(f"Kade Operator UI running at {url}")
    if app.state.config.auto_open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

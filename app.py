import json
import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

try:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
except Exception:  # pragma: no cover - optional dependency
    ProxyHeadersMiddleware = None  # type: ignore[assignment, misc]

try:  # pragma: no cover - optional speed-up
    import uvloop

    uvloop.install()
except Exception:
    pass

from db import init_db
from routes import router
from scanner import compute_scan_for_ticker
from scheduler import setup_scheduler
from services import http_client
from utils import market_is_open, now_et


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = getattr(record, "run_id", None)
        if run_id:
            data["run_id"] = run_id
        return json.dumps(data)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    logger.info("Initializing database")
    run_migrations = os.getenv("RUN_MIGRATIONS", "true").lower() not in {
        "0",
        "false",
        "",
    }
    if run_migrations:
        logger.info("Running database migrations")
        init_db()
    else:
        logger.info("Skipping database migrations")

    app = FastAPI()
    if ProxyHeadersMiddleware is not None:
        logger.info("ProxyHeadersMiddleware enabled")
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")  # type: ignore[arg-type]
    else:
        logger.info("ProxyHeadersMiddleware unavailable; skipping")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.include_router(router)
    setup_scheduler(app, market_is_open, now_et, compute_scan_for_ticker)

    @app.on_event("shutdown")
    async def _shutdown():
        await http_client.aclose()

    return app


app = create_app()

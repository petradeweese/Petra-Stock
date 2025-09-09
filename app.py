import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

try:  # pragma: no cover - optional speed-up
    import uvloop

    uvloop.install()
except Exception:
    pass

from db import init_db
from routes import router
from scheduler import setup_scheduler
from utils import now_et, market_is_open
from scanner import compute_scan_for_ticker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    logger.info("Initializing database")
    try:
        logger.info("Running database migrations")
        init_db()
    except Exception:
        logger.exception("Failed to initialize database")
        raise

    app = FastAPI()
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.include_router(router)
    setup_scheduler(app, market_is_open, now_et, compute_scan_for_ticker)
    return app


app = create_app()

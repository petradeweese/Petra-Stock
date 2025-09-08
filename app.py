import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from db import init_db
from routes import router
from scheduler import setup_scheduler
from utils import now_et, market_is_open
from scanner import compute_scan_for_ticker


def create_app() -> FastAPI:
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    init_db()

    app = FastAPI()
    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.include_router(router)
    setup_scheduler(app, market_is_open, now_et, compute_scan_for_ticker)
    return app


app = create_app()

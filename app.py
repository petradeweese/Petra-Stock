import json
import logging
import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

try:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
except Exception:  # pragma: no cover - optional dependency
    ProxyHeadersMiddleware = None  # type: ignore[assignment]

try:  # pragma: no cover - optional speed-up
    import uvloop

    uvloop.install()
except Exception:
    pass

from db import init_db, get_db
from routes import router
from scanner import compute_scan_for_ticker
from scheduler import setup_scheduler
from services import http_client
from services.overnight import start_background_runner
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
    if os.getenv("RUN_MIGRATIONS", "true").lower() not in {"0", "false", ""}:
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

    def _overnight_scan(payload: dict, silent: bool) -> int:
        params = dict(payload.get("settings") or {})
        pattern = payload.get("pattern")
        if pattern:
            params["rule"] = pattern
        universe = [
            t.strip().upper()
            for t in str(payload.get("universe", "")).split(",")
            if t.strip()
        ]
        started = datetime.utcnow().isoformat()
        rows: list[dict] = []
        for t in universe:
            try:
                res = compute_scan_for_ticker(t, params)
            except Exception:
                logger.exception("overnight scan failed ticker=%s", t)
                continue
            if res:
                rows.append(res)
        finished = datetime.utcnow().isoformat()
        gen = get_db()
        db = next(gen)
        try:
            db.execute(
                """
                INSERT INTO runs(
                    started_at, scan_type, params_json, universe, finished_at,
                    hit_count, settings_json
                )
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    started,
                    params.get("scan_type", "overnight"),
                    json.dumps(params),
                    ",".join(universe),
                    finished,
                    len(rows),
                    json.dumps(params),
                ),
            )
            run_id = db.lastrowid
            for r in rows:
                db.execute(
                    """
                    INSERT INTO run_results(
                        run_id, ticker, direction, avg_roi_pct, hit_pct, support,
                        avg_tt, avg_dd_pct, stability, rule
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        r.get("ticker"),
                        r.get("direction", "UP"),
                        float(r.get("avg_roi_pct", 0.0)),
                        float(r.get("hit_pct", 0.0)),
                        int(r.get("support", 0)),
                        float(r.get("avg_tt", 0.0)),
                        float(r.get("avg_dd_pct", 0.0)),
                        float(r.get("stability", 0.0)),
                        r.get("rule", ""),
                    ),
                )
            db.connection.commit()
            return run_id
        finally:
            gen.close()

    start_background_runner(_overnight_scan)

    @app.on_event("shutdown")
    async def _shutdown():
        await http_client.aclose()

    return app


app = create_app()

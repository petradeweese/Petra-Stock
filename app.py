import json
import logging
import os
from datetime import datetime, timezone

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
from services.market_data import override_window_end
from services.overnight import (
    OvernightScanError,
    resolve_overnight_symbols,
    resolve_overnight_window,
    start_background_runner,
)
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

    def _overnight_scan(payload: dict, silent: bool) -> dict:
        params = dict(payload.get("settings") or {})
        pattern = payload.get("pattern")
        if pattern:
            params["rule"] = pattern
        resolution = resolve_overnight_symbols(payload)
        universe = resolution.symbols
        now_ts = now_et()
        window_start_et, window_end_et = resolve_overnight_window(now_ts)
        window_end_utc = window_end_et.astimezone(timezone.utc)
        metadata: dict = {
            "scheduled_at": now_ts.isoformat(),
            "window_start": window_start_et.isoformat(),
            "window_end": window_end_et.isoformat(),
            "timezone": "America/New_York",
            "symbols_total": resolution.symbols_total,
            "universe": resolution.universe,
            "ticker": resolution.ticker,
            "silent": silent,
        }
        processed = 0
        successes = 0
        failures: list[dict] = []
        rows: list[dict] = []

        with override_window_end(window_end_utc):
            for raw in universe:
                ticker = raw.strip().upper()
                if not ticker:
                    continue
                processed += 1
                try:
                    res = compute_scan_for_ticker(ticker, params)
                except Exception as exc:
                    failures.append({"ticker": ticker, "error": repr(exc)})
                    logger.exception("overnight scan failed ticker=%s", ticker)
                    continue
                if res:
                    rows.append(res)
                    successes += 1

        metadata["processed"] = processed
        metadata["successes"] = successes
        metadata["failures"] = len(failures)
        if failures:
            metadata["failure_examples"] = failures[:5]

        if processed == 0:
            err_msg = resolution.detail or "No symbols to scan"
            metadata["error"] = err_msg
            logger.error(json.dumps({"type": "overnight_scan", **metadata}))
            raise OvernightScanError(err_msg, meta=metadata)

        if successes == 0 and len(failures) == processed:
            metadata["error"] = "Scan failed for all symbols"
            logger.error(json.dumps({"type": "overnight_scan", **metadata}))
            raise OvernightScanError("Scan failed for all symbols", meta=metadata)

        if successes == 0:
            metadata["note"] = f"Scanned {processed} symbols — 0 results"
        elif failures:
            metadata["note"] = f"{successes} results • {len(failures)} tickers failed"
        else:
            metadata["note"] = f"Scanned {processed} symbols"

        metadata["rows_returned"] = len(rows)

        params_with_meta = dict(params)
        params_with_meta["_meta"] = metadata

        started = now_ts.astimezone(timezone.utc).isoformat()
        finished = datetime.now(timezone.utc).isoformat()
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
                    json.dumps(params_with_meta),
                    ",".join(universe),
                    finished,
                    len(rows),
                    json.dumps(params_with_meta),
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
        finally:
            gen.close()

        metadata["run_id"] = run_id
        log_payload = {k: v for k, v in metadata.items() if k != "failure_examples"}
        log_payload["type"] = "overnight_scan"
        if metadata.get("failure_examples"):
            log_payload["failure_examples"] = metadata["failure_examples"]
        logger.info(json.dumps(log_payload))

        return {"run_id": run_id, **metadata}

    start_background_runner(_overnight_scan)

    @app.on_event("shutdown")
    async def _shutdown():
        await http_client.aclose()

    return app


app = create_app()

# ruff: noqa: E501
import asyncio
import inspect
import atexit
import base64
import csv
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import statistics
import time
import traceback
import urllib.parse
from concurrent.futures import as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates

from config import settings
from db import (
    DB_PATH,
    _ensure_scanner_column,
    get_db,
    get_schema_status,
    get_settings,
    row_to_dict,
)
from indices import SP100, TOP150, TOP250
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from scanner import compute_scan_for_ticker, export_simulation_artifacts
from services import (
    data_provider,
    executor,
    favorites_alerts,
    favorites_sim,
    http_client,
    paper_trading,
    price_store,
    schwab_client,
    sms_consent,
    twilio_client,
)
from services.scalper import hf_engine as scalper_hf, lf_engine as scalper_lf
from services.favorites import canonical_direction, ensure_favorite_directions
from services.notify import send_email_smtp
from services.scanner_params import coerce_scan_params
from services.telemetry import log as log_telemetry
from services.config import DEBUG_SIMULATION, SMS_MAX_PER_MONTH
from services.favorites_alerts import deliver_preview_alert
from services.simulator import SimEvent, build_sim_hit, build_sim_stop
from services.scheduler import _align_to_bar, _dedupe_key, was_sent_key
from services.forward_runs import log_forward_entry, log_forward_exit
from services.market_data import (
    expected_bar_count,
    fetch_prices,
    get_prices,
    window_from_lookback,
)
from services.oauth_tokens import store_refresh_token
from services.forward_runs import (
    forward_rule_hash,
    get_forward_history_for_cursor,
    log_forward_entry,
    log_forward_exit,
)
from services.forward_summary import get_cached_forward_summary
from services.schwab_client import TOKEN_URL, update_refresh_token
from utils import TZ, now_et

from .archive import _format_rule_summary as _format_rule_summary
from .archive import router as archive_router
from .overnight import router as overnight_router
from .template_helpers import register_template_helpers

router = APIRouter()
templates = Jinja2Templates(directory="templates")
register_template_helpers(templates)
logger = logging.getLogger(__name__)
SCAN_BATCH_WRITES = os.getenv("SCAN_BATCH_WRITES", "1") not in {"0", "false", "no"}

SCHWAB_AUTH_URL = os.getenv(
    "SCHWAB_AUTH_URL", "https://api.schwab.com/oauth2/v1/authorize"
)
_STATE_TTL_SECONDS = 600
_SCHWAB_STATES: Dict[str, tuple[str, float]] = {}
_SCHWAB_STATE_LOCK = Lock()
_PAPER_RATE_LIMIT: Dict[str, float] = {}
_PAPER_RATE_LOCK = Lock()

_EMPTY_HEATMAP: Dict[str, Any] = {"index": [], "columns": [], "values": [], "meta": {}}
_LATEST_HEATMAP: Dict[str, Any] = dict(_EMPTY_HEATMAP)
_HEATMAP_CACHE: Dict[str, Any] = {"data": None, "expires": 0.0}

_SCAN_TASKS: dict[str, asyncio.Task] = {}


_FAVORITE_INSERT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "direction",
    "interval",
    "rule",
    "target_pct",
    "stop_pct",
    "window_value",
    "window_unit",
    "ref_avg_dd",
    "lookback_years",
    "min_support",
    "support_snapshot",
    "roi_snapshot",
    "hit_pct_snapshot",
    "dd_pct_snapshot",
    "rule_snapshot",
    "settings_json_snapshot",
    "snapshot_at",
)

_FAVORITE_OPTIONAL_COLUMNS = frozenset({"roi_snapshot"})
_FAVORITE_COLUMN_CACHE: dict[int, tuple[str, ...]] = {}


@dataclass
class ScanPlan:
    tickers: list[str]
    interval: str
    lookback: float
    window_start: datetime
    window_end: datetime
    need_fetch: list[str]
    no_gap: list[str]
    expected: int
    coverage_ms: float


def _build_scan_plan(tickers: list[str], params: dict) -> ScanPlan:
    total = len(tickers)
    interval = params.get("interval", "15m")
    lookback = float(params.get("lookback_years", 2.0))
    window_start, window_end = window_from_lookback(lookback)

    cov_start = _perf_counter()
    cov: dict[str, tuple[datetime, datetime, int]] = {}
    batch = max(1, settings.scan_coverage_batch_size)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        cov.update(price_store.bulk_coverage(chunk, interval, window_start, window_end))
    expected = expected_bar_count(window_start, window_end, interval)
    no_gap: list[str] = []
    need_fetch: list[str] = []
    for sym in tickers:
        cmin, cmax, cnt = cov.get(sym, (None, None, 0))
        if cnt >= expected and price_store.covers(window_start, window_end, cmin, cmax):
            no_gap.append(sym)
        else:
            need_fetch.append(sym)
    cov_elapsed = (_perf_counter() - cov_start) * 1000.0
    coverage_symbols_total.inc(total)
    coverage_symbols_no_gap.inc(len(no_gap))
    coverage_symbols_gap_fetched.inc(len(need_fetch))
    coverage_elapsed_seconds.observe(cov_elapsed / 1000.0)
    logger.info(
        "coverage_mode=batch symbols=%d rows=%d elapsed_ms=%.0f",
        total,
        len(cov),
        cov_elapsed,
    )
    return ScanPlan(
        tickers=tickers,
        interval=interval,
        lookback=lookback,
        window_start=window_start,
        window_end=window_end,
        need_fetch=need_fetch,
        no_gap=no_gap,
        expected=expected,
        coverage_ms=cov_elapsed,
    )

_SMS_CONSENT_TEXT = (
    "I agree to receive automated Petra Stock SMS alerts about pattern signals and "
    "performance updates. "
    f"No more than {SMS_MAX_PER_MONTH} texts/month. Msg & data rates may apply. "
    "Text STOP to opt out, HELP for help. "
    "By checking this box, I agree to the Terms and Privacy Policy."
)


@router.get("/schwab/login")
def schwab_login() -> Response:
    client_id = settings.schwab_client_id
    redirect_uri = settings.schwab_redirect_uri
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail="Schwab integration not configured")

    state = secrets.token_urlsafe(16)
    verifier = _generate_code_verifier()
    _remember_state(state, verifier)
    challenge = _code_challenge(verifier)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    authorize_url = f"{SCHWAB_AUTH_URL}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"
    logger.info("schwab_oauth_redirect")
    return RedirectResponse(authorize_url, status_code=307)


@router.get("/callback", response_class=HTMLResponse)
async def schwab_callback(
    state: str | None = Query(default=None),
    code: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db=Depends(get_db),
):
    if error:
        logger.warning("schwab_oauth_error error=%s", error)
        raise HTTPException(status_code=400, detail="Authorization failed")
    if not state or not code:
        raise HTTPException(status_code=400, detail="Missing state or code")

    verifier = _consume_state(state)
    if not verifier:
        raise HTTPException(status_code=400, detail="Invalid state")

    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.schwab_redirect_uri,
        "client_id": settings.schwab_client_id,
        "client_secret": settings.schwab_client_secret,
        "code_verifier": verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        resp = await http_client.request("POST", TOKEN_URL, data=payload, headers=headers)
    except Exception:
        logger.exception("schwab_token_exchange_exception")
        raise HTTPException(status_code=400, detail="Token exchange failed")

    if resp.status_code >= 400:
        logger.warning("schwab_token_exchange_error status=%s", resp.status_code)
        raise HTTPException(status_code=400, detail="Token exchange failed")

    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}

    refresh_token = data.get("refresh_token")
    if not refresh_token:
        logger.warning("schwab_token_exchange_missing_refresh")
        raise HTTPException(status_code=400, detail="Token exchange failed")

    store_refresh_token(
        "schwab",
        str(refresh_token),
        account_id=(settings.schwab_account_id or None),
        db_cursor=db,
    )
    update_refresh_token(str(refresh_token))
    logger.info("schwab_oauth_success")

    html = """<!DOCTYPE html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"utf-8\" />\n    <title>Schwab Linked</title>\n  </head>\n  <body>\n    <p>Schwab linked. You may close this window.</p>\n  </body>\n</html>\n"""
    return HTMLResponse(content=html, status_code=200)


def _cleanup_schwab_states(now: float | None = None) -> None:
    current = now or time.time()
    expired = [
        key
        for key, (_, ts) in list(_SCHWAB_STATES.items())
        if current - ts > _STATE_TTL_SECONDS
    ]
    for key in expired:
        _SCHWAB_STATES.pop(key, None)


def _generate_code_verifier() -> str:
    while True:
        verifier = secrets.token_urlsafe(64).replace("=", "")
        if len(verifier) < 43:
            verifier = f"{verifier}{secrets.token_urlsafe(43)}"
        if len(verifier) >= 43:
            return verifier[:128]


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _remember_state(state: str, verifier: str) -> None:
    now = time.time()
    with _SCHWAB_STATE_LOCK:
        _cleanup_schwab_states(now)
        _SCHWAB_STATES[state] = (verifier, now)


def _consume_state(state: str) -> str:
    now = time.time()
    with _SCHWAB_STATE_LOCK:
        entry = _SCHWAB_STATES.pop(state, None)
    if not entry:
        return ""
    verifier, created_at = entry
    if now - created_at > _STATE_TTL_SECONDS:
        return ""
    return verifier


def _request_user_id(request: Request | None) -> str:
    if request is not None:
        state = getattr(request, "state", None)
        for attr in ("user_id", "owner_id", "account_id", "user"):
            value = getattr(state, attr, None) if state is not None else None
            if value not in (None, ""):
                return str(value)
        for header in ("x-user-id", "x-account-id", "x-owner-id"):
            header_val = request.headers.get(header)
            if header_val:
                return header_val
    return sms_consent.DEFAULT_USER_ID

_SECTOR_OVERRIDES: Dict[str, str] = {}
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMD",
            "INTC",
            "ADBE",
            "CRM",
            "ORCL",
            "CSCO",
            "AVGO",
            "QCOM",
            "TXN",
            "SMCI",
            "SNOW",
            "NOW",
            "TEAM",
            "PLTR",
            "PANW",
            "FTNT",
            "CRWD",
            "ZS",
            "OKTA",
            "DDOG",
            "NET",
            "INTU",
            "ADP",
            "WDAY",
            "HPQ",
            "DELL",
            "IBM",
            "AMAT",
            "FSLR",
            "ENPH",
            "SEDG",
            "V",
            "MA",
        ],
        "Information Technology",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "META",
            "GOOGL",
            "GOOG",
            "NFLX",
            "DIS",
            "CMCSA",
            "CHTR",
            "TMUS",
            "T",
            "VZ",
            "ROKU",
            "RBLX",
        ],
        "Communication Services",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "AMZN",
            "TSLA",
            "SBUX",
            "MCD",
            "CMG",
            "DPZ",
            "YUM",
            "NKE",
            "LOW",
            "HD",
            "TGT",
            "ABNB",
            "ORLY",
            "AZO",
            "CCL",
            "RCL",
            "BABA",
            "JD",
            "PDD",
            "NIO",
            "XPEV",
            "LI",
            "RIVN",
            "LCID",
            "GM",
            "F",
        ],
        "Consumer Discretionary",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "COST",
            "WMT",
            "KO",
            "PEP",
            "PG",
            "PM",
            "MO",
            "WBA",
            "MDLZ",
            "KHC",
        ],
        "Consumer Staples",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "LLY",
            "PFE",
            "MRK",
            "ABBV",
            "BMY",
            "UNH",
            "TMO",
            "ISRG",
            "MDT",
            "CI",
            "HUM",
            "VRTX",
            "REGN",
            "GILD",
            "AMGN",
            "CVS",
            "DHR",
        ],
        "Health Care",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "JPM",
            "BAC",
            "GS",
            "MS",
            "C",
            "WFC",
            "SCHW",
            "BLK",
            "SPGI",
            "MSCI",
            "ICE",
            "CME",
            "COIN",
            "SOFI",
            "PYPL",
            "AXP",
            "COF",
            "USB",
            "BRK-B",
            "BK",
            "MET",
        ],
        "Financials",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "XOM",
            "CVX",
            "SLB",
            "COP",
            "OXY",
        ],
        "Energy",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "CAT",
            "DE",
            "BA",
            "GE",
            "HON",
            "MMM",
            "RTX",
            "LMT",
            "NOC",
            "UNP",
            "UPS",
            "FDX",
            "DAL",
            "UAL",
            "LUV",
            "UBER",
            "EMR",
        ],
        "Industrials",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "LIN",
            "APD",
            "NUE",
            "FCX",
            "DOW",
            "CF",
            "MOS",
        ],
        "Materials",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        [
            "NEE",
            "SO",
            "DUK",
            "EXC",
            "AEP",
            "D",
        ],
        "Utilities",
    )
)
_SECTOR_OVERRIDES.update(
    dict.fromkeys(
        ["IYR", "O", "AMT", "PLD", "EQIX", "WELL", "ARE", "EQR", "DLR", "BXP"],
        "Real Estate",
    )
)
_SECTOR_OVERRIDES.update(dict.fromkeys(["SPY", "QQQ"], "ETF"))

_CSV_EXPORT_COLUMNS: List[str] = [
    "ticker",
    "direction",
    "avg_roi_pct",
    "hit_pct",
    "hit_lb95",
    "support",
    "avg_tt",
    "avg_dd_pct",
    "stability",
    "sharpe",
    "rule",
    "stop_pct",
    "timeout_pct",
    "confidence",
    "confidence_label",
    "recent3",
]


def _lookup_sector(ticker: Any) -> str:
    if not ticker:
        return "Unknown"
    try:
        symbol = str(ticker).upper()
    except Exception:
        return "Unknown"
    return _SECTOR_OVERRIDES.get(symbol, "Unknown")


def _csv_value(row: dict[str, Any], column: str) -> Any:
    value = row.get(column)
    if column == "recent3":
        try:
            return json.dumps(value or [])
        except Exception:
            return "[]"
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return value


def _rows_to_csv_table(rows: List[dict]) -> tuple[List[str], List[List[Any]]]:
    data: List[List[Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        data.append([_csv_value(row, col) for col in _CSV_EXPORT_COLUMNS])
    return list(_CSV_EXPORT_COLUMNS), data

_perf_counter = time.perf_counter

FORWARD_SLIPPAGE = 0.0008


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _favorite_roi_value(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is not None:
        return numeric

    if isinstance(value, (bytes, str)):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return _coerce_float(parsed)

    return None


def _interval_minutes(interval: Any) -> int:
    try:
        label = str(interval or "").strip().lower()
    except Exception:
        return 0
    if not label:
        return 0
    try:
        if label.endswith("m"):
            return int(float(label[:-1]))
        if label.endswith("h"):
            return int(float(label[:-1]) * 60)
        if label.endswith("d"):
            return int(float(label[:-1]) * 24 * 60)
        return int(float(label))
    except (TypeError, ValueError):
        return 0


def _parse_forward_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        stamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    try:
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(TZ)
    except (TypeError, ValueError):
        try:
            stamp = stamp.tz_localize("UTC")
        except Exception:
            return None
    if stamp.tzinfo is None:
        return None
    try:
        return stamp.tz_convert("UTC")
    except Exception:
        return None


def _format_event_ts(stamp: pd.Timestamp | None) -> str | None:
    if stamp is None:
        return None
    iso = stamp.isoformat()
    if iso.endswith("+00:00"):
        return f"{iso[:-6]}Z"
    return iso


def _estimate_exit_price(
    forward_row: dict[str, Any],
    roi_pct: float | None,
) -> float | None:
    entry_price = _coerce_float(forward_row.get("entry_price"))
    if entry_price is None or roi_pct is None:
        return None
    direction = canonical_direction(forward_row.get("direction")) or "UP"
    side = 1 if direction == "UP" else -1
    slip = FORWARD_SLIPPAGE
    entry_fill = entry_price * (1 + slip * side)
    try:
        exit_fill = entry_fill * (1 + (roi_pct / 100.0) / side)
    except ZeroDivisionError:
        return None
    exit_factor = 1 - slip * side
    if exit_factor == 0:
        return None
    return float(exit_fill / exit_factor)


def _forward_events(forward_row: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    entry_price = _coerce_float(forward_row.get("entry_price"))
    entry_stamp = _parse_forward_timestamp(forward_row.get("created_at"))
    interval_mins = _interval_minutes(forward_row.get("interval"))
    entry_time = entry_stamp
    if entry_stamp is not None and interval_mins:
        try:
            entry_time = entry_stamp + pd.Timedelta(minutes=interval_mins)
        except Exception:
            entry_time = entry_stamp
    entry_iso = _format_event_ts(entry_time or entry_stamp)
    if entry_iso and entry_price is not None:
        events.append({"t": "entry", "ts": entry_iso, "px": float(entry_price)})

    exit_reason_raw = forward_row.get("exit_reason")
    exit_reason = str(exit_reason_raw or "").strip().lower()
    if exit_reason not in {"target", "stop", "timeout"}:
        return events

    base_entry_time = entry_time or entry_stamp
    exit_stamp: pd.Timestamp | None = None
    elapsed_minutes: float | None = None
    if exit_reason == "target":
        elapsed_minutes = _coerce_float(forward_row.get("time_to_hit"))
    elif exit_reason == "stop":
        elapsed_minutes = _coerce_float(forward_row.get("time_to_stop"))
    if elapsed_minutes is not None and base_entry_time is not None:
        try:
            exit_stamp = base_entry_time + pd.Timedelta(minutes=float(elapsed_minutes))
        except Exception:
            exit_stamp = None
    if exit_stamp is None and base_entry_time is not None:
        bars = _coerce_int(forward_row.get("bars_to_exit"))
        if bars and interval_mins:
            try:
                exit_stamp = base_entry_time + pd.Timedelta(minutes=bars * interval_mins)
            except Exception:
                exit_stamp = None
    if exit_stamp is None:
        exit_stamp = _parse_forward_timestamp(
            forward_row.get("updated_at") or forward_row.get("last_run_at")
        )

    event: dict[str, Any] = {"t": "hit" if exit_reason == "target" else exit_reason}
    exit_iso = _format_event_ts(exit_stamp)
    if exit_iso:
        event["ts"] = exit_iso

    roi_pct = _coerce_float(forward_row.get("roi_forward"))
    exit_price = _estimate_exit_price(forward_row, roi_pct)
    if exit_price is not None:
        event["px"] = exit_price

    if roi_pct is not None:
        event["roi"] = roi_pct / 100.0

    bars_to_exit = _coerce_int(forward_row.get("bars_to_exit"))
    if bars_to_exit is not None:
        event["tt_bars"] = bars_to_exit

    events.append(event)
    return events


def _heatmap_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return f


def _heatmap_int(value: Any) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(f):
        return 0
    try:
        iv = int(round(f))
    except Exception:
        return 0
    return iv if iv >= 0 else 0


def _sort_by_lb95_roi_support(
    out: Any,
) -> Any:
    """Sort scan outputs by hit_lb95, avg_roi, support descending.

    Accepts either a pandas ``DataFrame`` or a list of dictionaries, matching
    the shapes produced by the web scanner endpoints and CSV exporters.
    Missing ``hit_lb95`` values are coerced to ``0.0`` before sorting so that
    the ordering is deterministic.
    """

    if isinstance(out, pd.DataFrame):
        df = out.copy()
        if "hit_lb95" not in df.columns:
            df["hit_lb95"] = 0.0
        else:
            df["hit_lb95"] = pd.to_numeric(df["hit_lb95"], errors="coerce").fillna(0.0)

        if "support" not in df.columns:
            df["support"] = 0
        support_sort = pd.to_numeric(df.get("support"), errors="coerce").fillna(0.0)

        if "avg_roi" in df.columns:
            avg_roi_source = df["avg_roi"]
        elif "avg_roi_pct" in df.columns:
            avg_roi_source = df["avg_roi_pct"]
        else:
            avg_roi_source = 0.0
        avg_roi_sort = pd.to_numeric(avg_roi_source, errors="coerce").fillna(0.0)

        df = df.assign(
            _avg_roi_sort=avg_roi_sort,
            _support_sort=support_sort,
        ).sort_values(
            ["hit_lb95", "_avg_roi_sort", "_support_sort"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        return df.drop(columns=["_avg_roi_sort", "_support_sort"])

    if isinstance(out, list):

        def _metric(row: dict[str, Any] | Any, key: str) -> float:
            if not isinstance(row, dict):
                return 0.0
            value: Any
            if key == "avg_roi":
                value = row.get("avg_roi")
                if value in (None, ""):
                    value = row.get("avg_roi_pct")
            else:
                value = row.get(key)
            try:
                return float(value) if value not in (None, "") else 0.0
            except (TypeError, ValueError):
                return 0.0

        for row in out:
            if isinstance(row, dict) and ("hit_lb95" not in row or row.get("hit_lb95") is None):
                row["hit_lb95"] = 0.0

        out.sort(
            key=lambda row: (
                _metric(row, "hit_lb95"),
                _metric(row, "avg_roi"),
                _metric(row, "support"),
            ),
            reverse=True,
        )
        return out

    return out


def _wilson_lb95(hits: int, n: int) -> float:
    if n <= 0:
        return 0.0
    z = 1.96
    p = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))
    lb = (center - margin) / denom
    if lb < 0.0:
        return 0.0
    if lb > 1.0:
        return 1.0
    return lb


def _build_heatmap(rows: list[dict[str, Any]] | None) -> Dict[str, Any]:
    if not rows:
        return dict(_EMPTY_HEATMAP)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        ticker = item.get("ticker")
        sector = item.get("sector")
        if isinstance(sector, str):
            sector_value = sector.strip() or "Unknown"
        else:
            sector_value = None
        if not sector_value:
            sector_value = _lookup_sector(ticker)
        item["sector"] = sector_value or "Unknown"
        enriched.append(item)

    if not enriched:
        return dict(_EMPTY_HEATMAP)

    try:
        df = pd.DataFrame(enriched)
    except Exception:
        return dict(_EMPTY_HEATMAP)

    if df.empty or "ticker" not in df.columns:
        return dict(_EMPTY_HEATMAP)

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)

    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    else:
        df["sector"] = df["sector"].apply(
            lambda v: (str(v).strip() or "Unknown") if v not in (None, "") else "Unknown"
        )

    df["support"] = pd.to_numeric(df.get("support"), errors="coerce").fillna(0).astype(int)
    df.loc[df["support"] < 0, "support"] = 0
    df = df[df["support"] >= 10].copy()
    if df.empty:
        return dict(_EMPTY_HEATMAP)

    if "avg_roi" in df.columns:
        df["avg_roi"] = pd.to_numeric(df["avg_roi"], errors="coerce").fillna(0.0)
    elif "avg_roi_pct" in df.columns:
        df["avg_roi"] = pd.to_numeric(df["avg_roi_pct"], errors="coerce").fillna(0.0) / 100.0
    else:
        df["avg_roi"] = 0.0

    if "hit_lb95" in df.columns:
        df["hit_lb95"] = pd.to_numeric(df["hit_lb95"], errors="coerce").fillna(0.0)
    else:
        hit_series = pd.to_numeric(df.get("hit_pct"), errors="coerce").fillna(0.0) / 100.0
        lb_vals: list[float] = []
        for frac, supp in zip(hit_series.tolist(), df["support"].tolist()):
            if supp <= 0:
                lb_vals.append(0.0)
                continue
            frac = max(0.0, min(1.0, float(frac)))
            hits = int(round(frac * supp))
            if hits < 0:
                hits = 0
            elif hits > supp:
                hits = supp
            lb_vals.append(_wilson_lb95(hits, supp))
        df["hit_lb95"] = lb_vals

    df["hit_lb95"] = df["hit_lb95"].apply(lambda v: max(0.0, min(1.0, _heatmap_float(v))))

    heat = df.pivot_table(index="sector", columns="ticker", values="hit_lb95", aggfunc="max")
    if heat.empty:
        return dict(_EMPTY_HEATMAP)

    heat = heat.sort_index().sort_index(axis=1)
    index_labels = [str(x) for x in heat.index]
    column_labels = [str(x) for x in heat.columns]
    values = [
        [None if pd.isna(val) else float(val) for val in row]
        for row in heat.to_numpy()
    ]

    meta = df.groupby(["sector", "ticker"]).agg(
        support=("support", "max"),
        avg_roi=("avg_roi", "mean"),
        hit_lb95=("hit_lb95", "max"),
    ).reset_index()

    meta_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for _, row in meta.iterrows():
        sector = str(row.get("sector", "Unknown"))
        ticker = str(row.get("ticker", ""))
        meta_lookup[(sector, ticker)] = {
            "support": _heatmap_int(row.get("support")),
            "avg_roi": _heatmap_float(row.get("avg_roi")),
            "hit_lb95": _heatmap_float(row.get("hit_lb95")),
        }

    meta_json = {
        f"{sector}|{ticker}": meta_lookup.get((sector, ticker), {})
        for sector in index_labels
        for ticker in column_labels
    }

    return {"index": index_labels, "columns": column_labels, "values": values, "meta": meta_json}


def _update_heatmap(rows: list[dict[str, Any]] | None) -> None:
    global _LATEST_HEATMAP
    try:
        _LATEST_HEATMAP = _build_heatmap(rows)
        _HEATMAP_CACHE["data"] = None
        _HEATMAP_CACHE["expires"] = 0.0
    except Exception:
        logger.exception("Failed to build heatmap data")
        _LATEST_HEATMAP = dict(_EMPTY_HEATMAP)
        _HEATMAP_CACHE["data"] = None
        _HEATMAP_CACHE["expires"] = 0.0


def _parse_support_snapshot(raw: Any) -> tuple[int | None, dict[str, Any] | None]:
    data: dict[str, Any] | None = None
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
    elif isinstance(raw, dict):
        data = raw

    if isinstance(data, dict):
        support_val = data.get("count")
        if support_val is None:
            support_val = data.get("support")
        return _coerce_int(support_val), data

    return _coerce_int(raw), None


def _parse_settings_snapshot(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    settings_dict: dict[str, Any] = {}
    if isinstance(raw, dict):
        settings_dict = raw
    elif isinstance(raw, (str, bytes)):
        text = raw.decode() if isinstance(raw, bytes) else raw
        text = text.strip()
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    settings_dict = data
            except Exception:
                settings_dict = {}

    if not settings_dict:
        return {}, {}

    coerced = coerce_scan_params(settings_dict)
    return settings_dict, coerced


def _snapshot_has_value(snapshot: dict[str, Any], key: str) -> bool:
    if key not in snapshot:
        return False
    value = snapshot.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _format_lookback(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return f"{int(rounded)}y"
    return f"{value:.1f}y"


def _normalize_favorite(row: dict[str, Any]) -> dict[str, Any]:
    support_count, support_snapshot = _parse_support_snapshot(
        row.get("support_snapshot")
    )
    settings_raw, settings_params = _parse_settings_snapshot(
        row.get("settings_json_snapshot")
    )
    saved_direction = canonical_direction(row.get("direction"))
    lookback = _coerce_float(row.get("lookback_years"))
    lookback_from_support = (
        _coerce_float(support_snapshot.get("lookback_years"))
        if isinstance(support_snapshot, dict)
        else None
    )
    lookback_from_settings = None
    if settings_params and _snapshot_has_value(settings_raw, "lookback_years"):
        lookback_from_settings = _coerce_float(settings_params.get("lookback_years"))
    if lookback_from_settings is not None:
        lookback = lookback_from_settings
    elif lookback is None and lookback_from_support is not None:
        lookback = lookback_from_support

    min_support = _coerce_int(row.get("min_support"))
    min_support_from_support = (
        _coerce_int(support_snapshot.get("min_support"))
        if isinstance(support_snapshot, dict)
        else None
    )
    min_support_from_settings = None
    if settings_params and _snapshot_has_value(settings_raw, "min_support"):
        min_support_from_settings = _coerce_int(settings_params.get("min_support"))
    if min_support_from_settings is not None:
        min_support = min_support_from_settings
    elif min_support is None and min_support_from_support is not None:
        min_support = min_support_from_support

    if support_count is None:
        support_count = min_support

    row["lookback_years"] = lookback
    row["support_count"] = support_count
    row["lookback_display"] = _format_lookback(lookback)
    row["support_display"] = (
        str(support_count) if support_count is not None else "—"
    )
    if min_support is not None:
        row["min_support"] = min_support

    snapshot_direction = None
    if settings_params and _snapshot_has_value(settings_raw, "direction"):
        snapshot_direction = canonical_direction(settings_params.get("direction"))

    if saved_direction is None:
        saved_direction = snapshot_direction or "UP"

    row["direction"] = saved_direction

    if settings_params:
        if _snapshot_has_value(settings_raw, "target_pct"):
            target = _coerce_float(settings_params.get("target_pct"))
            if target is not None:
                row["target_pct"] = target
        if _snapshot_has_value(settings_raw, "stop_pct"):
            stop = _coerce_float(settings_params.get("stop_pct"))
            if stop is not None:
                row["stop_pct"] = stop
        if _snapshot_has_value(settings_raw, "window_value"):
            window_val = _coerce_float(settings_params.get("window_value"))
            if window_val is not None:
                row["window_value"] = window_val
        if _snapshot_has_value(settings_raw, "window_unit"):
            unit = settings_params.get("window_unit")
            if isinstance(unit, str) and unit:
                row["window_unit"] = unit
        if _snapshot_has_value(settings_raw, "interval") and settings_params.get("interval"):
            row["interval"] = settings_params.get("interval")
    return row


_SMTP_FIELD_LABELS = {
    "smtp_host": "SMTP host",
    "smtp_port": "SMTP port",
    "smtp_user": "SMTP user",
    "smtp_pass": "SMTP password",
    "mail_from": "Mail From",
    "recipients": "Recipients",
}


def _smtp_config_status(settings_row: dict) -> tuple[dict[str, Any], list[str], bool]:
    """Normalize SMTP settings and report missing fields."""

    host = (settings_row.get("smtp_host") or "").strip()
    try:
        port = int(settings_row.get("smtp_port") or 0)
    except (TypeError, ValueError):
        port = 0
    user = (settings_row.get("smtp_user") or "").strip()
    password = (settings_row.get("smtp_pass") or "").strip()
    mail_from = (settings_row.get("mail_from") or "").strip()
    recipients = [
        r.strip()
        for r in (settings_row.get("recipients") or "").split(",")
        if r.strip()
    ]

    missing: list[str] = []
    if not host:
        missing.append("smtp_host")
    if port <= 0:
        missing.append("smtp_port")
    if not user:
        missing.append("smtp_user")
    if not password:
        missing.append("smtp_pass")
    if not mail_from:
        missing.append("mail_from")
    if not recipients:
        missing.append("recipients")

    has_any = any([host, user, password, mail_from]) or bool(recipients)

    cfg = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "mail_from": mail_from,
        "recipients": recipients,
    }
    return cfg, missing, has_any


def _format_missing_fields(missing: list[str]) -> str:
    labels = [_SMTP_FIELD_LABELS.get(field, field) for field in missing]
    return ", ".join(labels)



def _extract_test_alert_message(result: Any) -> tuple[bool, str, str]:
    ok = False
    subject = ""
    body = ""
    payload: Optional[Any] = None
    if isinstance(result, tuple):
        ok = bool(result[0])
        if len(result) > 1:
            payload = result[1]
    else:
        ok = bool(result)
        if ok:
            payload = result
    if isinstance(payload, dict):
        subject = str(payload.get("subject") or "")
        body = str(payload.get("body") or "")
    elif isinstance(payload, str):
        body = payload
    elif payload is not None:
        body = str(payload)
    return ok, subject, body

def _get_next_earnings(ticker: str):  # pragma: no cover - external service
    return None


def _get_adv(ticker: str):  # pragma: no cover - external service
    return None


def check_guardrails(
    ticker: str,
    *,
    earnings_window: int = 7,
    adv_threshold: float = 1_000_000.0,
    get_earnings: Callable[[str], pd.Timestamp | None] | None = None,
    get_adv: Callable[[str], float | None] | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate guardrails for a forward test.

    Returns ``(True, [])`` when the test may proceed. When guardrails trigger,
    returns ``(False, flags)`` where ``flags`` describes the failing rules.
    ``earnings`` is flagged when within ``earnings_window`` trading days of a
    known earnings date.  ``low_liquidity`` is flagged when average daily volume
    falls below ``adv_threshold``.
    """

    get_earnings = get_earnings or _get_next_earnings
    get_adv = get_adv or _get_adv

    flags: list[str] = []
    today = now_et().date()

    try:
        edate = get_earnings(ticker)
        if edate is not None:
            edate = pd.Timestamp(edate).date()
            diff = abs(len(pd.bdate_range(today, edate)) - 1)
            if diff <= earnings_window:
                flags.append("earnings")
    except Exception:
        logger.exception("earnings guardrail failed ticker=%s", ticker)

    try:
        adv = get_adv(ticker)
        if adv is not None and adv < adv_threshold:
            flags.append("low_liquidity")
    except Exception:
        logger.exception("liquidity guardrail failed ticker=%s", ticker)

    return (not flags, flags)


router.include_router(archive_router)
router.include_router(overnight_router)


@router.get("/history")
def history_redirect():
    """Redirect legacy history link to archive."""
    return RedirectResponse(url="/archive", status_code=308)


scan_duration = Histogram("scan_duration_seconds", "Duration of /scanner/run requests")
scan_tickers = Counter("scan_tickers_total", "Tickers processed by /scanner/run")
coverage_symbols_total = Counter(
    "coverage_symbols_total", "Symbols processed during coverage checks"
)
coverage_symbols_no_gap = Counter(
    "coverage_symbols_no_gap", "Symbols with full local coverage"
)
coverage_symbols_gap_fetched = Counter(
    "coverage_symbols_gap_fetched", "Symbols fetched due to coverage gaps"
)
coverage_elapsed_seconds = Histogram(
    "coverage_elapsed_seconds", "Time spent performing bulk coverage checks"
)


def healthz() -> dict:
    """Simple health check used by tests and the /health endpoint."""
    return {"status": "ok"}


@router.get("/health")
def health() -> dict:
    """Return app health along with schema status information."""
    return {**healthz(), **get_schema_status()}


def metrics() -> Response:
    """Expose Prometheus metrics used by tests and the /metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if settings.metrics_enabled:

    @router.get("/metrics")
    def metrics_endpoint() -> Response:
        return metrics()


_TASK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scan_tasks (
    id TEXT PRIMARY KEY,
    total INTEGER,
    done INTEGER,
    percent REAL,
    state TEXT,
    message TEXT,
    ctx TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);
"""


def _ensure_task_columns(conn: sqlite3.Connection) -> None:
    """Ensure ``scan_tasks`` has ``started_at``/``updated_at`` columns.

    Older databases created before these columns existed should still work.  We
    perform a lightweight, idempotent check on each connection and add the
    missing columns if needed.  Existing rows are backfilled so callers relying
    on the timestamps do not crash.
    """

    cur = conn.execute("PRAGMA table_info(scan_tasks)")
    cols = {r[1] for r in cur.fetchall()}
    altered = False
    if "started_at" not in cols:
        _execute_write(conn, "ALTER TABLE scan_tasks ADD COLUMN started_at TEXT")
        altered = True
    if "updated_at" not in cols:
        _execute_write(conn, "ALTER TABLE scan_tasks ADD COLUMN updated_at TEXT")
        altered = True
    if altered:
        _execute_write(
            "UPDATE scan_tasks SET started_at=COALESCE(started_at, CURRENT_TIMESTAMP), "
            "updated_at=COALESCE(updated_at, started_at)"
        )
        conn.commit()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _execute_write(conn, _TASK_TABLE_SQL)
    conn.commit()
    try:
        _ensure_task_columns(conn)
    except Exception:
        # best effort; existing scans may proceed even if this fails
        pass
    return conn


_TASK_MEM: dict[str, dict[str, Any]] = {}
_TASK_WRITE_TS: dict[str, float] = {}


def _execute_write(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()):
    assert "bars" not in sql.lower(), "writes to bars table are not allowed"
    return conn.execute(sql, params)


def _task_create(task_id: str, total: int) -> None:
    _task_gc()
    conn = _get_conn()
    try:
        now_iso = now_et().isoformat()
        _execute_write(
            conn,
            (
                "INSERT INTO scan_tasks (id, total, done, percent, state, started_at, updated_at) "
                "VALUES (?, ?, 0, 0.0, 'queued', ?, ?)"
            ),
            (task_id, total, now_iso, now_iso),
        )
        conn.commit()
    finally:
        conn.close()
    _TASK_MEM[task_id] = {
        "total": total,
        "done": 0,
        "percent": 0.0,
        "state": "queued",
        "message": "",
        "ctx": None,
        "started_at": now_iso,
        "updated_at": now_iso,
        "first_status_seen": False,
    }
    _TASK_WRITE_TS[task_id] = time.monotonic()


def _task_update_db(task_id: str, fields: dict[str, Any]) -> None:
    if not fields:
        return
    conn = _get_conn()
    try:
        cols = ["updated_at=?"]
        vals: list[Any] = [now_et().isoformat()]
        for k, v in fields.items():
            if k == "first_status_seen":
                continue
            cols.append(f"{k}=?")
            if k == "ctx" and v is not None:
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        vals.append(task_id)
        _execute_write(
            conn, f"UPDATE scan_tasks SET {', '.join(cols)} WHERE id=?", tuple(vals)
        )
        conn.commit()
    finally:
        conn.close()


def _task_update(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    now_iso = now_et().isoformat()
    task = _TASK_MEM.get(task_id)
    if task is not None:
        task.update(fields)
        task["updated_at"] = now_iso
    last = _TASK_WRITE_TS.get(task_id, 0.0)
    need_persist = False
    items = settings.scan_progress_flush_items
    interval = settings.scan_status_flush_ms / 1000.0
    if task is not None:
        done = task.get("done")
        if isinstance(done, int) and items > 0 and done % items == 0:
            need_persist = True
    if time.monotonic() - last >= interval:
        need_persist = True
    if fields.get("state") in {"succeeded", "failed"}:
        need_persist = True
    if need_persist:
        _task_update_db(task_id, task if task is not None else fields)
        _TASK_WRITE_TS[task_id] = time.monotonic()


def _task_get(task_id: str) -> Optional[Dict[str, Any]]:
    task = _TASK_MEM.get(task_id)
    if task is not None:
        return task
    conn = _get_conn()
    try:
        cur = conn.execute(
            (
                "SELECT total, done, percent, state, message, ctx, started_at, updated_at "
                "FROM scan_tasks WHERE id=?"
            ),
            (task_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    total, done, percent, state, message, ctx_json, started_at, updated_at = row
    task = {
        "total": total,
        "done": done,
        "percent": percent,
        "state": state,
        "message": message,
        "ctx": json.loads(ctx_json) if ctx_json else None,
        "started_at": started_at,
        "updated_at": updated_at,
        "first_status_seen": False,
    }
    _TASK_MEM[task_id] = task
    _TASK_WRITE_TS[task_id] = time.monotonic()
    return task


def _task_delete(task_id: str) -> None:
    conn = _get_conn()
    try:
        _execute_write(conn, "DELETE FROM scan_tasks WHERE id=?", (task_id,))
        conn.commit()
    finally:
        conn.close()
    _TASK_MEM.pop(task_id, None)
    _TASK_WRITE_TS.pop(task_id, None)


def _task_gc(ttl_hours: int = 48) -> None:
    cutoff = (now_et() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn()
    try:
        _execute_write(conn, "DELETE FROM scan_tasks WHERE started_at < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()
    to_drop = [tid for tid, t in _TASK_MEM.items() if t.get("started_at", "") < cutoff]
    for tid in to_drop:
        _TASK_MEM.pop(tid, None)
        _TASK_WRITE_TS.pop(tid, None)


def _task_flush_all() -> None:
    """Persist all in-memory task states to the database."""
    for tid, task in list(_TASK_MEM.items()):
        try:
            _task_update_db(tid, task)
        except Exception:
            pass


atexit.register(_task_flush_all)


def _scan_single(t: str, params: dict) -> tuple[dict | None, float]:
    t0 = _perf_counter()
    res = compute_scan_for_ticker(t, params)
    return res, _perf_counter() - t0


def _scan_chunk(ts: list[str], params: dict) -> list[tuple[str, dict | None, float]]:
    pc = _perf_counter
    out: list[tuple[str, dict | None, float]] = []
    for t in ts:
        t0 = pc()
        try:
            res = compute_scan_for_ticker(t, params)
        except Exception as e:
            logger.error("scan failed for %s: %s", t, e)
            res = {}
        out.append((t, res, pc() - t0))
    return out


def _perform_scan(
    tickers: list[str],
    params: dict,
    sort_key: str,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    progress_every: int = 5,
    *,
    plan: Optional[ScanPlan] = None,
    fetch_missing: bool = True,
    fetch_elapsed_ms: Optional[float] = None,
) -> tuple[list[dict], int, dict]:
    start = _perf_counter()
    active_plan = plan or _build_scan_plan(tickers, params)
    total = len(tickers)
    interval = active_plan.interval
    lookback = active_plan.lookback
    need_fetch = list(active_plan.need_fetch)
    no_gap = list(active_plan.no_gap)
    cov_elapsed = active_plan.coverage_ms

    fetch_elapsed = float(fetch_elapsed_ms or 0.0)
    if fetch_missing and need_fetch:

        async def _fetch_all(symbols: list[str]) -> None:
            fetch_concurrency = int(settings.scan_fetch_concurrency or 0)
            if fetch_concurrency <= 0:
                logger.warning(
                    "scan fetch concurrency clamped", extra={"requested": fetch_concurrency}
                )
            sem = asyncio.Semaphore(max(1, fetch_concurrency))

            async def _worker(sym: str) -> None:
                async with sem:
                    logger.debug("fetch_gap symbol=%s", sym)
                    await asyncio.to_thread(fetch_prices, [sym], interval, lookback)

            await asyncio.gather(
                *(_worker(s) for s in symbols),
                return_exceptions=False,
            )

        fetch_start = _perf_counter()
        logger.info("fetch_start symbols=%d", len(need_fetch))
        asyncio.run(_fetch_all(need_fetch))
        fetch_elapsed = (_perf_counter() - fetch_start) * 1000.0
        logger.info(
            "fetch_done symbols=%d elapsed_ms=%.0f",
            len(need_fetch),
            fetch_elapsed,
        )

    if progress_cb:
        progress_cb(0, total, "preloading")

    rows: list[dict] = []
    skipped_missing_data = 0
    ex = executor.EXECUTOR
    chunk_size = max(1, settings.scan_symbols_per_task)
    futures = [
        ex.submit(_scan_chunk, tickers[i : i + chunk_size], params)
        for i in range(0, len(tickers), chunk_size)
    ]
    logger.info(
        "scan dispatch mode=%s workers=%d symbols_per_task=%d tasks=%d",
        executor.MODE,
        executor.WORKERS,
        chunk_size,
        len(futures),
    )
    step = max(1, int(progress_every))
    times: list[float] = []
    buffer: list[dict] = []
    seen: set[str] = set()
    processed = 0
    for fut in as_completed(futures):
        results = fut.result()
        for ticker, r, elapsed in results:
            processed += 1
            times.append(elapsed)
            if r is None:
                skipped_missing_data += 1
            elif r and ticker not in seen:
                seen.add(ticker)
                if SCAN_BATCH_WRITES:
                    buffer.append(r)
                    if len(buffer) >= 50:
                        rows.extend(buffer)
                        buffer.clear()
                else:
                    rows.append(r)
            if progress_cb and (processed % step == 0 or processed == total):
                progress_cb(processed, total, f"Scanning {processed}/{total}")
            if processed % 10 == 0:
                el = _perf_counter() - start
                logger.info(
                    "scan progress %d/%d elapsed=%.1fs avg_per_task=%.2fs",
                    processed,
                    total,
                    el,
                    el / processed,
                )
    if buffer:
        rows.extend(buffer)
    logger.info(
        "scan complete %d symbols in %.1fs",
        total,
        _perf_counter() - start,
    )

    try:
        scan_min_hit = float(params.get("scan_min_hit", 0.0))
        scan_max_dd = float(params.get("scan_max_dd", 100.0))
    except Exception:
        scan_min_hit, scan_max_dd = 0.0, 100.0

    rows = [
        r
        for r in rows
        if (r.get("hit_pct", 0.0) >= scan_min_hit)
        and (r.get("avg_dd_pct", 100.0) <= scan_max_dd)
    ]

    rows = _sort_by_lb95_roi_support(rows)

    if sort_key == "ticker":
        rows.sort(key=lambda r: (r.get("ticker") or ""))
    elif sort_key == "roi":
        rows.sort(
            key=lambda r: (
                (
                    r.get("avg_roi")
                    if r.get("avg_roi") not in (None, "")
                    else r.get("avg_roi_pct", 0.0)
                ),
                r.get("hit_pct", 0.0),
                r.get("support", 0),
            ),
            reverse=True,
        )
    elif sort_key == "hit":
        rows.sort(
            key=lambda r: (
                r.get("hit_pct", 0.0),
                (
                    r.get("avg_roi")
                    if r.get("avg_roi") not in (None, "")
                    else r.get("avg_roi_pct", 0.0)
                ),
                r.get("support", 0),
            ),
            reverse=True,
        )

    _update_heatmap(rows)

    duration = _perf_counter() - start
    scan_duration.observe(duration)
    scan_tickers.inc(len(tickers))
    avg_ms = (statistics.mean(times) * 1000.0) if times else 0.0
    p95_ms = (
        sorted(times)[max(int(len(times) * 0.95) - 1, 0)] * 1000.0 if times else 0.0
    )
    metrics = {
        "coverage_ms": cov_elapsed,
        "fetch_ms": fetch_elapsed,
        "symbols_no_gap": len(no_gap),
        "symbols_gap": len(need_fetch),
        "avg_per_symbol_ms": avg_ms,
        "p95_per_symbol_ms": p95_ms,
        "db_reads": 1,
        "db_writes": 0,
    }
    logger.info(
        "scan completed total=%d no_gap=%d gaps=%d avg_ms=%.1f p95_ms=%.1f db_reads=%d db_writes=%d skipped_missing_data=%d",
        len(tickers),
        len(no_gap),
        len(need_fetch),
        avg_ms,
        p95_ms,
        metrics["db_reads"],
        metrics["db_writes"],
        skipped_missing_data,
    )

    return rows, skipped_missing_data, metrics


def _sort_rows(rows, sort_key):
    if not rows or not sort_key:
        return rows
    keymap = {
        "ticker": lambda r: (r.get("ticker") or ""),
        "roi": lambda r: (r.get("avg_roi_pct") or 0.0),
        "hit": lambda r: (r.get("hit_pct") or 0.0),
    }
    keyfn = keymap.get(sort_key)
    if not keyfn:
        return rows
    reverse = sort_key != "ticker"
    return sorted(rows, key=keyfn, reverse=reverse)


def _send_email(
    st: dict,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    *,
    list_field: str = "recipients",
    allow_sms: bool = True,
) -> None:
    """Send an email using settings stored in the database.

    This helper mirrors the logic used by the desktop application: the
    configured SMTP user/password are expected to work with Gmail.  The
    function silently returns if mandatory settings are missing so the
    scanner can proceed without failing.
    """

    cfg, missing, has_any = _smtp_config_status(st)
    recips = cfg["recipients"] if list_field == "recipients" else [
        r.strip()
        for r in (st.get(list_field) or "").split(",")
        if r.strip()
    ]
    if not allow_sms:
        from services.notifications import is_carrier_address

        recips = [r for r in recips if not is_carrier_address(r)]
    if list_field != "recipients":
        missing = [m for m in missing if m != "recipients"]

    if not recips:
        if list_field == "recipients":
            missing = [m for m in missing if m != "recipients"] + ["recipients"]
        elif not has_any:
            return
        else:
            missing = missing + [list_field]

    if missing:
        if has_any or list_field != "recipients":
            logger.warning(
                "SMTP configuration incomplete: %s",
                _format_missing_fields(sorted(set(missing))),
            )
        return

    payload = html_body or body
    result = send_email_smtp(
        cfg["host"],
        cfg["port"],
        cfg["user"],
        cfg["password"],
        cfg["mail_from"] or cfg["user"],
        recips,
        subject,
        payload,
    )
    if not result.get("ok"):
        logger.warning("SMTP send failed: %s", result.get("error"))
        return
    logger.info(
        "email sent via smtp list=%s recipients=%d message_id=%s",
        list_field,
        len(recips),
        result.get("message_id"),
    )


def compile_weekly_digest(
    db: sqlite3.Cursor, ts: Optional[pd.Timestamp] = None
) -> tuple[str, str]:
    ts = pd.Timestamp(ts or now_et())
    week_start = (ts - pd.Timedelta(days=ts.weekday())).date()
    start_iso = pd.Timestamp(week_start, tz=TZ).isoformat()

    db.execute("SELECT COUNT(*) FROM forward_tests WHERE created_at>=?", (start_iso,))
    new_count = db.fetchone()[0]
    db.execute(
        "SELECT COUNT(*) FROM forward_tests WHERE status='ok' AND updated_at>=?",
        (start_iso,),
    )
    resolved_count = db.fetchone()[0]
    db.execute(
        "SELECT AVG(hit_forward), AVG(roi_forward) FROM forward_tests WHERE created_at>=?",
        (start_iso,),
    )
    hit_avg, roi_avg = db.fetchone()
    hit_avg = hit_avg or 0.0
    roi_avg = roi_avg or 0.0

    subject = (
        f"[Forward Digest] Week of {week_start:%Y-%m-%d} – {new_count} New | {resolved_count} Resolved | "
        f"Hit% {hit_avg:.0f} | Avg ROI {roi_avg:.1f}%"
    )

    db.execute(
        "SELECT reason, COUNT(*) FROM guardrail_skips WHERE created_at>=? GROUP BY reason",
        (start_iso,),
    )
    guard = {r[0]: r[1] for r in db.fetchall()}

    lines = [
        f"New tests: {new_count}",
        f"Resolved tests: {resolved_count}",
        f"Hit%: {hit_avg:.0f}",
        f"Avg ROI: {roi_avg:.1f}%",
        f"Guardrails - earnings: {guard.get('earnings',0)}, low_liquidity: {guard.get('low_liquidity',0)}",
        "Top Winners:",
    ]
    db.execute(
        "SELECT ticker, roi_forward FROM forward_tests WHERE created_at>=? ORDER BY roi_forward DESC LIMIT 3",
        (start_iso,),
    )
    for sym, roi in db.fetchall():
        lines.append(f"  {sym} {roi or 0:.1f}%")
    lines.append("Top Losers:")
    db.execute(
        "SELECT ticker, dd_forward FROM forward_tests WHERE created_at>=? ORDER BY dd_forward DESC LIMIT 3",
        (start_iso,),
    )
    for sym, dd in db.fetchall():
        lines.append(f"  {sym} {dd or 0:.1f}% DD")
    lines.append("Link: /forward?filter=last7d")
    body = "\n".join(lines)
    return subject, body


def send_weekly_digest(db: sqlite3.Cursor) -> None:
    st = get_settings(db)
    subject, body = compile_weekly_digest(db)
    _send_email(st, subject, body)


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        {"canonical_path": "/"},
    )


@router.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"active_tab": "scanner", "canonical_path": "/scanner"},
    )


@router.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse(
        request,
        "about.html",
        {"canonical_path": "/about"},
    )


@router.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse(
        request,
        "contact.html",
        {"canonical_path": "/contact"},
    )


@router.get("/heatmap.json")
def heatmap_json():
    now = time.time()
    if (
        isinstance(_HEATMAP_CACHE.get("data"), dict)
        and now < float(_HEATMAP_CACHE.get("expires", 0.0))
    ):
        return _HEATMAP_CACHE["data"]

    data = _LATEST_HEATMAP or _EMPTY_HEATMAP
    payload = {
        "index": list(data.get("index", [])),
        "columns": list(data.get("columns", [])),
        "values": [list(row) for row in data.get("values", [])],
        "meta": dict(data.get("meta", {})),
    }
    _HEATMAP_CACHE["data"] = payload
    _HEATMAP_CACHE["expires"] = now + 300.0
    return payload


@router.get("/heatmap", response_class=HTMLResponse)
def heatmap_page(request: Request):
    return templates.TemplateResponse(request, "heatmap.html", {"active_tab": "heatmap"})


@router.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request, db=Depends(get_db)):
    ensure_favorite_directions(db)
    db.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = [_normalize_favorite(row_to_dict(r, db)) for r in db.fetchall()]
    for f in favs:
        f["direction"] = canonical_direction(f.get("direction")) or "UP"
        f["avg_roi_pct"] = _favorite_roi_value(f.get("roi_snapshot"))
        f["hit_pct"] = f.get("hit_pct_snapshot")
        f["avg_dd_pct"] = f.get("dd_pct_snapshot")
        if f.get("rule_snapshot"):
            f["rule"] = f.get("rule_snapshot")
    return templates.TemplateResponse(
        request,
        "favorites.html",
        {"favorites": favs, "active_tab": "favorites"},
    )


def _window_to_minutes(value: float, unit: str) -> int:
    unit = (unit or "").lower()
    if unit.startswith("min"):
        return int(value)
    if unit.startswith("hour"):
        return int(value * 60)
    if unit.startswith("day"):
        return int(value * 60 * 24)
    if unit.startswith("week"):
        return int(value * 60 * 24 * 7)
    return int(value * 60)


def _create_forward_test(db: sqlite3.Cursor, fav: dict) -> None:
    allowed, flags = check_guardrails(fav.get("ticker"))
    if not allowed:
        logger.info(
            "forward test for %s skipped due to guardrails: %s",
            fav.get("ticker"),
            ",".join(flags),
        )
        db.execute(
            "INSERT INTO guardrail_skips(ticker, reason) VALUES (?, ?)",
            (fav.get("ticker"), ",".join(flags)),
        )
        return

    start, end = window_from_lookback(fav.get("lookback_years", 1.0))
    data = get_prices([fav["ticker"]], fav.get("interval", "15m"), start, end).get(
        fav["ticker"]
    )
    if data is None or getattr(data, "empty", True):
        return
    last_bar = data.iloc[-1]
    ts = last_bar.name
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    entry_ts = ts.astimezone(TZ).isoformat()
    entry_price = float(last_bar["Close"])
    window_minutes = _window_to_minutes(
        fav.get("window_value", 4.0), fav.get("window_unit", "Hours")
    )

    db.execute(
        """SELECT version, target_pct, stop_pct, window_minutes, rule
            FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1""",
        (fav["id"],),
    )
    row = db.fetchone()
    version = 1
    if row:
        version = row[0]
        if (
            float(row[1]) != float(fav.get("target_pct", 1.0))
            or float(row[2]) != float(fav.get("stop_pct", 0.5))
            or int(row[3]) != window_minutes
            or (row[4] or "") != fav.get("rule")
        ):
            db.execute(
                "UPDATE forward_tests SET status='closed' WHERE fav_id=? AND version=?",
                (fav["id"], row[0]),
            )
            version = row[0] + 1

    now_iso = now_et().isoformat()
    db.execute(
        """INSERT INTO forward_tests
            (fav_id, ticker, direction, interval, rule, version, entry_price,
             target_pct, stop_pct, window_minutes, status, roi_forward, hit_forward, dd_forward,
             roi_1, roi_3, roi_5, roi_expiry, mae, mfe, time_to_hit, time_to_stop,
             exit_reason, bars_to_exit, max_drawdown_pct, max_runup_pct, r_multiple,
             option_expiry, option_strike, option_delta, option_roi_proxy,
             last_run_at, next_run_at, runs_count, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0.0, NULL, 0.0,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, ?, 0.0,
                    NULL, NULL, 0, NULL, ?, ?)""",
        (
            fav["id"],
            fav["ticker"],
            canonical_direction(fav.get("direction")) or "UP",
            fav.get("interval", "15m"),
            fav.get("rule"),
            version,
            entry_price,
            fav.get("target_pct", 1.0),
            fav.get("stop_pct", 0.5),
            window_minutes,
            fav.get("delta"),
            entry_ts,
            now_iso,
        ),
    )
    db.connection.commit()


def _update_forward_tests(db: sqlite3.Cursor) -> None:
    db.execute(
        """SELECT id, fav_id, ticker, direction, interval, rule, created_at, entry_price,
                  target_pct, stop_pct, window_minutes, status, option_delta
               FROM forward_tests
               WHERE status IN ('queued','running')"""
    )
    rows = [row_to_dict(r, db) for r in db.fetchall()]
    for row in rows:
        now_iso = now_et().isoformat()
        entry_history_ts: str | None = None
        try:
            db.execute(
                "UPDATE forward_tests SET status='running', last_run_at=?, updated_at=?, runs_count=runs_count+1 WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
            start, end = window_from_lookback(1.0)
            data = get_prices([row["ticker"]], row["interval"], start, end).get(
                row["ticker"]
            )
            if data is None or getattr(data, "empty", True):
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            entry_signal_ts = pd.Timestamp(row["created_at"])
            after = data[data.index > entry_signal_ts]
            if after.empty or "Open" not in after.columns:
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            entry_bar = after.iloc[0]
            try:
                entry_underlying = float(entry_bar["Open"])
            except (TypeError, ValueError):
                db.execute(
                    "UPDATE forward_tests SET status='queued' WHERE id=?",
                    (row["id"],),
                )
                continue
            entry_time = pd.Timestamp(entry_bar.name)
            window_minutes = int(row.get("window_minutes") or 0)
            expire_ts = entry_time + pd.Timedelta(minutes=window_minutes)
            eligible = after[after.index <= expire_ts]
            if eligible.empty:
                db.execute(
                    "UPDATE forward_tests SET status='queued', entry_price=? WHERE id=?",
                    (entry_underlying, row["id"]),
                )
                continue
            entry_history_ts = entry_time.isoformat()
            log_forward_entry(
                db,
                row.get("fav_id"),
                entry_history_ts,
                entry_underlying,
                forward_rule_hash(row.get("rule")),
            )
            direction = canonical_direction(row.get("direction")) or "UP"
            side = 1 if direction == "UP" else -1
            slip = FORWARD_SLIPPAGE
            entry_fill = entry_underlying * (1 + slip * side)
            exit_factor = 1 - slip * side
            target_pct = float(row.get("target_pct") or 0.0)
            stop_pct = float(row.get("stop_pct") or 0.0)
            if side == 1:
                target_level = entry_underlying * (1 + target_pct / 100.0)
                stop_level = entry_underlying * (1 - stop_pct / 100.0)
            else:
                target_level = entry_underlying * (1 - target_pct / 100.0)
                stop_level = entry_underlying * (1 + stop_pct / 100.0)
            closes = eligible["Close"].astype(float)
            highs = eligible["High"].astype(float)
            lows = eligible["Low"].astype(float)
            roi_closes = side * ((closes * exit_factor - entry_fill) / entry_fill) * 100.0
            favorable_prices = highs if side == 1 else lows
            adverse_prices = lows if side == 1 else highs
            roi_favorable = side * (
                (favorable_prices * exit_factor - entry_fill) / entry_fill
            ) * 100.0
            roi_adverse = side * (
                (adverse_prices * exit_factor - entry_fill) / entry_fill
            ) * 100.0
            roi_forward = float(roi_closes.iloc[-1]) if not roi_closes.empty else 0.0
            roi_curve = {1: None, 3: None, 5: None}
            for n in roi_curve:
                if len(roi_closes) >= n:
                    roi_curve[n] = float(roi_closes.iloc[n - 1])
            has_expired = after.index[-1] >= expire_ts
            if side == 1:
                hit_cond = highs >= target_level
                stop_cond = lows <= stop_level
            else:
                hit_cond = lows <= target_level
                stop_cond = highs >= stop_level
            hit_time = hit_cond[hit_cond].index[0] if hit_cond.any() else None
            stop_time = stop_cond[stop_cond].index[0] if stop_cond.any() else None
            status = "ok"
            hit_pct = None
            exit_reason = None
            exit_time = None
            exit_level = None

            def _roi_from_level(level: float) -> float:
                return float(
                    side * (((level * exit_factor) - entry_fill) / entry_fill) * 100.0
                )

            if hit_time is not None and (stop_time is None or hit_time < stop_time):
                exit_time = hit_time
                exit_level = float(target_level)
                exit_reason = "target"
                hit_pct = 100.0
                roi_forward = _roi_from_level(target_level)
            elif stop_time is not None and (hit_time is None or stop_time <= hit_time):
                exit_time = stop_time
                exit_level = float(stop_level)
                exit_reason = "stop"
                hit_pct = 0.0
                roi_forward = _roi_from_level(stop_level)
            elif not has_expired:
                status = "queued"
            else:
                exit_time = eligible.index[-1]
                exit_level = float(closes.iloc[-1])
                exit_reason = "timeout"
                hit_pct = 0.0
                roi_forward = float(roi_closes.iloc[-1])
            entry_price_update = entry_underlying
            roi_expiry = None
            bars_to_exit = None
            time_to_hit = None
            time_to_stop = None
            if exit_reason:
                status = "ok"
                exit_idx = eligible.index.get_loc(exit_time)
                bars_to_exit = int(exit_idx + 1)
                for n in roi_curve:
                    if exit_idx <= n - 1:
                        roi_curve[n] = roi_forward
                roi_expiry = roi_forward
                roi_adverse_slice = roi_adverse.iloc[: exit_idx + 1]
                roi_favorable_slice = roi_favorable.iloc[: exit_idx + 1]
            else:
                exit_idx = len(eligible) - 1
                roi_adverse_slice = roi_adverse
                roi_favorable_slice = roi_favorable
                roi_expiry = float(roi_closes.iloc[-1]) if has_expired else None
            mae = float(roi_adverse_slice.min()) if not roi_adverse_slice.empty else 0.0
            mfe = float(roi_favorable_slice.max()) if not roi_favorable_slice.empty else 0.0
            dd_forward = float(max(0.0, -mae))
            max_drawdown_pct = dd_forward
            max_runup_pct = float(max(0.0, mfe))
            if exit_reason and entry_history_ts:
                exit_iso = (
                    pd.Timestamp(exit_time).isoformat() if exit_time is not None else None
                )
                log_forward_exit(
                    db,
                    row.get("fav_id"),
                    entry_history_ts,
                    exit_iso,
                    exit_level,
                    exit_reason,
                    (roi_forward / 100.0) if roi_forward is not None else None,
                    bars_to_exit,
                    (dd_forward / 100.0) if dd_forward is not None else None,
                )
            if exit_reason:
                entry_time_dt = pd.Timestamp(entry_time)
                elapsed = (
                    (pd.Timestamp(exit_time) - entry_time_dt).total_seconds() / 60.0
                )
                if exit_reason == "target":
                    time_to_hit = elapsed
                elif exit_reason == "stop":
                    time_to_stop = elapsed
            stop_fill = stop_level * exit_factor
            r_multiple = None
            if exit_reason:
                exit_fill_price = exit_level * exit_factor if exit_level is not None else None
                if exit_fill_price is not None:
                    if side == 1:
                        risk = entry_fill - stop_fill
                        if risk > 0:
                            r_multiple = (exit_fill_price - entry_fill) / risk
                    else:
                        risk = stop_fill - entry_fill
                        if risk > 0:
                            r_multiple = (entry_fill - exit_fill_price) / risk
            delta = row.get("option_delta")
            option_roi_proxy = roi_forward / delta if delta else None
            run_iso = now_et().isoformat()
            db.execute(
                """UPDATE forward_tests
                       SET entry_price=?, roi_forward=?, dd_forward=?, status=?, hit_forward=?, roi_1=?, roi_3=?, roi_5=?, roi_expiry=?, mae=?, mfe=?, time_to_hit=?, time_to_stop=?, exit_reason=?, bars_to_exit=?, max_drawdown_pct=?, max_runup_pct=?, r_multiple=?, option_roi_proxy=?, last_run_at=?, next_run_at=?, updated_at=?
                       WHERE id=?""",
                (
                    entry_price_update,
                    roi_forward,
                    dd_forward,
                    status,
                    hit_pct,
                    roi_curve[1],
                    roi_curve[3],
                    roi_curve[5],
                    roi_expiry,
                    mae,
                    mfe,
                    time_to_hit,
                    time_to_stop,
                    exit_reason,
                    bars_to_exit,
                    max_drawdown_pct,
                    max_runup_pct,
                    r_multiple,
                    option_roi_proxy,
                    run_iso,
                    run_iso,
                    now_iso,
                    row["id"],
                ),
            )
        except Exception:
            logger.exception("Forward test %s failed", row["id"])
            db.execute(
                "UPDATE forward_tests SET status='error', last_run_at=?, updated_at=? WHERE id=?",
                (now_iso, now_iso, row["id"]),
            )
    db.connection.commit()


FORWARD_HISTORY_LIMIT = 5
try:
    _forward_summary_limit_raw = int(
        os.getenv("FORWARD_SUMMARY_LIMIT", "20") or "20"
    )
except ValueError:
    _forward_summary_limit_raw = 20
FORWARD_SUMMARY_LIMIT = max(0, _forward_summary_limit_raw)


def _serialize_forward_favorite(
    fav: dict[str, Any],
    db_cursor: sqlite3.Cursor | None = None,
) -> dict[str, Any]:
    forward_row = fav.get("forward") if isinstance(fav, dict) else None
    forward: dict[str, Any] | None = None
    if isinstance(forward_row, dict) and forward_row:
        forward = {
            "id": forward_row.get("id"),
            "version": _coerce_int(forward_row.get("version")),
            "status": (forward_row.get("status") or None),
            "roi_pct": _coerce_float(forward_row.get("roi_forward")),
            "hit_pct": _coerce_float(forward_row.get("hit_forward")),
            "dd_pct": _coerce_float(forward_row.get("dd_forward")),
            "roi_1": _coerce_float(forward_row.get("roi_1")),
            "roi_3": _coerce_float(forward_row.get("roi_3")),
            "roi_5": _coerce_float(forward_row.get("roi_5")),
            "roi_expiry": _coerce_float(forward_row.get("roi_expiry")),
            "mae": _coerce_float(forward_row.get("mae")),
            "mfe": _coerce_float(forward_row.get("mfe")),
            "time_to_hit": _coerce_float(forward_row.get("time_to_hit")),
            "time_to_stop": _coerce_float(forward_row.get("time_to_stop")),
            "exit_reason": forward_row.get("exit_reason"),
            "bars_to_exit": _coerce_int(forward_row.get("bars_to_exit")),
            "max_drawdown_pct": _coerce_float(forward_row.get("max_drawdown_pct")),
            "max_runup_pct": _coerce_float(forward_row.get("max_runup_pct")),
            "r_multiple": _coerce_float(forward_row.get("r_multiple")),
            "option_roi_proxy": _coerce_float(forward_row.get("option_roi_proxy")),
            "runs_count": _coerce_int(forward_row.get("runs_count")),
            "created_at": forward_row.get("created_at"),
            "updated_at": forward_row.get("updated_at"),
            "last_run_at": forward_row.get("last_run_at"),
            "events": _forward_events(forward_row),
        }
    history: list[dict[str, Any]] = []
    summary = get_cached_forward_summary(None, [])
    fav_id = fav.get("id")
    history_rows: list[dict[str, Any]] = []
    if fav_id is not None:
        fetch_limit = max(FORWARD_HISTORY_LIMIT, FORWARD_SUMMARY_LIMIT)
        if db_cursor is not None:
            history_rows = get_forward_history_for_cursor(
                db_cursor, fav_id, fetch_limit
            )
        else:
            from services.forward_runs import get_forward_history  # local import to avoid cycle

            history_rows = get_forward_history(str(fav_id), fetch_limit)
        summary = get_cached_forward_summary(
            fav_id, history_rows[:FORWARD_SUMMARY_LIMIT]
        )
    history_slice = history_rows[:FORWARD_HISTORY_LIMIT]
    current_hash = forward_rule_hash(fav.get("rule"))
    for run in history_slice:
        item = dict(run)
        rule_hash = item.get("rule_hash")
        item["rule_mismatch"] = bool(
            rule_hash and current_hash and rule_hash != current_hash
        )
        history.append(item)

    return {
        "id": fav.get("id"),
        "ticker": fav.get("ticker"),
        "direction": canonical_direction(fav.get("direction")) or "UP",
        "interval": fav.get("interval"),
        "rule": fav.get("rule"),
        "rule_snapshot": fav.get("rule_snapshot"),
        "target_pct": _coerce_float(fav.get("target_pct")),
        "stop_pct": _coerce_float(fav.get("stop_pct")),
        "window_value": _coerce_float(fav.get("window_value")),
        "window_unit": fav.get("window_unit"),
        "lookback_years": _coerce_float(fav.get("lookback_years")),
        "lookback_display": fav.get("lookback_display"),
        "support_count": fav.get("support_count"),
        "support_display": fav.get("support_display"),
        "roi_snapshot": _favorite_roi_value(fav.get("roi_snapshot")),
        "hit_pct_snapshot": _coerce_float(fav.get("hit_pct_snapshot")),
        "dd_pct_snapshot": _coerce_float(fav.get("dd_pct_snapshot")),
        "snapshot_at": fav.get("snapshot_at"),
        "forward": forward,
        "summary": summary,
        "forward_history": history,
    }


def _fetch_favorite_metadata(
    db_cursor: sqlite3.Cursor, favorite_id: Any
) -> dict[str, Any] | None:
    try:
        fav_id = int(favorite_id)
    except (TypeError, ValueError):
        return None
    db_cursor.execute(
        "SELECT * FROM favorites WHERE id=?",
        (fav_id,),
    )
    row = db_cursor.fetchone()
    if not row:
        return None
    return row_to_dict(row, db_cursor)


def _favorite_visible_to_request(
    favorite: dict[str, Any] | None, request: Request | None
) -> bool:
    if not favorite:
        return False
    owner_keys = ("owner_id", "user_id", "account_id", "owner")
    for key in owner_keys:
        if key in favorite and favorite[key] not in (None, ""):
            request_owner = None
            if request is not None:
                request_owner = getattr(getattr(request, "state", None), key, None)
                if request_owner is None:
                    request_owner = getattr(getattr(request, "state", None), "user_id", None)
            if request_owner is None or str(request_owner) != str(favorite[key]):
                return False
    return True


def _load_forward_favorites(
    db: sqlite3.Cursor, ids: list[int] | None = None
) -> list[dict[str, Any]]:
    ensure_favorite_directions(db)
    if ids:
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT * FROM favorites WHERE id IN ({placeholders})"
        db.execute(query, tuple(ids))
    else:
        db.execute("SELECT * FROM favorites ORDER BY id DESC")
    rows = [_normalize_favorite(row_to_dict(row, db)) for row in db.fetchall()]
    if not rows:
        return []

    fav_ids: list[int] = []
    seen: set[int] = set()
    for fav in rows:
        fid = _coerce_int(fav.get("id"))
        if fid is None or fid in seen:
            fav["forward"] = None
            continue
        seen.add(fid)
        fav_ids.append(fid)

    forward_map: dict[int, dict[str, Any]] = {}
    if fav_ids:
        placeholders = ",".join("?" for _ in fav_ids)
        query = (
            "SELECT ft.* FROM forward_tests AS ft "
            "JOIN ("
            " SELECT fav_id, MAX(id) AS max_id"
            " FROM forward_tests"
            f" WHERE fav_id IN ({placeholders})"
            " GROUP BY fav_id"
            " ) AS latest"
            " ON ft.fav_id = latest.fav_id AND ft.id = latest.max_id"
        )
        db.execute(query, tuple(fav_ids))
        forward_rows = [row_to_dict(row, db) for row in db.fetchall()]
        for row in forward_rows:
            fid = _coerce_int(row.get("fav_id"))
            if fid is not None:
                forward_map[fid] = row

        support_counts: dict[int, int] = {}
        support_query = (
            "SELECT fav_id, created_at, window_minutes, status, exit_reason "
            f"FROM forward_tests WHERE fav_id IN ({placeholders}) ORDER BY created_at"
        )
        db.execute(support_query, tuple(fav_ids))
        support_rows = [row_to_dict(row, db) for row in db.fetchall()]
        grouped_support: dict[int, list[dict[str, Any]]] = {}
        for row in support_rows:
            fid = _coerce_int(row.get("fav_id"))
            if fid is None:
                continue
            grouped_support.setdefault(fid, []).append(row)
        now_ts = pd.Timestamp(now_et()).tz_convert(TZ)
        for fav in rows:
            fid = _coerce_int(fav.get("id"))
            if fid is None:
                continue
            tests = grouped_support.get(fid)
            if not tests:
                continue
            lookback = _coerce_float(fav.get("lookback_years")) or 1.0
            window_start = now_ts - pd.Timedelta(days=365 * lookback)
            count = 0
            last_time: pd.Timestamp | None = None
            for test in tests:
                status_val = (test.get("status") or "").lower()
                if status_val != "ok":
                    continue
                created_raw = test.get("created_at")
                if not created_raw:
                    continue
                created_ts = pd.Timestamp(created_raw)
                if created_ts.tzinfo is None:
                    created_ts = created_ts.tz_localize(TZ)
                else:
                    created_ts = created_ts.tz_convert(TZ)
                if created_ts < window_start:
                    continue
                window_minutes = int(test.get("window_minutes") or 0)
                threshold = (
                    last_time + pd.Timedelta(minutes=window_minutes)
                    if last_time is not None
                    else None
                )
                if threshold is None or created_ts >= threshold:
                    count += 1
                    last_time = created_ts
            support_counts[fid] = count

        for fav in rows:
            fid = _coerce_int(fav.get("id"))
            if fid is None:
                continue
            if fid in support_counts:
                count = support_counts[fid]
                fav["support_count"] = count
                fav["support_display"] = str(count)

    for fav in rows:
        fid = _coerce_int(fav.get("id"))
        fav["forward"] = forward_map.get(fid) if fid is not None else None

    if ids:
        fav_map = {
            int(fav.get("id")): fav
            for fav in rows
            if fav.get("id") is not None
        }
        ordered: list[dict[str, Any]] = []
        for fid in ids:
            if fid in fav_map:
                ordered.append(fav_map[fid])
        return ordered
    return rows


def _queue_forward_tests_for(
    db: sqlite3.Cursor, favorites: list[dict[str, Any]]
) -> int:
    created = 0
    for fav in favorites:
        fav_id = fav.get("id")
        if fav_id is None:
            continue
        db.execute(
            "SELECT status FROM forward_tests WHERE fav_id=? ORDER BY id DESC LIMIT 1",
            (fav_id,),
        )
        row = db.fetchone()
        if row is None or row["status"] in ("ok", "error"):
            _create_forward_test(db, fav)
            created += 1
    _update_forward_tests(db)
    return created


@router.get("/forward", response_class=HTMLResponse)
def forward_page(request: Request, db=Depends(get_db)):
    try:
        favorites = _load_forward_favorites(db)
        _queue_forward_tests_for(db, favorites)
    except Exception:
        logger.exception("Failed to prepare forward tests for page load")
    return templates.TemplateResponse(
        request, "forward.html", {"active_tab": "forward"}
    )


@router.get("/paper", response_class=HTMLResponse)
def paper_page(request: Request, db=Depends(get_db)):
    summary = paper_trading.get_summary(db)
    settings = paper_trading.load_settings(db)
    scalper_lf_status = scalper_lf.status_payload(db)
    scalper_hf_status = scalper_hf.status_payload(db)
    lf_equity_seed = [
        {"ts": point.ts, "balance": point.balance}
        for point in scalper_lf.get_equity_points(db, "1m")
    ]
    hf_equity_seed = [
        {"ts": point.ts, "balance": point.balance}
        for point in scalper_hf.get_equity_points(db, "1m")
    ]
    lf_activity_seed = scalper_lf.list_activity(db, limit=250)
    hf_activity_seed = scalper_hf.list_activity(db, limit=250)
    return templates.TemplateResponse(
        request,
        "paper.html",
        {
            "active_tab": "paper",
            "canonical_path": "/paper",
            "summary": summary,
            "paper_settings": settings,
            "scalper_lf_status": scalper_lf_status,
            "scalper_lf_equity": lf_equity_seed,
            "scalper_lf_activity": lf_activity_seed,
            "scalper_hf_status": scalper_hf_status,
            "scalper_hf_equity": hf_equity_seed,
            "scalper_hf_activity": hf_activity_seed,
        },
    )


@router.get("/paper/favorites", response_class=HTMLResponse)
def paper_favorites_page(request: Request, db=Depends(get_db)):
    settings = favorites_sim.load_settings(db)
    status = favorites_sim.status_payload(db)
    summary = favorites_sim.summary_payload(db)
    equity_seed = favorites_sim.get_equity_points(db, "all")
    activity_seed = favorites_sim.activity_payload(db)
    universe_count = favorites_sim.favorites_count(db)
    return templates.TemplateResponse(
        request,
        "paper_favorites.html",
        {
            "active_tab": "paper",
            "canonical_path": "/paper/favorites",
            "favorites_settings": settings,
            "favorites_status": status,
            "favorites_summary": summary,
            "favorites_equity": equity_seed,
            "favorites_activity": activity_seed,
            "favorites_universe_count": universe_count,
        },
    )


def _enforce_paper_rate_limit(key: str, interval: float = 1.0) -> None:
    now_ts = time.time()
    with _PAPER_RATE_LOCK:
        previous = _PAPER_RATE_LIMIT.get(key)
        if previous is not None and now_ts - previous < interval:
            raise HTTPException(status_code=429, detail="Paper trading endpoint throttled")
        _PAPER_RATE_LIMIT[key] = now_ts


@router.get("/paper/summary")
def paper_summary(db=Depends(get_db)):
    return paper_trading.get_summary(db)


@router.get("/paper/equity")
def paper_equity(range: str = Query("1m"), db=Depends(get_db)):
    allowed = {"1d", "1w", "1m", "1y"}
    range_key = (range or "").lower()
    if range_key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid range")
    points = paper_trading.get_equity_points(db, range_key)
    return {
        "range": range_key,
        "points": [{"ts": p.ts, "balance": p.balance} for p in points],
    }


@router.get("/paper/trades")
def paper_trades(
    status: str = Query("all"),
    export: str | None = Query(default=None),
    db=Depends(get_db),
):
    status_key = (status or "").lower()
    allowed = {"open", "closed", "all"}
    if status_key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid status filter")
    status_filter = None if status_key == "all" else status_key
    if (export or "").lower() == "csv":
        csv_text, filename = paper_trading.export_trades_csv(db, status_filter)
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return Response(content=csv_text, media_type="text/csv", headers=headers)
    return {"trades": paper_trading.list_trades(db, status_filter)}


@router.post("/paper/start")
def paper_start(db=Depends(get_db)):
    _enforce_paper_rate_limit("paper_start")
    paper_trading.start_engine(db)
    return paper_trading.get_summary(db)


@router.post("/paper/stop")
def paper_stop(db=Depends(get_db)):
    _enforce_paper_rate_limit("paper_stop")
    paper_trading.stop_engine(db)
    return paper_trading.get_summary(db)


@router.post("/paper/restart")
def paper_restart(db=Depends(get_db)):
    _enforce_paper_rate_limit("paper_restart", interval=2.0)
    paper_trading.restart_engine(db)
    return paper_trading.get_summary(db)


@router.get("/api/paper/scalper/lf/status")
def scalper_lf_status(db=Depends(get_db)):
    return scalper_lf.status_payload(db)


@router.post("/api/paper/scalper/lf/start")
def scalper_lf_start(db=Depends(get_db)):
    scalper_lf.start_engine(db)
    return scalper_lf.status_payload(db)


@router.post("/api/paper/scalper/lf/stop")
def scalper_lf_stop(db=Depends(get_db)):
    scalper_lf.stop_engine(db)
    return scalper_lf.status_payload(db)


@router.post("/api/paper/scalper/lf/restart")
def scalper_lf_restart(db=Depends(get_db)):
    scalper_lf.restart_engine(db)
    return scalper_lf.status_payload(db)


@router.get("/api/paper/scalper/lf/equity.json")
def scalper_lf_equity(range: str = Query("1m"), db=Depends(get_db)):
    range_key = (range or "").lower()
    allowed = {"1d", "1w", "1m", "1y"}
    if range_key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid range")
    points = scalper_lf.get_equity_points(db, range_key)
    return {
        "range": range_key,
        "points": [
            {"ts": point.ts, "balance": point.balance}
            for point in points
        ],
    }


@router.get("/api/paper/scalper/lf/activity.csv")
def scalper_lf_activity_csv(db=Depends(get_db)):
    csv_text, filename = scalper_lf.export_activity_csv(db)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_text, media_type="text/csv", headers=headers)


@router.get("/api/paper/scalper/lf/activity")
def scalper_lf_activity(db=Depends(get_db)):
    return {"rows": scalper_lf.list_activity(db, limit=500)}


@router.get("/api/paper/scalper/hf/status")
def scalper_hf_status(db=Depends(get_db)):
    return scalper_hf.status_payload(db)


@router.post("/api/paper/scalper/hf/start")
def scalper_hf_start(db=Depends(get_db)):
    scalper_hf.start_engine(db)
    return scalper_hf.status_payload(db)


@router.post("/api/paper/scalper/hf/stop")
def scalper_hf_stop(db=Depends(get_db)):
    scalper_hf.stop_engine(db)
    return scalper_hf.status_payload(db)


@router.post("/api/paper/scalper/hf/restart")
def scalper_hf_restart(db=Depends(get_db)):
    scalper_hf.restart_engine(db)
    return scalper_hf.status_payload(db)


@router.get("/api/paper/scalper/hf/equity.json")
def scalper_hf_equity(range: str = Query("1m"), db=Depends(get_db)):
    range_key = (range or "").lower()
    allowed = {"1d", "1w", "1m", "1y"}
    if range_key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid range")
    points = scalper_hf.get_equity_points(db, range_key)
    return {
        "range": range_key,
        "points": [
            {"ts": point.ts, "balance": point.balance}
            for point in points
        ],
    }


@router.get("/api/paper/scalper/hf/activity.csv")
def scalper_hf_activity_csv(db=Depends(get_db)):
    csv_text, filename = scalper_hf.export_activity_csv(db)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_text, media_type="text/csv", headers=headers)


@router.get("/api/paper/scalper/hf/activity")
def scalper_hf_activity(db=Depends(get_db)):
    return {"rows": scalper_hf.list_activity(db, limit=500)}


@router.get("/api/paper/favorites/status")
def favorites_sim_status(db=Depends(get_db)):
    status = favorites_sim.status_payload(db)
    summary = favorites_sim.summary_payload(db)
    universe = favorites_sim.favorites_count(db)
    payload = dict(summary)
    payload.update(status)
    payload["favorites_universe_count"] = universe
    return payload


@router.post("/api/paper/favorites/start")
def favorites_sim_start(db=Depends(get_db)):
    favorites_sim.start(db)
    return favorites_sim_status(db)


@router.post("/api/paper/favorites/stop")
def favorites_sim_stop(db=Depends(get_db)):
    favorites_sim.stop(db)
    return favorites_sim_status(db)


@router.post("/api/paper/favorites/restart")
def favorites_sim_restart(db=Depends(get_db)):
    favorites_sim.restart(db)
    return favorites_sim_status(db)


@router.get("/api/paper/favorites/equity.json")
def favorites_sim_equity(range: str = Query("1m"), db=Depends(get_db)):
    range_key = (range or "").lower()
    allowed = {"1d", "1w", "1m", "3m", "all"}
    if range_key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid range")
    points = favorites_sim.get_equity_points(db, range_key)
    return {"range": range_key, "points": points}


@router.get("/api/paper/favorites/activity")
def favorites_sim_activity(db=Depends(get_db)):
    return favorites_sim.activity_payload(db)


@router.get("/api/paper/scalper/metrics.json")
def scalper_metrics(mode: str = Query("lf"), db=Depends(get_db)):
    mode_key = (mode or "").lower()
    if mode_key == "lf":
        metrics = scalper_lf.metrics_snapshot(db)
    elif mode_key == "hf":
        metrics = scalper_hf.metrics_snapshot(db)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")
    return {"mode": mode_key, "metrics": metrics}




def _sanitize_pagination(
    limit: int | float | str,
    offset: int | float | str,
    *,
    max_limit: int,
    default_limit: int | None = None,
    min_limit: int = 0,
) -> tuple[int, int]:
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = None

    if default_limit is not None and (limit_value is None or limit_value <= 0):
        limit_value = default_limit
    if limit_value is None:
        limit_value = 0

    if min_limit and limit_value < min_limit:
        limit_value = min_limit
    if max_limit > 0:
        limit_value = min(limit_value, max_limit)

    try:
        offset_value = int(offset)
    except (TypeError, ValueError):
        offset_value = 0
    if offset_value < 0:
        offset_value = 0
    return limit_value, offset_value


def _forward_runs_csv_rows(
    favorite: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "symbol",
        "direction",
        "entry_ts",
        "entry_px",
        "exit_ts",
        "exit_px",
        "outcome",
        "roi",
        "tt_bars",
        "dd",
        "rule_hash",
    ]
    ticker = (favorite.get("ticker") or "").upper()
    direction = canonical_direction(favorite.get("direction")) or "UP"

    def normalize(value: Any) -> Any:
        return "" if value is None else value

    records: list[list[Any]] = []
    for row in rows:
        records.append(
            [
                ticker,
                (direction or "").upper(),
                normalize(row.get("entry_ts")),
                normalize(row.get("entry_px")),
                normalize(row.get("exit_ts")),
                normalize(row.get("exit_px")),
                normalize(row.get("outcome")),
                normalize(row.get("roi")),
                normalize(row.get("tt_bars")),
                normalize(row.get("dd")),
                normalize(row.get("rule_hash")),
            ]
        )
    return headers, records


@router.get("/forward/export.csv")
def forward_runs_export_csv(
    favorite_id: int = Query(...),
    limit: int = Query(1000, ge=0),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    favorite = _fetch_favorite_metadata(db, favorite_id)
    if not favorite:
        return JSONResponse({"error": "Favorite not found"}, status_code=404)
    limit_value, offset_value = _sanitize_pagination(
        limit, offset, max_limit=5000, default_limit=1000, min_limit=0
    )
    rows = []
    if limit_value > 0:
        rows = get_forward_history_for_cursor(
            db, favorite_id, limit_value, offset_value
        )
    headers, records = _forward_runs_csv_rows(favorite, rows)

    def row_iter() -> Any:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(headers)
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        for record in records:
            writer.writerow(record)
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    filename = f"forward_runs_{(favorite.get('ticker') or favorite_id)}.csv"
    response = StreamingResponse(row_iter(), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@router.get("/api/forward/favorites")
def api_forward_favorites(db=Depends(get_db)):
    favorites = [
        _serialize_forward_favorite(fav, db) for fav in _load_forward_favorites(db)
    ]
    return {"favorites": favorites}


@router.get("/api/forward/{favorite_id}")
def api_forward_runs_detail(
    request: Request,
    favorite_id: int,
    limit: int = Query(50, ge=0),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    favorite = _fetch_favorite_metadata(db, favorite_id)
    if not favorite or not _favorite_visible_to_request(favorite, request):
        return JSONResponse({"error": "Favorite not found"}, status_code=404)

    limit_value, offset_value = _sanitize_pagination(
        limit, offset, max_limit=200, default_limit=50, min_limit=1
    )

    fav_key = str(favorite.get("id") or favorite_id)
    db.execute(
        "SELECT MAX(entry_ts) FROM forward_runs WHERE favorite_id=?",
        (fav_key,),
    )
    max_entry_row = db.fetchone()
    max_entry_ts = max_entry_row[0] if max_entry_row else None
    etag_value = f'W/"forward-runs:{fav_key}:{max_entry_ts or "none"}"'
    request_etag = request.headers.get("if-none-match") or ""
    etag_tokens = [token.strip() for token in request_etag.split(",") if token.strip()]
    if "*" in etag_tokens or etag_value in etag_tokens:
        return Response(status_code=304, headers={"ETag": etag_value})

    rows = get_forward_history_for_cursor(db, fav_key, limit_value, offset_value)
    current_hash = forward_rule_hash(favorite.get("rule"))
    payload = [
        {
            "entry_ts": row.get("entry_ts"),
            "entry_px": row.get("entry_px"),
            "exit_ts": row.get("exit_ts"),
            "exit_px": row.get("exit_px"),
            "outcome": row.get("outcome"),
            "roi": row.get("roi"),
            "tt_bars": row.get("tt_bars"),
            "dd": row.get("dd"),
            "rule_hash": row.get("rule_hash"),
            "rule_mismatch": bool(
                current_hash
                and row.get("rule_hash")
                and row.get("rule_hash") != current_hash
            ),
        }
        for row in rows
    ]
    return JSONResponse(payload, headers={"ETag": etag_value})


@router.post("/api/forward/run")
async def api_forward_run(payload: dict | None = Body(None), db=Depends(get_db)):
    payload = payload or {}
    raw_ids = payload.get("favorite_ids")
    ids: list[int] | None = None
    if raw_ids is not None:
        ids = []
        seen: set[int] = set()
        raw_iter = raw_ids if isinstance(raw_ids, list) else [raw_ids]
        for raw in raw_iter:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value in seen:
                continue
            seen.add(value)
            ids.append(value)
        if not ids:
            return JSONResponse(
                {"ok": False, "error": "No valid favorites selected"},
                status_code=400,
            )
    favorites = _load_forward_favorites(db, ids)
    if not favorites:
        return JSONResponse(
            {"ok": False, "error": "No favorites found"}, status_code=404
        )

    try:
        queued = _queue_forward_tests_for(db, favorites)
        total = len(favorites)
        if queued:
            message = f"Queued forward tests for {queued} favorite{'s' if queued != 1 else ''}"
        else:
            message = (
                f"Forward tests already running for {total} favorite{'s' if total != 1 else ''}"
            )
        return {
            "ok": True,
            "queued": queued,
            "count": total,
            "message": message,
        }
    except Exception:
        logger.exception("Failed to queue forward tests")
        return JSONResponse(
            {"ok": False, "error": "Server error starting forward tests"},
            status_code=500,
        )


@router.post("/favorites/delete/{fav_id}")
def favorites_delete(fav_id: int, db=Depends(get_db)):
    db.execute("DELETE FROM favorites WHERE id=?", (fav_id,))
    db.connection.commit()
    return RedirectResponse(url="/favorites", status_code=302)


@router.post("/favorites/delete-duplicates")
def favorites_delete_duplicates(db=Depends(get_db)):
    db.execute(
        """
        DELETE FROM favorites
        WHERE id NOT IN (
            SELECT MIN(id) FROM favorites GROUP BY ticker, direction, interval, rule
        )
        """
    )
    db.connection.commit()
    return RedirectResponse(url="/favorites", status_code=302)


def _favorite_record_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    ticker = str(payload.get("ticker") or "").strip().upper()
    rule = str(payload.get("rule") or "").strip()
    if not ticker or not rule:
        return {}, "missing ticker or rule"

    direction_raw = payload.get("direction")
    direction = canonical_direction(direction_raw) or "UP"

    interval_raw = payload.get("interval")
    interval = str(interval_raw or "").strip()
    if not interval:
        interval = "15m"

    target_pct = _coerce_float(payload.get("target_pct"))
    stop_pct = _coerce_float(payload.get("stop_pct"))
    window_value = _coerce_float(payload.get("window_value"))
    window_unit = str(payload.get("window_unit") or "").strip()

    roi_snapshot_raw = payload.get("roi_snapshot")
    if isinstance(roi_snapshot_raw, (dict, list)):
        try:
            roi_snapshot = json.dumps(roi_snapshot_raw)
        except (TypeError, ValueError):
            roi_snapshot = None
    else:
        roi_snapshot = _coerce_float(roi_snapshot_raw)
    hit_snapshot = _coerce_float(payload.get("hit_pct_snapshot"))
    dd_snapshot = _coerce_float(payload.get("dd_pct_snapshot"))

    support_raw = payload.get("support_snapshot")
    support = _coerce_int(support_raw)

    ref_dd = payload.get("ref_avg_dd")
    try:
        if ref_dd in (None, ""):
            ref_dd_value = None
        else:
            ref_dd_value = float(ref_dd)
            if ref_dd_value > 1:
                ref_dd_value /= 100.0
    except (TypeError, ValueError):
        ref_dd_value = None

    settings_candidates = (
        payload.get("settings_json_snapshot"),
        payload.get("settings"),
    )
    settings_dict: dict[str, Any] = {}
    settings_json: str | None = None
    for candidate in settings_candidates:
        if candidate in (None, ""):
            continue
        if isinstance(candidate, dict):
            settings_dict = dict(candidate)
            try:
                settings_json = json.dumps(settings_dict, default=str)
            except TypeError:
                settings_json = json.dumps(settings_dict, default=str)
            break
        if isinstance(candidate, (bytes, bytearray)):
            try:
                candidate = candidate.decode("utf-8", "ignore")
            except Exception:
                candidate = candidate.decode("latin-1", "ignore")
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                settings_json = text
                break
            if isinstance(parsed, dict):
                settings_dict = parsed
                settings_json = json.dumps(parsed, default=str)
            else:
                settings_json = text
            break
        else:
            try:
                settings_json = json.dumps(candidate, default=str)
            except TypeError:
                settings_json = json.dumps(str(candidate))
            break

    if settings_json is None:
        try:
            settings_json = json.dumps(settings_dict or {}, default=str)
        except TypeError:
            settings_json = json.dumps(str(settings_dict))

    params: dict[str, Any] = {}
    if settings_dict:
        try:
            params = coerce_scan_params(settings_dict) or {}
        except Exception:
            params = {}

    lookback = _coerce_float(payload.get("lookback_years"))
    min_support = _coerce_int(payload.get("min_support"))
    if params:
        if lookback is None:
            lookback = _coerce_float(params.get("lookback_years"))
        if min_support is None:
            min_support = _coerce_int(params.get("min_support"))
        if target_pct is None:
            target_pct = _coerce_float(params.get("target_pct"))
        if stop_pct is None:
            stop_pct = _coerce_float(params.get("stop_pct"))
        if window_value is None:
            window_value = _coerce_float(params.get("window_value"))
        if not window_unit:
            unit = params.get("window_unit")
            if isinstance(unit, str):
                window_unit = unit.strip()
        if not interval_raw and isinstance(params.get("interval"), str):
            interval = params.get("interval") or interval
        if direction_raw in (None, ""):
            snapshot_dir = canonical_direction(params.get("direction"))
            if snapshot_dir:
                direction = snapshot_dir

    if target_pct is None:
        target_pct = 1.0
    if stop_pct is None:
        stop_pct = 0.5
    if window_value is None:
        window_value = 4.0
    if not window_unit:
        window_unit = "Hours"

    support_payload: dict[str, Any] = {}
    if support is not None:
        support_payload["count"] = support
    if min_support is not None:
        support_payload["min_support"] = min_support
    if lookback is not None:
        support_payload["lookback_years"] = lookback
    support_snapshot = json.dumps(support_payload) if support_payload else None

    record = {
        "ticker": ticker,
        "direction": direction,
        "interval": interval,
        "rule": rule,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "window_value": window_value,
        "window_unit": window_unit,
        "ref_avg_dd": ref_dd_value,
        "lookback_years": lookback,
        "min_support": min_support,
        "support_snapshot": support_snapshot,
        "roi_snapshot": roi_snapshot,
        "hit_pct_snapshot": hit_snapshot,
        "dd_pct_snapshot": dd_snapshot,
        "rule_snapshot": payload.get("rule_snapshot") or rule,
        "settings_json_snapshot": settings_json,
    }

    return record, None


def _favorite_insert_columns(db: sqlite3.Cursor) -> tuple[str, ...]:
    conn = getattr(db, "connection", None)
    if conn is None:
        return _FAVORITE_INSERT_COLUMNS
    if not isinstance(conn, sqlite3.Connection):
        return _FAVORITE_INSERT_COLUMNS

    cache_key = id(conn)
    cached = _FAVORITE_COLUMN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    available: set[str] = set()
    try:
        db.execute("PRAGMA table_info(favorites)")
        available = {row[1] for row in db.fetchall()}
    except sqlite3.Error:
        columns = tuple(
            column
            for column in _FAVORITE_INSERT_COLUMNS
            if column not in _FAVORITE_OPTIONAL_COLUMNS
        )
        _FAVORITE_COLUMN_CACHE[cache_key] = columns
        return columns

    columns = tuple(
        column
        for column in _FAVORITE_INSERT_COLUMNS
        if column not in _FAVORITE_OPTIONAL_COLUMNS or column in available
    )
    _FAVORITE_COLUMN_CACHE[cache_key] = columns
    return columns


def _insert_favorite_record(db: sqlite3.Cursor, record: Mapping[str, Any]) -> None:
    columns = _favorite_insert_columns(db)
    columns_sql = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    values = tuple(record.get(column) for column in columns)
    db.execute(
        f"INSERT INTO favorites({columns_sql}) VALUES ({placeholders})",
        values,
    )


def _render_toggle_error(message: str) -> str:
    safe = html.escape(message or "Unable to update favorite", quote=False)
    return f"<div class='toast error'>{safe}</div>"


@router.post("/favorites/add")
async def favorites_add(request: Request, db=Depends(get_db)):
    try:
        payload = await request.json()
    except Exception:
        payload = None
    if not isinstance(payload, Mapping):
        return JSONResponse({"ok": False, "error": "invalid payload"}, status_code=400)

    record, error = _favorite_record_from_payload(payload)
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=400)

    record["snapshot_at"] = now_et().isoformat()
    try:
        _insert_favorite_record(db, record)
        fav_id = db.lastrowid
        db.connection.commit()
    except sqlite3.OperationalError:
        logger.exception("favorite_insert_failed_schema")
        return JSONResponse(
            {"ok": False, "error": "Favorites schema out of date"},
            status_code=400,
        )
    except Exception:
        logger.exception("favorite_insert_failed")
        return JSONResponse({"ok": False, "error": "Failed to add favorite"}, status_code=500)

    log_telemetry(
        {
            "type": "favorite_saved",
            "symbol": record.get("ticker"),
            "lookback_years": record.get("lookback_years"),
        }
    )

    response: dict[str, Any] = {"ok": True}
    if fav_id is not None:
        response["favorite_id"] = fav_id
    return response


@router.post("/favorites/toggle")
async def favorites_toggle(request: Request, db=Depends(get_db)):
    content_type = request.headers.get("content-type", "")
    payload: dict[str, Any] = {}
    if "application/json" in content_type:
        try:
            parsed = await request.json()
        except Exception:
            parsed = None
        if isinstance(parsed, Mapping):
            payload = dict(parsed)
    else:
        try:
            form = await request.form()
            payload = {k: v for k, v in form.multi_items()}
        except Exception:
            payload = {}

    record, error = _favorite_record_from_payload(payload)
    if error:
        return HTMLResponse(_render_toggle_error(error), status_code=400)

    ticker = record["ticker"]
    direction = record["direction"]
    interval = record["interval"]
    rule = record["rule"]

    try:
        db.execute(
            """
            SELECT 1 FROM favorites
            WHERE ticker=? AND direction=? AND interval=? AND rule=?
            LIMIT 1
            """,
            (ticker, direction, interval, rule),
        )
        existing = db.fetchone()
    except Exception:
        logger.exception("favorite_toggle_lookup_failed")
        return HTMLResponse(
            _render_toggle_error("Failed to toggle favorite"), status_code=400
        )

    if existing:
        try:
            db.execute(
                """
                DELETE FROM favorites
                WHERE ticker=? AND direction=? AND interval=? AND rule=?
                """,
                (ticker, direction, interval, rule),
            )
            db.connection.commit()
        except Exception:
            logger.exception("favorite_toggle_delete_failed")
            return HTMLResponse(
                _render_toggle_error("Failed to remove favorite"), status_code=400
            )
        return HTMLResponse("", status_code=204)

    record["snapshot_at"] = now_et().isoformat()
    try:
        _insert_favorite_record(db, record)
        fav_id = db.lastrowid
        db.connection.commit()
    except Exception:
        logger.exception("favorite_toggle_insert_failed")
        return HTMLResponse(
            _render_toggle_error("Failed to add favorite"), status_code=400
        )

    fragment = templates.get_template("partials/fav_star.html").render(
        {
            "ticker": ticker,
            "direction": direction,
            "interval": interval,
            "rule": rule,
            "favorite_id": fav_id,
            "is_favorite": True,
        }
    )
    return HTMLResponse(fragment, status_code=201)


@router.post("/favorites/test_alert")
@router.post("/settings/test-alert")
async def favorites_test_alert(
    request: Request,
    db=Depends(get_db),
):
    content_type = request.headers.get("content-type", "")
    form_data: dict[str, Any] = {}
    payload: dict[str, Any] = {}

    if "application/json" in content_type:
        try:
            parsed = await request.json()
        except Exception:
            parsed = None
        if isinstance(parsed, Mapping):
            payload = dict(parsed)
    else:
        try:
            form = await request.form()
            form_data = {k: v for k, v in form.multi_items()}
        except Exception:
            form_data = {}
    symbol_raw = (
        form_data.get("symbol")
        or payload.get("symbol")
        or payload.get("ticker")
        or "AAPL"
    )
    symbol = str(symbol_raw or "AAPL").upper()

    configured_channel = getattr(
        settings, "alert_channel", getattr(settings, "ALERT_CHANNEL", "Email")
    )
    raw_channel = (
        form_data.get("channel")
        or payload.get("channel")
        or configured_channel
        or "email"
    )
    channel = str(raw_channel or "email").strip().lower() or "email"
    if channel not in {"email", "mms", "sms"}:
        channel = "email"
    channel_label_map = {"email": "Email", "mms": "MMS", "sms": "SMS"}
    channel_label = channel_label_map.get(channel, "Email")

    outcomes_config = getattr(
        settings, "alert_outcomes", getattr(settings, "ALERT_OUTCOMES", "hit")
    )
    raw_outcomes = (
        form_data.get("outcomes")
        or payload.get("outcomes")
        or outcomes_config
        or "hit"
    )
    outcomes = str(raw_outcomes or "hit").strip().lower() or "hit"

    logger.info(
        "favorites_test_alert_request channel=%s outcomes=%s", channel, outcomes
    )

    ok, message = favorites_alerts.enrich_and_send_test(
        symbol,
        "UP",
        channel=channel,
        compact=False,
        outcomes=outcomes,
    )
    subject = message.get("subject", "") if isinstance(message, dict) else ""
    body = message.get("body", "") if isinstance(message, dict) else ""

    response = {
        "ok": ok,
        "symbol": symbol,
        "channel": channel_label,
        "outcomes": outcomes,
        "subject": subject,
        "body": body,
    }

    base_telem = {
        "type": "favorites_test_alert",
        "symbol": symbol,
        "channel": channel,
        "outcomes": outcomes,
        "ok": ok,
    }

    if not ok:
        response["error"] = "Unable to generate alert body"
        log_telemetry(base_telem)
        return JSONResponse(response, status_code=500)

    delivery_context = {"symbol": symbol, "fav_id": "test", "outcomes": outcomes}

    def send_email_via_smtp(body_text: str, *, fallback: bool = False):
        st = get_settings(db)
        cfg, missing, _ = _smtp_config_status(st)
        if not cfg["recipients"]:
            missing = list(dict.fromkeys(missing + ["recipients"]))
        if missing:
            error_msg = f"SMTP not configured: {_format_missing_fields(sorted(set(missing)))}"
            return False, error_msg, None, 400

        host = cfg.get("host")
        port = cfg.get("port")
        user = cfg.get("user", "")
        password = cfg.get("password", "")
        mail_from = cfg.get("mail_from") or user
        if not host or not port or not mail_from:
            return False, "SMTP not configured: host/port/mail_from missing", None, 400

        context = {**delivery_context, "channel": "email"}
        if fallback:
            context["fallback_from"] = channel

        try:
            send_result = send_email_smtp(
                str(host),
                int(port),
                str(user or ""),
                str(password or ""),
                str(mail_from),
                cfg["recipients"],
                subject or f"Favorites Alert Test: {symbol}",
                body_text,
                context=context,
            )
        except Exception:
            logger.exception("favorites test alert email send raised")
            return False, "SMTP send failed", None, 502

        if not send_result.get("ok"):
            error_msg = send_result.get("error", "SMTP send failed")
            return False, error_msg, None, 502

        message_id = send_result.get("message_id", "")
        return True, None, message_id, 200

    if channel == "email":
        success, error_msg, message_id, status_code = send_email_via_smtp(body)
        log_telemetry(
            {
                "type": "favorites_test_alert_send",
                "channel": "email",
                "provider": "smtp",
                "ok": success,
                "error": error_msg if not success else None,
                "message_id": message_id if success else None,
                "outcomes": outcomes,
            }
        )
        base_telem["ok"] = success
        log_telemetry(base_telem)
        if not success:
            response.update({"ok": False, "error": error_msg or "SMTP send failed"})
            return JSONResponse(response, status_code=status_code)
        response.update({"ok": True, "message_id": message_id})
        return response

    twilio_enabled = twilio_client.is_enabled()
    destinations = sms_consent.active_destinations() if twilio_enabled else []
    success = False
    fallback_needed = not twilio_enabled
    fallback_reason = "Twilio not configured" if not twilio_enabled else ""
    if twilio_enabled and destinations:
        message_body = sms_consent.append_footer(body)
        for dest in destinations:
            number = sms_consent.normalize_phone(str(dest.get("phone_e164") or ""))
            if not number:
                continue
            allowed, consent_row = sms_consent.allow_sending(number)
            if not allowed:
                continue
            context = {
                **delivery_context,
                "channel": channel,
                "to": number,
                "user_id": (consent_row or {}).get("user_id"),
            }
            try:
                send_ok = twilio_client.send_mms(number, message_body, context=context)
            except Exception:
                logger.exception("favorites test alert mms send raised")
                send_ok = False
            if send_ok:
                sms_consent.record_delivery(
                    number,
                    (consent_row or {}).get("user_id"),
                    message_body,
                    message_type="test",
                )
                success = True
        if not success:
            fallback_needed = True
            fallback_reason = f"No {channel.upper()} deliveries succeeded"
    elif twilio_enabled and not destinations:
        fallback_needed = False
        fallback_reason = "No SMS recipients opted-in"

    log_telemetry(
        {
            "type": "favorites_test_alert_send",
            "channel": channel,
            "provider": "twilio",
            "ok": success,
            "error": None if success else fallback_reason,
            "outcomes": outcomes,
        }
    )

    if success:
        base_telem["ok"] = True
        log_telemetry(base_telem)
        return response

    if fallback_needed:
        if not twilio_enabled:
            fallback_body = f"{body}\n\n[Sent via Email — SMS unavailable]"
        else:
            fallback_label = "MMS" if channel != "sms" else "SMS"
            fallback_body = f"{body}\n\n[Sent via Email — {fallback_label} unavailable]"
        response["body"] = fallback_body
        response["channel"] = "Email"
        channel_label = "Email"
        email_success, email_error, message_id, status_code = send_email_via_smtp(
            fallback_body, fallback=True
        )
        log_telemetry(
            {
                "type": "favorites_test_alert_send",
                "channel": "email",
                "provider": "smtp",
                "ok": email_success,
                "error": email_error if not email_success else None,
                "message_id": message_id if email_success else None,
                "outcomes": outcomes,
            }
        )
        base_telem.update({"ok": email_success, "channel": "email"})
        log_telemetry(base_telem)
        if email_success:
            response.update({"ok": True, "message_id": message_id})
            return response
        response.update({"ok": False, "error": email_error or fallback_reason})
        return JSONResponse(response, status_code=status_code)

    base_telem["ok"] = False
    log_telemetry(base_telem)
    error_message = fallback_reason or "Unable to send MMS"
    response.update({"ok": False, "error": error_message})
    return JSONResponse(response, status_code=400)


@router.post("/favorites/test_alert/preview")
def favorites_test_alert_preview(payload: dict = Body(...)):
    symbol = (payload.get("symbol") or payload.get("ticker") or "AAPL").upper()
    channel = (payload.get("channel") or "mms").lower()
    compact = bool(payload.get("compact"))
    outcomes = (payload.get("outcomes") or "hit").strip().lower() or "hit"
    subject, body = favorites_alerts.build_preview(
        symbol,
        channel=channel,
        outcomes=outcomes,
        compact=compact,
    )
    return {
        "ok": bool(body),
        "symbol": symbol,
        "channel": channel,
        "compact": compact,
        "outcomes": outcomes,
        "subject": subject,
        "body": body,
    }


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db=Depends(get_db)):
    user_id = _request_user_id(request)
    st = dict(get_settings(db))
    st.setdefault(
        "alert_outcomes",
        getattr(settings, "alert_outcomes", getattr(settings, "ALERT_OUTCOMES", "hit")),
    )
    st.setdefault(
        "forward_recency_mode",
        getattr(
            settings,
            "forward_recency_mode",
            getattr(settings, "FORWARD_RECENCY_MODE", "off"),
        ),
    )
    st.setdefault(
        "forward_recency_halflife_days",
        getattr(
            settings,
            "forward_recency_halflife_days",
            getattr(settings, "FORWARD_RECENCY_HALFLIFE_DAYS", 30.0),
        ),
    )
    _, smtp_missing, smtp_has_any = _smtp_config_status(st)
    smtp_configured = not smtp_missing
    smtp_warning = smtp_has_any and bool(smtp_missing)
    sms_active = sms_consent.latest_for_user(user_id, db_cursor=db)
    sms_history = sms_consent.history_for_user(user_id, limit=10, db_cursor=db)
    twilio_enabled = twilio_client.is_enabled()
    twilio_verify_enabled = twilio_client.is_verify_enabled()
    sms_state = {
        "active": bool(sms_active),
        "phone": (sms_active or {}).get("phone_e164"),
        "consent_at": (sms_active or {}).get("consent_at"),
        "method": (sms_active or {}).get("method"),
        "user_id": user_id,
        "history": sms_history,
        "twilio_enabled": twilio_enabled,
        "twilio_verify_enabled": twilio_verify_enabled,
    }
    paper_settings_state = paper_trading.load_settings(db)
    paper_summary = paper_trading.get_summary(db)
    scalper_settings = scalper_lf.load_settings(db)
    scalper_status = scalper_lf.status_payload(db)
    scalper_hf_settings = scalper_hf.load_settings(db)
    scalper_hf_status = scalper_hf.status_payload(db)
    fav_sim_settings = favorites_sim.load_settings(db)
    fav_sim_status = favorites_sim.status_payload(db)
    fav_sim_summary = favorites_sim.summary_payload(db)
    fav_universe_count = favorites_sim.favorites_count(db)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "st": st,
            "active_tab": "settings",
            "smtp_configured": smtp_configured,
            "smtp_missing": smtp_missing,
            "smtp_missing_label": _format_missing_fields(smtp_missing),
            "smtp_warning": smtp_warning,
            "twilio_configured": twilio_enabled,
            "alert_channel": getattr(settings, "alert_channel", "Email"),
            "alert_outcomes": st.get("alert_outcomes", "hit"),
            "sms_consent_text": _SMS_CONSENT_TEXT,
            "sms_state": sms_state,
            "twilio_enabled": twilio_enabled,
            "twilio_verify_enabled": twilio_verify_enabled,
            "paper_settings": paper_settings_state,
            "paper_summary": paper_summary,
            "scalper_settings": scalper_settings,
            "scalper_status": scalper_status,
            "scalper_hf_settings": scalper_hf_settings,
            "scalper_hf_status": scalper_hf_status,
            "favorites_sim_settings": fav_sim_settings,
            "favorites_sim_status": fav_sim_status,
            "favorites_sim_summary": fav_sim_summary,
            "favorites_universe_count": fav_universe_count,
            "canonical_path": "/settings",
        },
    )


@router.get("/info", response_class=HTMLResponse)
def info_page(request: Request):
    return templates.TemplateResponse(
        request,
        "info.html",
        {"active_tab": "info", "canonical_path": "/info"},
    )


@router.get("/sms-consent", response_class=HTMLResponse)
def sms_consent_page(request: Request):
    return templates.TemplateResponse(
        request,
        "sms_consent.html",
        {"canonical_path": "/sms-consent"},
    )


@router.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return templates.TemplateResponse(
        request,
        "privacy.html",
        {"canonical_path": "/privacy"},
    )


@router.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return templates.TemplateResponse(
        request,
        "terms.html",
        {"canonical_path": "/terms"},
    )


@router.get("/robots.txt")
def robots_txt() -> FileResponse:
    return FileResponse("static/robots.txt", media_type="text/plain")


@router.get("/sitemap.xml")
def sitemap_xml() -> FileResponse:
    return FileResponse("static/sitemap.xml", media_type="application/xml")


@router.post("/settings/save")
def settings_save(
    request: Request,
    smtp_host: str = Form(""),
    smtp_port: int = Form(587),
    smtp_user: str = Form(""),
    smtp_pass: str = Form(""),
    mail_from: str = Form(""),
    recipients: str = Form(""),
    scanner_recipients: str = Form(""),
    scheduler_enabled: int = Form(1),
    throttle_minutes: int = Form(60),
    alert_outcomes: str = Form("hit"),
    forward_recency_mode: str = Form("off"),
    forward_recency_halflife_days: str = Form("30"),
    paper_starting_balance: str = Form("10000"),
    paper_max_pct: str = Form("10"),
    scalper_starting_balance: str = Form("100000"),
    scalper_pct_per_trade: str = Form("3"),
    scalper_daily_cap: str = Form("20"),
    scalper_tickers: str = Form("SPY,QQQ,TSLA,NVDA"),
    scalper_target_pct: str = Form("6"),
    scalper_stop_pct: str = Form("-3"),
    scalper_time_cap: str = Form("15"),
    scalper_session_start: str = Form("09:30"),
    scalper_session_end: str = Form("16:00"),
    scalper_allow_premarket: int = Form(0),
    scalper_allow_postmarket: int = Form(0),
    scalper_per_contract_fee: str = Form("0.65"),
    scalper_per_order_fee: str = Form("0"),
    scalper_rsi_filter: int = Form(0),
    scalper_hf_starting_balance: str = Form("100000"),
    scalper_hf_pct_per_trade: str = Form("1"),
    scalper_hf_daily_cap: str = Form("50"),
    scalper_hf_tickers: str = Form("SPY,QQQ,NVDA,TSLA,META,AMD"),
    scalper_hf_target_pct: str = Form("4"),
    scalper_hf_stop_pct: str = Form("-2"),
    scalper_hf_time_cap: str = Form("5"),
    scalper_hf_cooldown: str = Form("2"),
    scalper_hf_max_positions: str = Form("2"),
    scalper_hf_drawdown: str = Form("-6"),
    scalper_hf_volatility_gate: str = Form("3"),
    scalper_hf_per_contract_fee: str = Form("0.65"),
    scalper_hf_per_order_fee: str = Form("0"),
    favorites_starting_balance: str = Form("100000"),
    favorites_allocation_mode: str = Form("percent"),
    favorites_allocation_value: str = Form("10"),
    favorites_per_contract_fee: str = Form("0"),
    favorites_per_order_fee: str = Form("0"),
    favorites_slippage_bps: str = Form("0"),
    favorites_daily_cap: str = Form("10"),
    favorites_allow_premarket: int = Form(0),
    favorites_allow_postmarket: int = Form(0),
    favorites_entry_rule: str = Form("next_open"),
    favorites_exit_time_cap: str = Form("15"),
    favorites_exit_profit_target: str = Form(""),
    favorites_exit_max_adverse: str = Form(""),
    db=Depends(get_db),
):
    _ensure_scanner_column(db)
    from email.utils import parseaddr

    from services.notifications import is_carrier_address

    def _clean(raw: str, *, allow_sms: bool) -> str:
        parts = [r.strip() for r in raw.split(",") if r.strip()]
        cleaned: list[str] = []
        for r in parts:
            addr = parseaddr(r)[1]
            if "@" not in addr:
                continue
            if not allow_sms and is_carrier_address(addr):
                continue
            cleaned.append(addr)
        return ",".join(cleaned)

    clean_fav = _clean(recipients, allow_sms=True)
    clean_scan = _clean(scanner_recipients, allow_sms=False)

    try:
        port_value = int(smtp_port)
    except (TypeError, ValueError):
        port_value = 0

    password_value = smtp_pass.replace(" ", "").strip()
    outcomes_choice = (alert_outcomes or "hit").strip().lower()
    if outcomes_choice not in {"hit", "all"}:
        outcomes_choice = "hit"

    mode_choice = (forward_recency_mode or "off").strip().lower()
    if mode_choice not in {"off", "exp"}:
        mode_choice = "off"
    try:
        half_life_value = float((forward_recency_halflife_days or "").strip() or 30.0)
    except (TypeError, ValueError):
        half_life_value = getattr(settings, "forward_recency_halflife_days", 30.0)
    if half_life_value <= 0:
        half_life_value = getattr(settings, "forward_recency_halflife_days", 30.0) or 30.0

    current_paper_settings = paper_trading.load_settings(db)
    current_scalper_settings = scalper_lf.load_settings(db)
    current_scalper_hf_settings = scalper_hf.load_settings(db)
    current_favorites_settings = favorites_sim.load_settings(db)
    try:
        start_balance_value = float(
            (paper_starting_balance or "").replace(",", "").strip() or current_paper_settings.starting_balance
        )
    except (TypeError, ValueError):
        start_balance_value = current_paper_settings.starting_balance
    try:
        max_pct_value = float((paper_max_pct or "").strip() or current_paper_settings.max_pct)
    except (TypeError, ValueError):
        max_pct_value = current_paper_settings.max_pct
    if max_pct_value <= 0:
        max_pct_value = current_paper_settings.max_pct

    db.execute(
        """
        UPDATE settings
           SET smtp_host=?, smtp_port=?, smtp_user=?, smtp_pass=?, mail_from=?, recipients=?, scanner_recipients=?, alert_outcomes=?, forward_recency_mode=?, forward_recency_halflife_days=?, scheduler_enabled=?, throttle_minutes=?
         WHERE id=1
        """,
        (
            smtp_host.strip(),
            port_value,
            smtp_user.strip(),
            password_value,
            mail_from.strip(),
            clean_fav,
            clean_scan,
            outcomes_choice,
            mode_choice,
            half_life_value,
            int(scheduler_enabled),
            int(throttle_minutes),
        ),
    )
    db.connection.commit()
    os.environ["ALERT_OUTCOMES"] = outcomes_choice
    settings.alert_outcomes = outcomes_choice
    setattr(settings, "ALERT_OUTCOMES", outcomes_choice)
    os.environ["FORWARD_RECENCY_MODE"] = mode_choice
    os.environ["FORWARD_RECENCY_HALFLIFE_DAYS"] = str(half_life_value)
    settings.forward_recency_mode = mode_choice
    setattr(settings, "FORWARD_RECENCY_MODE", mode_choice)
    settings.forward_recency_halflife_days = half_life_value
    setattr(settings, "FORWARD_RECENCY_HALFLIFE_DAYS", half_life_value)
    paper_trading.update_settings(
        db,
        starting_balance=start_balance_value,
        max_pct=max_pct_value,
    )

    def _float(value: str, default: float) -> float:
        try:
            return float((value or "").replace(",", "").strip() or default)
        except (TypeError, ValueError):
            return default

    def _int(value: str, default: int) -> int:
        try:
            return int((value or "").strip() or default)
        except (TypeError, ValueError):
            return default

    def _optional_float(value: str, default: float | None) -> float | None:
        text = (value or "").replace(",", "").strip()
        if text == "":
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    scalper_balance = _float(scalper_starting_balance, current_scalper_settings.starting_balance)
    scalper_pct_value = _float(scalper_pct_per_trade, current_scalper_settings.pct_per_trade)
    scalper_cap_value = _int(scalper_daily_cap, current_scalper_settings.daily_trade_cap)
    scalper_tickers_value = scalper_tickers or current_scalper_settings.tickers
    scalper_target_value = _float(scalper_target_pct, current_scalper_settings.profit_target_pct)
    scalper_stop_value = _float(scalper_stop_pct, current_scalper_settings.max_adverse_pct)
    scalper_time_cap_value = _int(scalper_time_cap, current_scalper_settings.time_cap_minutes)
    session_start_value = (scalper_session_start or current_scalper_settings.session_start).strip()
    session_end_value = (scalper_session_end or current_scalper_settings.session_end).strip()
    per_contract_value = _float(scalper_per_contract_fee, current_scalper_settings.per_contract_fee)
    per_order_value = _float(scalper_per_order_fee, current_scalper_settings.per_order_fee)
    scalper_hf_balance = _float(scalper_hf_starting_balance, current_scalper_hf_settings.starting_balance)
    scalper_hf_pct_value = _float(scalper_hf_pct_per_trade, current_scalper_hf_settings.pct_per_trade)
    scalper_hf_cap_value = _int(scalper_hf_daily_cap, current_scalper_hf_settings.daily_trade_cap)
    scalper_hf_tickers_value = scalper_hf_tickers or current_scalper_hf_settings.tickers
    scalper_hf_target_value = _float(scalper_hf_target_pct, current_scalper_hf_settings.profit_target_pct)
    scalper_hf_stop_value = _float(scalper_hf_stop_pct, current_scalper_hf_settings.max_adverse_pct)
    scalper_hf_time_cap_value = _int(scalper_hf_time_cap, current_scalper_hf_settings.time_cap_minutes)
    scalper_hf_cooldown_value = _int(scalper_hf_cooldown, current_scalper_hf_settings.cooldown_minutes)
    scalper_hf_positions_value = _int(scalper_hf_max_positions, current_scalper_hf_settings.max_open_positions)
    scalper_hf_drawdown_value = _float(scalper_hf_drawdown, current_scalper_hf_settings.daily_max_drawdown_pct)
    scalper_hf_vol_gate = _float(scalper_hf_volatility_gate, current_scalper_hf_settings.volatility_gate)
    scalper_hf_per_contract_value = _float(scalper_hf_per_contract_fee, current_scalper_hf_settings.per_contract_fee)
    scalper_hf_per_order_value = _float(scalper_hf_per_order_fee, current_scalper_hf_settings.per_order_fee)
    fav_balance_value = _float(
        favorites_starting_balance,
        current_favorites_settings.starting_balance,
    )
    fav_allocation_mode_value = (
        (favorites_allocation_mode or current_favorites_settings.allocation_mode)
        .strip()
        .lower()
    )
    if fav_allocation_mode_value not in {"percent", "fixed"}:
        fav_allocation_mode_value = current_favorites_settings.allocation_mode
    fav_allocation_value = _float(
        favorites_allocation_value,
        current_favorites_settings.allocation_value,
    )
    fav_per_contract_value = _float(
        favorites_per_contract_fee,
        current_favorites_settings.per_contract_fee,
    )
    fav_per_order_value = _float(
        favorites_per_order_fee,
        current_favorites_settings.per_order_fee,
    )
    fav_slippage_value = _float(
        favorites_slippage_bps,
        current_favorites_settings.slippage_bps,
    )
    fav_daily_cap_value = _int(
        favorites_daily_cap,
        current_favorites_settings.daily_trade_cap,
    )
    fav_entry_rule_value = (
        (favorites_entry_rule or current_favorites_settings.entry_rule)
        .strip()
        .lower()
    )
    if fav_entry_rule_value not in {"next_open", "signal_close"}:
        fav_entry_rule_value = current_favorites_settings.entry_rule
    fav_exit_time_cap_value = _int(
        favorites_exit_time_cap,
        current_favorites_settings.exit_time_cap_minutes,
    )
    fav_profit_target_value = _optional_float(
        favorites_exit_profit_target,
        current_favorites_settings.exit_profit_target_pct,
    )
    fav_max_adverse_value = _optional_float(
        favorites_exit_max_adverse,
        current_favorites_settings.exit_max_adverse_pct,
    )
    favorites_sim.update_settings(
        db,
        starting_balance=fav_balance_value,
        allocation_mode=fav_allocation_mode_value,
        allocation_value=fav_allocation_value,
        per_contract_fee=fav_per_contract_value,
        per_order_fee=fav_per_order_value,
        slippage_bps=fav_slippage_value,
        daily_trade_cap=fav_daily_cap_value,
        allow_premarket=bool(favorites_allow_premarket),
        allow_postmarket=bool(favorites_allow_postmarket),
        entry_rule=fav_entry_rule_value,
        exit_time_cap_minutes=fav_exit_time_cap_value,
        exit_profit_target_pct=fav_profit_target_value,
        exit_max_adverse_pct=fav_max_adverse_value,
    )
    scalper_lf.update_settings(
        db,
        starting_balance=scalper_balance,
        pct_per_trade=scalper_pct_value,
        daily_trade_cap=scalper_cap_value,
        tickers=scalper_tickers_value,
        profit_target_pct=scalper_target_value,
        max_adverse_pct=scalper_stop_value,
        time_cap_minutes=scalper_time_cap_value,
        session_start=session_start_value,
        session_end=session_end_value,
        allow_premarket=bool(scalper_allow_premarket),
        allow_postmarket=bool(scalper_allow_postmarket),
        per_contract_fee=per_contract_value,
        per_order_fee=per_order_value,
        rsi_filter=bool(scalper_rsi_filter),
    )
    scalper_hf.update_settings(
        db,
        starting_balance=scalper_hf_balance,
        pct_per_trade=scalper_hf_pct_value,
        daily_trade_cap=scalper_hf_cap_value,
        tickers=scalper_hf_tickers_value,
        profit_target_pct=scalper_hf_target_value,
        max_adverse_pct=scalper_hf_stop_value,
        time_cap_minutes=scalper_hf_time_cap_value,
        cooldown_minutes=scalper_hf_cooldown_value,
        max_open_positions=scalper_hf_positions_value,
        daily_max_drawdown_pct=scalper_hf_drawdown_value,
        per_contract_fee=scalper_hf_per_contract_value,
        per_order_fee=scalper_hf_per_order_value,
        volatility_gate=scalper_hf_vol_gate,
    )
    return RedirectResponse(url="/settings", status_code=302)


@router.post("/scanner/run")
async def scanner_run(request: Request):
    form = await request.form()
    params = coerce_scan_params(form)

    scan_type = params.get("scan_type", "scan150")
    single_ticker = (form.get("ticker") or "").strip().upper()
    if scan_type.lower() in ("single", "single_ticker") and single_ticker:
        tickers = [single_ticker]
    elif scan_type.lower() in ("sp100", "sp_100", "sp100_scan"):
        tickers = SP100
    elif scan_type.lower() in ("top250", "scan250", "options250"):
        tickers = TOP250
    else:
        tickers = TOP150

    sort_key = (form.get("sort") or "").strip().lower()
    if sort_key not in ("ticker", "roi", "hit"):
        sort_key = ""

    task_id = uuid4().hex
    _task_create(task_id, len(tickers))

    loop = asyncio.get_running_loop()

    async def _run() -> None:
        await _run_scan_task(task_id, tickers, params, sort_key, scan_type)

    task = loop.create_task(_run(), name=f"scan-{task_id}")
    _SCAN_TASKS[task_id] = task
    task.add_done_callback(lambda _: _SCAN_TASKS.pop(task_id, None))

    return JSONResponse({"task_id": task_id})


@router.get("/scanner/progress/{task_id}")
async def scanner_progress(task_id: str):
    task = _task_get(task_id)
    if not task:
        return JSONResponse(
            {"done": 0, "total": 0, "percent": 0.0, "state": "failed"}, status_code=404
        )
    data = {
        "done": task.get("done", 0),
        "total": task.get("total", 0),
        "percent": task.get("percent", 0.0),
        "state": task.get("state", "running"),
        "message": task.get("message", ""),
    }
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@router.get("/scanner/status/{task_id}")
async def scanner_status(task_id: str):
    _task_gc()
    task = _task_get(task_id)
    if not task:
        return JSONResponse({}, status_code=404)
    data = {
        "id": task_id,
        "total": task.get("total"),
        "completed": task.get("done"),
        "percent": task.get("percent"),
        "state": task.get("state"),
        "message": task.get("message"),
        "started_at": task.get("started_at"),
        "updated_at": task.get("updated_at"),
        "ctx": task.get("ctx"),
    }
    percent_value = data.get("percent")
    state_value = data.get("state")
    if isinstance(percent_value, (int, float)):
        if state_value not in {"succeeded", "failed"}:
            data["percent"] = min(float(percent_value), 99.9)
        elif not task.get("first_status_seen"):
            data["percent"] = min(float(percent_value), 99.9)
    task["first_status_seen"] = True
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@router.get("/scanner/results/{task_id}", response_class=HTMLResponse)
async def scanner_results(request: Request, task_id: str):
    task = _task_get(task_id)
    if not task or task.get("state") != "succeeded":
        return HTMLResponse("Not ready", status_code=404)
    ctx = (task.get("ctx") or {}).copy()
    logger.info("task %s rendered", task_id)
    response = templates.TemplateResponse(
        request, "results.html", ctx, headers={"Cache-Control": "no-store"}
    )
    _task_delete(task_id)
    return response


async def _run_scan_task(
    task_id: str,
    tickers: list[str],
    params: dict,
    sort_key: str,
    scan_type: str,
) -> None:
    total = len(tickers)
    start_ts = time.monotonic()
    _task_update(task_id, state="running", done=0, total=total, percent=0.0)
    logger.info("scan_start id=%s total=%d", task_id, total)

    def prog(done: int, total_count: int, msg: str) -> None:
        pct = 0.0 if total_count == 0 else (done / total_count) * 100.0
        if total_count and done >= total_count:
            pct = min(pct, 99.9)
        _task_update(
            task_id,
            done=done,
            total=total_count,
            percent=pct,
            state="running",
        )
        logger.info("task_progress id=%s %d/%d", task_id, done, total_count)

    def wait_cb(wait: float) -> None:
        if wait > 0:
            _task_update(task_id, message=f"waiting {wait:.1f}s due to rate limit")
        else:
            _task_update(task_id, message="")

    http_client.set_wait_callback(wait_cb)

    success = False
    plan = _build_scan_plan(tickers, params)
    remaining_symbols = len(plan.need_fetch)
    fetch_elapsed_ms = 0.0

    disabled_fetch, disabled_reason, _, _ = schwab_client.disabled_state()
    if disabled_fetch:
        logger.info(
            "routes fetch_skipped_schwab_disabled reason=%s",
            disabled_reason or "unknown",
        )
    if remaining_symbols and not disabled_fetch:
        logger.info(
            "routes fetch_start symbols=%d interval=%s",
            remaining_symbols,
            plan.interval,
        )

        def _progress(symbol: str, pending: int, _total: int) -> None:
            nonlocal remaining_symbols
            remaining_symbols = pending

        fetch_started = _perf_counter()
        fetch_task = asyncio.create_task(
            data_provider.fetch_bars_async(
                plan.need_fetch,
                plan.interval,
                plan.window_start,
                plan.window_end,
                progress_cb=_progress,
            )
        )

        async def _watchdog() -> None:
            while True:
                try:
                    await asyncio.wait_for(asyncio.shield(fetch_task), timeout=30.0)
                    return
                except asyncio.TimeoutError:
                    logger.warning(
                        "routes fetch_heartbeat pending=%d total=%d",
                        remaining_symbols,
                        len(plan.need_fetch),
                    )

        watchdog = asyncio.create_task(_watchdog())
        try:
            fetch_result = await fetch_task
        finally:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog

        elapsed = _perf_counter() - fetch_started
        fetch_elapsed_ms = elapsed * 1000.0
        ok = getattr(fetch_result, "ok", len(plan.need_fetch))
        err = getattr(fetch_result, "err", 0)
        elapsed_seconds = getattr(fetch_result, "elapsed", elapsed)
        logger.info(
            "routes fetch_done ok=%d err=%d elapsed=%.2fs",
            ok,
            err,
            elapsed_seconds,
        )

    try:
        perform_scan = _perform_scan
        sig = inspect.signature(perform_scan)
        extra: dict[str, object] = {}
        if "plan" in sig.parameters:
            extra["plan"] = plan
        if "fetch_missing" in sig.parameters:
            extra["fetch_missing"] = False
        if "fetch_elapsed_ms" in sig.parameters:
            extra["fetch_elapsed_ms"] = fetch_elapsed_ms
        rows, skipped, metrics = await asyncio.to_thread(
            perform_scan,
            tickers,
            params,
            sort_key,
            prog,
            **extra,
        )
        duration = time.monotonic() - start_ts
        csv_headers, csv_rows = _rows_to_csv_table(rows)
        ctx = {
            "rows": rows,
            "ran_at": now_et().strftime("%I:%M:%S %p").lstrip("0"),
            "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}",
            "skipped_missing_data": skipped,
            "metrics": metrics,
            "summary": {
                "successes": len(rows),
                "empties": skipped,
                "errors": 0,
                "duration": duration,
            },
            "errors": [],
            "csv_headers": csv_headers,
            "csv_rows": csv_rows,
        }
        interim_percent = 100.0 if total == 0 else 99.9
        _task_update(
            task_id,
            state="running",
            percent=interim_percent,
            done=total,
            ctx=ctx,
            message="",
        )
        await asyncio.sleep(0.1)
        _task_update(
            task_id,
            state="succeeded",
            percent=100.0,
            done=total,
            ctx=ctx,
            message="",
        )
        logger.info(
            "scan_done id=%s duration=%.2fs successes=%d empties=%d errors=%d no_gap=%d avg_ms=%.1f p95_ms=%.1f",
            task_id,
            duration,
            len(rows),
            skipped,
            0,
            metrics.get("symbols_no_gap", 0),
            metrics.get("avg_per_symbol_ms", 0.0),
            metrics.get("p95_per_symbol_ms", 0.0),
        )
        success = True
    except asyncio.CancelledError:
        logger.warning("scan_cancelled id=%s", task_id)
        if not success:
            _task_update(task_id, message="cancelled")
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("scan_error id=%s", task_id)
        _task_update(
            task_id,
            state="failed",
            message=str(exc),
            ctx={"errors": [tb]},
        )
    finally:
        http_client.set_wait_callback(None)


@router.post("/scanner/parity")
def scanner_parity(request: Request):
    PARAMS = dict(
        interval="15m",
        direction="BOTH",
        target_pct=1.5,
        stop_pct=0.7,
        window_value=8.0,
        window_unit="Hours",
        lookback_years=2.0,
        max_tt_bars=20,
        min_support=20,
        delta_assumed=0.25,
        theta_per_day_pct=0.20,
        atrz_gate=-0.5,
        slope_gate_pct=-0.01,
        use_regime=1,
        regime_trend_only=0,
        vix_z_max=3.0,
        slippage_bps=7.0,
        vega_scale=0.03,
        scan_min_hit=55.0,
        scan_max_dd=1.0,
    )
    sort_key = request.query_params.get("sort")
    rows = []
    for t in TOP150:
        r = compute_scan_for_ticker(t, PARAMS) or {}
        if not r:
            continue
        if (
            r.get("hit_pct", 0) >= PARAMS["scan_min_hit"]
            and r.get("avg_dd_pct", 999) <= PARAMS["scan_max_dd"]
        ):
            rows.append(r)

    rows = _sort_by_lb95_roi_support(rows)

    _update_heatmap(rows)

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "rows": _sort_rows(rows, sort_key),
            "ran_at": now_et().strftime("%Y-%m-%d %H:%M"),
            "note": f"TOP150 parity run • kept {len(rows)}",
        },
    )


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _twiml_response(message: str) -> Response:
    payload = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        f"<Response><Message>{html.escape(message)}</Message></Response>"
    )
    return Response(content=payload, media_type="application/xml")


_PHONE_REGEX = re.compile(r"^\+\d{10,15}$")


async def _load_sms_start_payload(request: Request) -> dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            data = await request.json()
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        return {k: v for k, v in form.items()}
    # Fallback attempts for missing/unknown content type headers
    try:
        data = await request.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    body = await request.body()
    if not body:
        return {}
    parsed = urllib.parse.parse_qs(body.decode(), keep_blank_values=True)
    return {k: values[-1] if values else "" for k, values in parsed.items()}


def _parse_consent(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


@router.post("/api/sms/verify/start")
async def sms_verify_start(request: Request):
    user_id = _request_user_id(request)
    payload = await _load_sms_start_payload(request)
    phone_raw = str(payload.get("phone") or "").strip()
    consent_value = _parse_consent(payload.get("consent"))
    if not consent_value:
        return JSONResponse({"ok": False, "error": "Consent is required"}, status_code=400)
    normalized = sms_consent.normalize_phone(phone_raw)
    if not normalized or not _PHONE_REGEX.fullmatch(normalized):
        return JSONResponse({"ok": False, "error": "Invalid phone number"}, status_code=400)
    consent_text = str(payload.get("consent_text") or _SMS_CONSENT_TEXT).strip()
    if not consent_text:
        return JSONResponse({"ok": False, "error": "Consent text required"}, status_code=400)
    method = str(payload.get("method") or "settings").strip() or "settings"
    verification_id = twilio_client.start_verification(normalized)
    if not verification_id:
        return JSONResponse(
            {"ok": False, "error": "Verification is unavailable"}, status_code=400
        )
    logger.info(
        "sms_verify_start",
        extra={"user_id": user_id, "phone": normalized, "method": method},
    )
    return {"ok": True, "sent": True}


@router.post("/api/sms/verify/check")
async def sms_verify_check(
    request: Request,
    payload: dict = Body(...),
    db=Depends(get_db),
):
    user_id = _request_user_id(request)
    phone = sms_consent.normalize_phone(str(payload.get("phone") or ""))
    code = str(payload.get("code") or "").strip()
    consent_text = str(payload.get("consent_text") or _SMS_CONSENT_TEXT).strip()
    method = str(payload.get("method") or "settings").strip() or "settings"
    if not phone:
        return JSONResponse({"ok": False, "error": "Invalid phone number"}, status_code=400)
    if not code:
        return JSONResponse({"ok": False, "error": "Verification code required"}, status_code=400)
    if not consent_text:
        return JSONResponse({"ok": False, "error": "Consent text required"}, status_code=400)
    ok, verification_id = twilio_client.check_verification(phone, code)
    if not ok:
        return JSONResponse(
            {"ok": False, "error": "Invalid verification code"}, status_code=400
        )
    ip = _client_ip(request)
    user_agent = request.headers.get("user-agent")
    record = sms_consent.record_consent(
        user_id,
        phone,
        consent_text,
        ip=ip,
        user_agent=user_agent,
        verification_id=verification_id,
        method=method,
        db_cursor=db,
    )
    logger.info(
        "sms_verify_check_ok",
        extra={"user_id": user_id, "phone": phone, "method": method},
    )
    return {"ok": True, "consent": record}


@router.post("/twilio/inbound-sms")
async def twilio_inbound_sms(request: Request, db=Depends(get_db)):
    form = await request.form()
    from_number = sms_consent.normalize_phone(
        str(form.get("From") or form.get("from") or "")
    )
    body_raw = str(form.get("Body") or form.get("body") or "").strip()
    lowered = body_raw.lower()

    if not from_number:
        return _twiml_response("Missing sender number.")

    if lowered in {"stop", "stopall", "unsubscribe", "quit"}:
        sms_consent.revoke_phone(from_number, db_cursor=db)
        return _twiml_response(
            "You’re opted out of Petra Stock SMS alerts. Reply START to opt back in."
        )

    if lowered in {"start", "unstop"}:
        latest = sms_consent.latest_for_phone(from_number, include_revoked=True, db_cursor=db)
        if latest and not latest.get("revoked_at"):
            return _twiml_response("You’re already opted in to Petra Stock SMS alerts.")

        consent_text = (latest or {}).get("consent_text") or _SMS_CONSENT_TEXT
        user_id = (latest or {}).get("user_id")
        record = sms_consent.record_consent(
            user_id,
            from_number,
            consent_text,
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            method="sms-keyword",
            db_cursor=db,
        )
        logger.info(
            "sms_inbound_start",
            extra={
                "phone": from_number,
                "user_id": record.get("user_id") if isinstance(record, dict) else None,
                "method": "sms-keyword",
            },
        )
        return _twiml_response("You’re opted in to Petra Stock SMS alerts.")

    if lowered == "help":
        return _twiml_response(
            "Petra Stock alerts. Msg & data rates may apply. Contact support@petrastock.com. Reply STOP to opt out."
        )

    if re.fullmatch(r"\d{4,8}", lowered):
        latest = sms_consent.latest_for_phone(from_number, db_cursor=db)
        consent_text = (latest or {}).get("consent_text") or _SMS_CONSENT_TEXT
        user_id = (latest or {}).get("user_id")
        ok, verification_id = twilio_client.check_verification(from_number, lowered)
        if not ok:
            return _twiml_response("Invalid verification code. Reply START for a new one.")
        sms_consent.record_consent(
            user_id,
            from_number,
            consent_text,
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            verification_id=verification_id,
            method="sms-keyword",
            db_cursor=db,
        )
        return _twiml_response("You’re opted in to Petra Stock SMS alerts.")

    return _twiml_response(
        "Unrecognized response. Reply HELP for info or START to opt back in."
    )

@router.post("/debug/simulate")
async def simulate_favorite(payload: dict = Body(...), db=Depends(get_db)):
    if not DEBUG_SIMULATION:
        raise HTTPException(status_code=403, detail="Simulation disabled")
    payload = payload or {}
    symbol = (payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    direction = canonical_direction(payload.get("direction")) or "UP"
    outcome = (payload.get("outcome") or "hit").strip().lower() or "hit"
    channel = (payload.get("channel") or "email").strip().lower() or "email"
    outcomes_mode = (payload.get("outcomes_mode") or "hit").strip().lower() or "hit"
    interval = (payload.get("interval") or "15m").strip() or "15m"
    bar_ts_raw = payload.get("bar_ts")
    if bar_ts_raw:
        text = str(bar_ts_raw)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        bar_dt = datetime.fromisoformat(text)
    else:
        bar_dt = datetime.now(timezone.utc)
    if bar_dt.tzinfo is None:
        bar_dt = bar_dt.replace(tzinfo=timezone.utc)
    bar_dt = bar_dt.astimezone(timezone.utc)
    aligned = _align_to_bar(bar_dt, interval)
    favorite_key = payload.get("favorite_id") or f"sim:{symbol}:{direction}"
    dedupe_key = _dedupe_key(favorite_key, interval, aligned, outcome)
    was_already = was_sent_key(
        favorite_key,
        aligned.isoformat(),
        interval=interval,
        dedupe_key=dedupe_key,
    )

    if outcome == "hit":
        sim_event = build_sim_hit(symbol, direction, aligned)
    elif outcome == "stop":
        sim_event = build_sim_stop(symbol, direction, aligned)
    else:
        sim_event = SimEvent(symbol=symbol, direction=direction, outcome=outcome, bar_ts=aligned)

    subject, _ = favorites_alerts.build_preview(
        symbol,
        direction,
        channel=channel,
        outcomes=outcomes_mode,
        outcome_mode=outcomes_mode,
        simulated=True,
    )
    bodies: dict[str, str] = {}
    for variant in ("email", "mms", "sms"):
        ok, preview_payload = favorites_alerts.enrich_and_send_test(
            symbol,
            direction,
            channel=variant,
            outcomes=outcomes_mode,
        )
        if ok and isinstance(preview_payload, dict):
            bodies[variant] = str(preview_payload.get("body", ""))
            if variant == channel and not subject:
                subject = str(preview_payload.get("subject", subject))
    if channel not in bodies:
        bodies[channel] = bodies.get("email") or bodies.get("mms") or bodies.get("sms") or ""

    smtp_config = None
    if isinstance(payload.get("smtp"), Mapping):
        smtp_config = {k: payload["smtp"].get(k) for k in ("host", "port", "user", "password", "mail_from")}

    email_recipients = None
    if isinstance(payload.get("recipients"), list):
        email_recipients = [str(value) for value in payload["recipients"] if value]

    response: dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "outcome": outcome,
        "dedupe_key": dedupe_key,
        "was_sent": bool(was_already),
        "event": {
            "symbol": sim_event.symbol,
            "direction": sim_event.direction,
            "outcome": sim_event.outcome,
            "bar_ts": sim_event.bar_ts.isoformat(),
        },
    }
    if not was_already:
        delivery = deliver_preview_alert(
            subject,
            bodies,
            channel=channel,
            favorite_id=favorite_key,
            bar_time=aligned.isoformat(),
            interval=interval,
            dedupe_key=dedupe_key,
            simulated=True,
            outcome=outcome,
            symbol=symbol,
            direction=direction,
            smtp_config=smtp_config,
            recipients=email_recipients,
        )
        response["delivery"] = delivery
        exit_dt = aligned + timedelta(minutes=15)
        roi_value = 1.8 if outcome == "hit" else (-1.0 if outcome == "stop" else 0.0)
        tt_bars = 3 if outcome == "hit" else 2
        dd_value = 0.5 if outcome != "hit" else 0.2
        log_forward_entry(
            db,
            favorite_key,
            aligned.isoformat(),
            None,
            None,
            simulated=True,
        )
        log_forward_exit(
            db,
            favorite_key,
            aligned.isoformat(),
            exit_dt.isoformat(),
            None,
            outcome,
            roi_value,
            tt_bars,
            dd_value,
            simulated=True,
        )
        export_simulation_artifacts([symbol])
    return response


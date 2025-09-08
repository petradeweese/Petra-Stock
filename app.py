from __future__ import annotations

import os
import json
import math
import time
import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from threading import Lock

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Market hours / calendar (XNYS)
import pandas as pd
import pandas_market_calendars as mcal

# Ensure required directories exist before mounting StaticFiles
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# -----------------------------
# App & Templating
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def _sort_rows(rows, sort_key):
    if not rows or not sort_key:
        return rows
    keymap = {
        'ticker': lambda r: (r.get('ticker') or ''),
        'roi':    lambda r: (r.get('avg_roi_pct') or 0.0),
        'hit':    lambda r: (r.get('hit_pct') or 0.0),
    }
    keyfn = keymap.get(sort_key)
    if not keyfn: return rows
    reverse = sort_key != 'ticker'
    return sorted(rows, key=keyfn, reverse=reverse)


# -----------------------------
# Paths & DB
# -----------------------------
DB_PATH = "patternfinder.db"

# -----------------------------
# DB Schema (+ migrations)
# -----------------------------
SCHEMA = [
    # Settings singleton row
    """
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY CHECK (id=1),
        smtp_user TEXT,
        smtp_pass TEXT,
        recipients TEXT,
        scheduler_enabled INTEGER DEFAULT 0,
        throttle_minutes INTEGER DEFAULT 60,
        last_boundary TEXT,
        last_run_at TEXT
    );
    """,
    """
    INSERT OR IGNORE INTO settings
      (id, smtp_user, smtp_pass, recipients, scheduler_enabled, throttle_minutes, last_boundary, last_run_at)
    VALUES
      (1, '', '', '', 0, 60, '', '');
    """,
    # Favorites
    """
    CREATE TABLE IF NOT EXISTS favorites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        direction TEXT NOT NULL,
        interval TEXT NOT NULL DEFAULT '15m',
        rule TEXT NOT NULL,
        target_pct REAL DEFAULT 1.0,
        stop_pct REAL DEFAULT 0.5,
        window_value REAL DEFAULT 4.0,
        window_unit TEXT DEFAULT 'Hours',
        lookback_years REAL DEFAULT 0.2,
        max_tt_bars INTEGER DEFAULT 12,
        min_support INTEGER DEFAULT 20,
        delta REAL DEFAULT 0.4,
        theta_day REAL DEFAULT 0.2,
        atrz REAL DEFAULT 0.10,
        slope REAL DEFAULT 0.02,
        use_regime INTEGER DEFAULT 0,
        trend_only INTEGER DEFAULT 0,
        vix_z_max REAL DEFAULT 3.0,
        slippage_bps REAL DEFAULT 7.0,
        vega_scale REAL DEFAULT 0.03,
        ref_avg_dd REAL
    );
    """,
    # Runs (archive)
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT,
        scan_type TEXT,
        params_json TEXT,
        universe TEXT,
        finished_at TEXT,
        hit_count INTEGER DEFAULT 0
    );
    """,
    # Run results (archive)
    """
    CREATE TABLE IF NOT EXISTS run_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        ticker TEXT,
        direction TEXT,
        avg_roi_pct REAL,
        hit_pct REAL,
        support INTEGER,
        avg_tt REAL,
        avg_dd_pct REAL,
        stability REAL,
        rule TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_run_results_run ON run_results(run_id);",
]

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
for stmt in SCHEMA:
    cur.executescript(stmt)
conn.commit()

# -----------------------------
# Calendar / TZ helpers
# -----------------------------
XNYS = mcal.get_calendar("XNYS")  # New York Stock Exchange
TZ = XNYS.tz  # exchange tz (pytz/zoneinfo)

def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(TZ)

def market_is_open(ts: Optional[datetime] = None) -> bool:
    ts = ts or now_et()
    sched = XNYS.schedule(start_date=ts.date(), end_date=ts.date())
    if sched.empty:
        return False
    open_ts = sched.iloc[0]["market_open"].to_pydatetime().astimezone(TZ)
    close_ts = sched.iloc[0]["market_close"].to_pydatetime().astimezone(TZ)
    return open_ts <= ts <= close_ts

# -----------------------------
# Lightweight TTL cache
# -----------------------------
class TTLCache:
    def __init__(self, ttl_seconds: int = 120):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            hit = self._store.get(key)
            if not hit:
                return None
            ts, val = hit
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, val: Any):
        with self._lock:
            self._store[key] = (time.time(), val)

CACHE = TTLCache(ttl_seconds=120)

# -----------------------------
# Settings helpers
# -----------------------------
def get_settings() -> sqlite3.Row:
    c = conn.cursor()
    c.execute("SELECT * FROM settings WHERE id=1")
    return c.fetchone()

def set_last_run(boundary_iso: str):
    c = conn.cursor()
    c.execute(
        "UPDATE settings SET last_boundary=?, last_run_at=? WHERE id=1",
        (boundary_iso, now_et().isoformat())
    )
    conn.commit()

# -----------------------------
# Universe helpers
# -----------------------------
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMD","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK",
    "BLK","BMY","BRK-B","C","CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS",
    "CVX","DHR","DIS","DOW","DUK","EMR","EXC","F","FDX","GE","GILD","GM","GOOGL","GS","HD",
    "HON","IBM","INTC","JNJ","JPM","KHC","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    "META","MET","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE",
    "PG","PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","SPGI","T","TGT","TMO","TMUS","TXN",
    "UNH","UNP","UPS","USB","V","VZ","WBA","WFC","WELL","WMT","XOM"
]

TOP150 = [
    "SPY","QQQ","AAPL","MSFT","NVDA","AMZN","TSLA","META","GOOGL","AMD","NFLX","AVGO","CRM","ADBE",
    "PYPL","INTC","ORCL","CSCO","QCOM","TXN","MU","SMCI","PLTR","SNOW","NOW","TEAM","JPM","BAC","GS",
    "MS","WFC","C","V","MA","AXP","KO","PEP","PG","PM","T","VZ","HD","LOW","COST","WMT","DIS","CMCSA",
    "TGT","MCD","SBUX","ABNB","UBER","NKE","DE","CAT","BA","GE","GM","F","XOM","CVX","SLB","COP","OXY",
    "PFE","MRK","LLY","ABBV","BMY","UNH","TMO","ISRG","MDT","CVS","CI","HUM","VRTX","REGN","PANW","FTNT",
    "CRWD","ZS","OKTA","DDOG","NET","CHTR","TMUS","NOC","LMT","RTX","HON","MMM","DELL","HPQ","IBM","INTU",
    "ADP","WDAY","ORLY","AZO","DAL","UAL","LUV","CCL","RCL","CMG","DPZ","YUM","FCX","NUE","APD","LIN",
    "DOW","CF","MOS","FSLR","ENPH","SEDG","BLK","SCHW","MSCI","SPGI","ICE","CME","COIN","SOFI","IYR","O",
    "AMT","PLD","EQIX","BABA","JD","PDD","NIO","XPEV","LI","ROKU","RBLX","RIVN","LCID"
]

def ticker_universe(scan_type: str, explicit_ticker: Optional[str]) -> List[str]:
    if scan_type == "scan150":
        return TOP150
    if scan_type == "sp100":
        return SP100
    if scan_type == "single" and explicit_ticker:
        return [explicit_ticker.strip().upper()]
    return ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]  # small default for sanity

# -----------------------------
# Adapter to your original ROI engine
# -----------------------------
_real_scan_single: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None

def _install_real_engine_adapter():
    """
    Integrate with pattern_finder_app. We support:
      - scan_parallel_threaded(tickers, cfg, max_workers=None) -> DataFrame
      - scan_parallel(tickers, cfg, max_workers=None) -> DataFrame
      - analyze_roi_mode(...) -> (model, df_te, forward)
    We normalize to: {ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule}
    """
    global _real_scan_single
    try:
        import importlib
        mod = importlib.import_module("pattern_finder_app")

        fn = None
        mode = None
        if hasattr(mod, "scan_parallel_threaded"):
            fn = getattr(mod, "scan_parallel_threaded")
            mode = "threaded"
        elif hasattr(mod, "scan_parallel"):
            fn = getattr(mod, "scan_parallel")
            mode = "parallel"
        elif hasattr(mod, "analyze_roi_mode"):
            fn = getattr(mod, "analyze_roi_mode")
            mode = "single"
        else:
            print("[adapter] pattern_finder_app found, but no known scan function.")
            _real_scan_single = None
            return

        def _row_to_dict(row: dict, params: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)

            def get(*keys, default=None):
                for k in keys:
                    if k in out and out[k] is not None:
                        return out[k]
                return default

            def fnum(x):
                try: return float(x)
                except: return 0.0

            def to_pct(x):
                x = fnum(x)
                # If looks like a fraction (<=1), convert to %
                if abs(x) <= 1.0:
                    return x * 100.0
                return x

            roi   = get("avg_roi_pct", "avg_roi", default=None)
            hit   = get("hit_pct", "hit_rate", default=None)
            dd    = get("avg_dd_pct", "avg_dd", default=0.0)
            supp  = get("support", "n", "count", default=0)
            tt    = get("avg_tt", default=0.0)
            stab  = get("stability", default=0.0)
            rule  = get("rule", "rule_str", default="")
            direct= get("direction", default=params.get("direction", "UP"))
            tkr   = get("ticker", default=params.get("ticker","?"))

            roi_pct = to_pct(roi) if roi is not None else None
            hit_pct = to_pct(hit) if hit is not None else None
            dd_pct  = to_pct(dd)  if dd  is not None else 0.0

            if roi_pct is None or hit_pct is None:
                return {}

            return {
                "ticker": str(tkr),
                "direction": str(direct).upper(),
                "avg_roi_pct": float(roi_pct),
                "hit_pct": float(hit_pct),
                "support": int(supp or 0),
                "avg_tt": fnum(tt),
                "avg_dd_pct": float(dd_pct),
                "stability": fnum(stab),
                "rule": str(rule or ""),
            }

        if mode in ("threaded", "parallel"):
            def wrapper(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    df = fn([ticker], params)
                    try:
                        import pandas as pd
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            row = df.iloc[0].to_dict()
                            return _row_to_dict(row, params)
                    except Exception:
                        pass
                    if isinstance(df, list) and df:
                        return _row_to_dict(df[0], params)
                    if isinstance(df, dict) and df:
                        return _row_to_dict(df, params)
                    return {}
                except Exception as e:
                    print(f"[adapter] scan_* error for {ticker}: {e!r}")
                    return {}
        else:
            def wrapper(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    model, df_te, _ = fn(
                        ticker=ticker,
                        interval=params.get("interval","15m"),
                        direction=params.get("direction","UP"),
                        target_pct=params.get("target_pct",1.0),
                        stop_pct=params.get("stop_pct",0.5),
                        window_value=params.get("window_value",4.0),
                        window_unit=params.get("window_unit","Hours"),
                        lookback_years=params.get("lookback_years",2.0),
                        max_tt_bars=params.get("max_tt_bars",12),
                        min_support=params.get("min_support",20),
                        delta_assumed=params.get("delta_assumed",0.40),
                        theta_per_day_pct=params.get("theta_per_day_pct",0.20),
                        atrz_gate=params.get("atrz_gate",0.10),
                        slope_gate_pct=params.get("slope_gate_pct",0.02),
                        use_regime=bool(params.get("use_regime",0)),
                        regime_trend_only=bool(params.get("regime_trend_only",0)),
                        vix_z_max=params.get("vix_z_max",3.0),
                        event_mask=None,
                        slippage_bps=params.get("slippage_bps",7.0),
                        vega_scale=params.get("vega_scale",0.03),
                    )
                    if df_te is None or getattr(df_te, "empty", True):
                        return {}
                    df_te = df_te.sort_values(["avg_roi","hit_rate","support","stability"], ascending=[False,False,False,False])
                    row = df_te.iloc[0].to_dict()
                    mapped = {
                        "ticker": ticker,
                        "direction": row.get("direction", params.get("direction","UP")),
                        "avg_roi": row.get("avg_roi", 0.0),
                        "hit_rate": row.get("hit_rate", 0.0),
                        "support": row.get("support", 0),
                        "avg_tt": row.get("avg_tt", 0.0),
                        "avg_dd": row.get("avg_dd", 0.0),
                        "stability": row.get("stability", 0.0),
                        "rule": row.get("rule", ""),
                    }
                    return _row_to_dict(mapped, params)
                except Exception as e:
                    print(f"[adapter] analyze_roi_mode error for {ticker}: {e!r}")
                    return {}

        _real_scan_single = wrapper
        print(f"[adapter] Using REAL engine from pattern_finder_app ({mode}).")
    except Exception as e:
        print("[adapter] pattern_finder_app not available or failed to import:", repr(e))
        _real_scan_single = None

_install_real_engine_adapter()

# Ensure the adapter mirrors desktop logic if available
try:
    import pattern_finder_app as _pfa
    import pandas as pd
except Exception:
    _pfa = None

def _desktop_like_single(ticker: str, params: dict) -> dict:
    """Match pattern_finder_app._scan_worker for a SINGLE ticker+direction."""
    if _pfa is None:
        return {}
    try:
        px = _pfa._download_prices(ticker, params["interval"], params["lookback_years"])
        ev = _pfa.build_event_mask(px.index, set())
        model, df, _ = _pfa.analyze_roi_mode(
            ticker=ticker,
            interval=params["interval"],
            direction=params["direction"],
            target_pct=params["target_pct"],
            stop_pct=params["stop_pct"],
            window_value=params["window_value"],
            window_unit=params["window_unit"],
            lookback_years=params["lookback_years"],
            max_tt_bars=params["max_tt_bars"],
            min_support=params["min_support"],
            delta_assumed=params["delta_assumed"],
            theta_per_day_pct=params["theta_per_day_pct"],
            atrz_gate=params["atrz_gate"],
            slope_gate_pct=params["slope_gate_pct"],
            use_regime=params["use_regime"],
            regime_trend_only=params["regime_trend_only"],
            vix_z_max=params["vix_z_max"],
            event_mask=ev,
            slippage_bps=params["slippage_bps"],
            vega_scale=params["vega_scale"],
        )
        if df is None or df.empty:
            return {}
        df = df[(df["hit_rate"] * 100.0 >= params["scan_min_hit"]) &
                (df["avg_dd"] * 100.0 <= params["scan_max_dd"])]
        if df.empty:
            return {}
        r = df.sort_values(
            ["avg_roi", "hit_rate", "support", "stability"],
            ascending=[False, False, False, False]
        ).iloc[0]
        return {
            "ticker": ticker,
            "direction": r.get("direction", params["direction"]),
            "avg_roi_pct": float(r["avg_roi"]) * 100.0,
            "hit_pct": float(r["hit_rate"]) * 100.0,
            "support": int(r["support"]),
            "avg_tt": float(r["avg_tt"]) if pd.notna(r["avg_tt"]) else 0.0,
            "avg_dd_pct": float(r["avg_dd"]) * 100.0,
            "stability": float(r.get("stability", 0.0)),
            "rule": str(r["rule"]),
        }
    except Exception as e:
        print(f"[adapter-single] {ticker} failed: {e!r}")
        return {}

# -----------------------------
# Scheduler (15m during market hours, throttled)
# -----------------------------
async def favorites_loop():
    print("[scheduler] started")
    while True:
        try:
            ts = now_et()
            if market_is_open(ts):
                boundary = ts.replace(second=0, microsecond=0)
                boundary = boundary.replace(minute=(boundary.minute - boundary.minute % 15))

                st = get_settings()
                throttle = int(st["throttle_minutes"] or 60)
                last_boundary = st["last_boundary"] or ""
                last_run_at = st["last_run_at"] or ""

                should_run = (boundary.isoformat() != last_boundary)
                if last_run_at:
                    last_dt = datetime.fromisoformat(last_run_at)
                    if (ts - last_dt).total_seconds() < throttle * 60:
                        should_run = False

                if should_run:
                    c = conn.cursor()
                    c.execute("SELECT ticker, direction, interval, rule FROM favorites ORDER BY id DESC")
                    favs = [dict(r) for r in c.fetchall()]

                    params = dict(
                        interval="15m",
                        direction="BOTH",
                        scan_min_hit=50.0,
                        atrz_gate=0.10,
                        slope_gate_pct=0.02,
                    )
                    hits = []
                    for f in favs:
                        row = compute_scan_for_ticker(f["ticker"], params)
                        if row and row.get("hit_pct", 0) >= 50 and row.get("avg_roi_pct", 0) > 0:
                            hits.append(row)

                    # TODO: email YES hits in a readable format
                    # TODO: archive favorites 15m scan results only if there are YES hits
                    set_last_run(boundary.isoformat())

            await asyncio.sleep(60)
        except Exception as e:
            print("[scheduler] error:", repr(e))
            await asyncio.sleep(60)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(favorites_loop())

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/results/{run_id}", response_class=HTMLResponse)
def results_from_archive(request: Request, run_id: int):
    c = conn.cursor()
    c.execute("SELECT * FROM runs WHERE id=?", (run_id,))
    run = c.fetchone()
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    c.execute(
        """SELECT ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule
           FROM run_results WHERE run_id=?""",
        (run_id,),
    )
    rows = [dict(r) for r in c.fetchall()]
    rows.sort(key=lambda r: (r["avg_roi_pct"], r["hit_pct"], r["support"], r["stability"]), reverse=True)

    return templates.TemplateResponse(request, "results.html", {
            "rows": rows,
            "scan_type": run["scan_type"],
            "universe_count": len((run["universe"] or "").split(",")) if run["universe"] else 0,
            "run_id": run_id,
        },
    )

@app.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request):
    c = conn.cursor()
    c.execute("SELECT * FROM favorites ORDER BY id DESC")
    favs = c.fetchall()
    return templates.TemplateResponse(request, "favorites.html", {"favorites": favs})

@app.post("/favorites/add")
async def favorites_add(request: Request):
    payload = await request.json()
    t = (payload.get("ticker") or "").strip().upper()
    rule = payload.get("rule") or ""
    direction = (payload.get("direction") or "UP").strip().upper()

    if not t or not rule:
        return JSONResponse({"ok": False, "error": "missing ticker or rule"}, status_code=400)

    c = conn.cursor()
    c.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES (?, ?, '15m', ?)",
        (t, direction, rule),
    )
    conn.commit()
    return {"ok": True}

@app.get("/archive", response_class=HTMLResponse)
def archive_page(request: Request):
    c = conn.cursor()
    c.execute("SELECT id, started_at, scan_type, universe, finished_at, hit_count FROM runs ORDER BY id DESC LIMIT 200")
    runs = c.fetchall()
    return templates.TemplateResponse(request, "archive.html", {"runs": runs})

@app.post("/archive/save")
async def archive_save(request: Request):
    """
    Accepts JSON: { params: {...}, rows: [ {ticker, direction, avg_roi_pct, hit_pct, support, avg_dd_pct, stability, rule}, ... ] }
    Writes a run to `runs` and details to `run_results`. Returns {ok: True, run_id}.
    """
    try:
        payload = await request.json()
        params = payload.get("params", {}) or {}
        rows = payload.get("rows", []) or []
        if not rows:
            return JSONResponse({"ok": False, "error": "no rows"}, status_code=400)

        started = datetime.now(timezone.utc).isoformat()
        finished = started
        scan_type = str(params.get("scan_type") or "scan150")
        universe = ",".join({r.get("ticker","") for r in rows if r.get("ticker")})

        c = conn.cursor()
        c.execute(
            """
            INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (started, scan_type, json.dumps(params), universe, finished, len(rows)),
        )
        run_id = c.lastrowid

        for r in rows:
            c.execute(
                """
                INSERT INTO run_results
                  (run_id, ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    r.get("ticker"),
                    (r.get("direction") or "UP"),
                    float(r.get("avg_roi_pct") or 0.0),
                    float(r.get("hit_pct") or 0.0),
                    int(r.get("support") or 0),
                    float(r.get("avg_tt") or 0.0),
                    float(r.get("avg_dd_pct") or 0.0),
                    float(r.get("stability") or 0.0),
                    r.get("rule") or "",
                ),
            )
        conn.commit()
        return {"ok": True, "run_id": run_id}
    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    st = get_settings()
    return templates.TemplateResponse(request, "settings.html", {"st": st})

@app.post("/settings/save")
def settings_save(
    request: Request,
    smtp_user: str = Form(""),
    smtp_pass: str = Form(""),
    recipients: str = Form(""),
    scheduler_enabled: int = Form(1),
    throttle_minutes: int = Form(60),
):
    c = conn.cursor()
    c.execute(
        """
        UPDATE settings
           SET smtp_user=?, smtp_pass=?, recipients=?, scheduler_enabled=?, throttle_minutes=?
         WHERE id=1
        """,
        (smtp_user.strip(), smtp_pass.strip(), recipients.strip(), int(scheduler_enabled), int(throttle_minutes)),
    )
    conn.commit()
    return RedirectResponse(url="/settings", status_code=302)

# -----------------------------
# Scanner: run (HTMX posts)
# -----------------------------
@app.post("/scanner/run", response_class=HTMLResponse)
async def scanner_run(request: Request):
    """
    HTMX target. Reads form, normalizes params, runs the scan, and renders results.html.
    """
    form = await request.form()
    params = _coerce_scan_params(form)

    # Figure out ticker universe
    scan_type = params.get("scan_type", "scan150")
    single_ticker = (form.get("ticker") or "").strip().upper()
    if scan_type.lower() in ("single", "single_ticker") and single_ticker:
        tickers = [single_ticker]
    elif scan_type.lower() in ("sp100", "sp_100", "sp100_scan"):
        tickers = SP100
    else:
        # default Top 150
        tickers = TOP150

    # Support server-side sorting triggered by buttons
    sort_key = (form.get("sort") or "").strip().lower()
    if sort_key not in ("ticker", "roi", "hit"):
        sort_key = ""

    # Run the scan (threaded path under the hood)
    rows = []
    for t in tickers:
        r = compute_scan_for_ticker(t, params)
        if r:
            rows.append(r)

    # Optional filters (match desktop defaults)
    try:
        scan_min_hit = float(params.get("scan_min_hit", 0.0))
        scan_max_dd  = float(params.get("scan_max_dd", 100.0))
    except Exception:
        scan_min_hit, scan_max_dd = 0.0, 100.0

    rows = [
        r for r in rows
        if (r.get("hit_pct", 0.0) >= scan_min_hit) and (r.get("avg_dd_pct", 100.0) <= scan_max_dd)
    ]

    # Sorting
    if sort_key == "ticker":
        rows.sort(key=lambda r: (r.get("ticker") or ""))
    elif sort_key == "roi":
        rows.sort(key=lambda r: (r.get("avg_roi_pct", 0.0),
                                 r.get("hit_pct", 0.0),
                                 r.get("support", 0)), reverse=True)
    elif sort_key == "hit":
        rows.sort(key=lambda r: (r.get("hit_pct", 0.0),
                                 r.get("avg_roi_pct", 0.0),
                                 r.get("support", 0)), reverse=True)
    else:
        # default ranking like desktop
        rows.sort(key=lambda r: (r.get("avg_roi_pct", 0.0),
                                 r.get("hit_pct", 0.0),
                                 r.get("support", 0),
                                 r.get("stability", 0.0)), reverse=True)

    ctx = {
        "request": request,
        "rows": rows,
        "ran_at": datetime.now().strftime("%-I:%M:%S %p"),
        "note": f"{scan_type} • {params.get('interval')} • {params.get('direction')} • window {params.get('window_value')} {params.get('window_unit')}",
    }
    return templates.TemplateResponse("results.html", ctx)

@app.post("/runs/archive")
async def archive_run(request: Request):
    """
    Body: JSON with {"scan_type": "...", "params": {...}, "rows": [...], "universe": [...]}
    Saves a run + run_results. Only saves rows that passed filters (what UI showed).
    """
    payload = await request.json()
    scan_type = payload.get("scan_type", "")
    params = payload.get("params", {})
    rows = payload.get("rows", [])
    universe = payload.get("universe", [])

    started_at = now_et().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO runs(started_at, scan_type, params_json, universe, finished_at, hit_count) VALUES (?, ?, ?, ?, ?, ?)",
        (started_at, scan_type, json.dumps(params), ",".join(universe), now_et().isoformat(), len(rows)),
    )
    run_id = c.lastrowid

    for r in rows:
        c.execute(
            """INSERT INTO run_results(run_id, ticker, direction, avg_roi_pct, hit_pct, support, avg_tt, avg_dd_pct, stability, rule)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                r.get("ticker"),
                r.get("direction", "UP"),
                float(r.get("avg_roi_pct", 0)),
                float(r.get("hit_pct", 0)),
                int(r.get("support", 0)),
                float(r.get("avg_tt", 0)),
                float(r.get("avg_dd_pct", 0)),
                float(r.get("stability", 0)),
                r.get("rule", ""),
            ),
        )
    conn.commit()
    return {"ok": True, "run_id": run_id}


# === FINAL override for desktop parity ===
def compute_scan_for_ticker(ticker: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final override: delegate to _desktop_like_single so the web API matches the desktop scanner.
    Handles BOTH by evaluating UP and DOWN and returning the better-scoring row.
    """
    try:
        dirn = str(params.get("direction", "UP")).upper()
    except Exception:
        dirn = "UP"

    if dirn == "BOTH":
        a = dict(params); a["direction"] = "UP"
        b = dict(params); b["direction"] = "DOWN"
        ra = _desktop_like_single(ticker, a)
        rb = _desktop_like_single(ticker, b)
        picks = [r for r in (ra, rb) if isinstance(r, dict) and r]
        if not picks:
            return {}
        return sorted(
            picks,
            key=lambda r: (
                r.get("avg_roi_pct", 0.0),
                r.get("hit_pct", 0.0),
                r.get("support", 0),
                r.get("stability", 0.0),
            ),
            reverse=True,
        )[0]
    else:
        return _desktop_like_single(ticker, params)


# --- Parity route: run fixed TOP150 scan (matches desktop screenshot) ---

@app.post("/scanner/parity")
def scanner_parity(request: Request, sort: str | None = None):
    PARAMS = dict(
        interval="15m", direction="BOTH",
        target_pct=1.5, stop_pct=0.7,
        window_value=8.0, window_unit="Hours",
        lookback_years=2.0, max_tt_bars=20, min_support=20,
        delta_assumed=0.25, theta_per_day_pct=0.20,
        atrz_gate=-0.5, slope_gate_pct=-0.01,
        use_regime=1, regime_trend_only=0, vix_z_max=3.0,
        slippage_bps=7.0, vega_scale=0.03,
        scan_min_hit=55.0,     # GUI-style thresholds
        scan_max_dd=1.0
    )
    rows = []
    for t in TOP150:
        r = compute_scan_for_ticker(t, PARAMS) or {}
        if not r:
            continue
        if r.get("hit_pct", 0) >= PARAMS["scan_min_hit"] and r.get("avg_dd_pct", 999) <= PARAMS["scan_max_dd"]:
            rows.append(r)

    rows.sort(key=lambda x: (x["avg_roi_pct"], x["hit_pct"], x["support"], x["stability"]), reverse=True)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "rows": _sort_rows(rows, sort),
            "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "note": f"TOP150 parity run • kept {len(rows)}",
        },
    )


def _coerce_scan_params(form: dict) -> dict:
    """Coerce scanner form fields into typed params (no silent hard-coding)."""
    def F(k, cast=float, default=None):
        v = form.get(k, None)
        if v in (None, ""): 
            return default
        try:
            return cast(v)
        except Exception:
            return default

    return {
        "scan_type": (form.get("scan_type") or "scan150"),
        "ticker": (form.get("ticker") or "").strip().upper(),
        "interval": (form.get("interval") or "15m").strip(),
        "direction": (form.get("direction") or "BOTH").strip().upper(),
        "target_pct": F("target_pct", float, 1.0),
        "stop_pct": F("stop_pct", float, 0.5),
        "window_value": F("window_value", float, 4.0),
        "window_unit": (form.get("window_unit") or "Hours").strip(),
        "lookback_years": F("lookback_years", float, 2.0),
        "max_tt_bars": F("max_tt_bars", int, 12),
        "min_support": F("min_support", int, 20),
        "delta_assumed": F("delta_assumed", float, 0.40),
        "theta_per_day_pct": F("theta_per_day_pct", float, 0.20),
        "atrz_gate": F("atrz_gate", float, 0.10),
        "slope_gate_pct": F("slope_gate_pct", float, 0.02),
        "use_regime": F("use_regime", int, 0),
        "regime_trend_only": F("regime_trend_only", int, 0),
        "vix_z_max": F("vix_z_max", float, 3.0),
        "slippage_bps": F("slippage_bps", float, 7.0),
        "vega_scale": F("vega_scale", float, 0.03),
        "scan_min_hit": F("scan_min_hit", float, 50.0),
        "scan_max_dd": F("scan_max_dd", float, 50.0),
        "email_checkbox": (form.get("email_checkbox") or ""),
    }

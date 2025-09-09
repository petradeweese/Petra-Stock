import logging
import sqlite3
from typing import Dict, Any, List, Optional

from utils import now_et
from scanner import compute_scan_for_ticker

logger = logging.getLogger(__name__)

DB_PATH = "patternfinder.db"

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
        scan_type TEXT,
        snapshot_at TEXT,
        roi_snapshot REAL,
        hit_snapshot REAL,
        dd_snapshot REAL,
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
    # Forward test results
    """
    CREATE TABLE IF NOT EXISTS forward_tests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fav_id INTEGER NOT NULL,
        ran_at TEXT,
        avg_roi_pct REAL,
        hit_pct REAL,
        avg_dd_pct REAL,
        FOREIGN KEY(fav_id) REFERENCES favorites(id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forward_tests_fav ON forward_tests(fav_id);",
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


def init_db():
    conn = None
    try:
        # Allow connections to be accessed across threads.
        # FastAPI runs dependency context managers in a threadpool which may
        # differ from the thread handling the request. SQLite by default
        # restricts connections to the creating thread, so we disable that
        # safeguard here. Each request still gets its own connection, so
        # concurrent access is safe for this application.
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.executescript(stmt)
        # --- Migrations for snapshot columns ---
        cur.execute("PRAGMA table_info(favorites)")
        cols = {r[1] for r in cur.fetchall()}
        migrations = {
            "scan_type": "TEXT",
            "snapshot_at": "TEXT",
            "roi_snapshot": "REAL",
            "hit_snapshot": "REAL",
            "dd_snapshot": "REAL",
        }
        for col, typ in migrations.items():
            if col not in cols:
                try:
                    cur.execute(f"ALTER TABLE favorites ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
        conn.commit()

        # Backfill snapshot metrics for existing favorites only once
        cur.execute(
            "SELECT id, ticker, direction, interval, target_pct, stop_pct, window_value, window_unit, lookback_years, max_tt_bars, min_support, delta, theta_day, atrz, slope, use_regime, trend_only, vix_z_max, slippage_bps, vega_scale, scan_type FROM favorites WHERE roi_snapshot IS NULL"
        )
        rows = cur.fetchall()
        for r in rows:
            params = {
                "interval": r["interval"],
                "direction": r["direction"],
                "target_pct": r["target_pct"],
                "stop_pct": r["stop_pct"],
                "window_value": r["window_value"],
                "window_unit": r["window_unit"],
                "lookback_years": r["lookback_years"],
                "max_tt_bars": r["max_tt_bars"],
                "min_support": r["min_support"],
                "delta_assumed": r["delta"],
                "theta_per_day_pct": r["theta_day"],
                "atrz_gate": r["atrz"],
                "slope_gate_pct": r["slope"],
                "use_regime": r["use_regime"],
                "regime_trend_only": r["trend_only"],
                "vix_z_max": r["vix_z_max"],
                "slippage_bps": r["slippage_bps"],
                "vega_scale": r["vega_scale"],
                "scan_min_hit": 0.0,
                "scan_max_dd": 100.0,
            }
            row = compute_scan_for_ticker(r["ticker"], params)
            if row:
                cur.execute(
                    "UPDATE favorites SET roi_snapshot=?, hit_snapshot=?, dd_snapshot=?, snapshot_at=?, scan_type=? WHERE id=?",
                    (
                        row.get("avg_roi_pct"),
                        row.get("hit_pct"),
                        row.get("avg_dd_pct"),
                        now_et().isoformat(),
                        r["scan_type"],
                        r["id"],
                    ),
                )
        conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to initialize database")
        raise
    finally:
        if conn is not None:
            conn.close()


def get_db():
    # Create a new connection for each request and allow it to be used from
    # the request-handling thread even though the connection is created in the
    # dependency threadpool.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        logger.exception("Database operation failed")
        raise
    finally:
        conn.close()


def get_settings(db: sqlite3.Cursor) -> sqlite3.Row:
    db.execute("SELECT * FROM settings WHERE id=1")
    return db.fetchone()


def set_last_run(boundary_iso: str, db: sqlite3.Cursor):
    db.execute(
        "UPDATE settings SET last_boundary=?, last_run_at=? WHERE id=1",
        (boundary_iso, now_et().isoformat()),
    )
    db.connection.commit()

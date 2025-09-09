import logging
import sqlite3
from typing import Dict, Any, List, Optional

from utils import now_et

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
        ref_avg_dd REAL,
        roi_snapshot REAL,
        hit_pct_snapshot REAL,
        dd_pct_snapshot REAL,
        rule_snapshot TEXT,
        settings_json_snapshot TEXT,
        snapshot_at TEXT
    );
    """,
    # Forward test tracking
    """
    CREATE TABLE IF NOT EXISTS forward_tests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fav_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        direction TEXT NOT NULL,
        interval TEXT NOT NULL,
        rule TEXT,
        entry_price REAL NOT NULL,
        target_pct REAL NOT NULL,
        stop_pct REAL NOT NULL,
        window_minutes INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        roi_forward REAL,
        hit_forward REAL,
        dd_forward REAL,
        last_run_at TEXT,
        next_run_at TEXT,
        runs_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT,
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
        hit_count INTEGER DEFAULT 0,
        settings_json TEXT
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
    # Scan tasks for cross-worker communication
    """
    CREATE TABLE IF NOT EXISTS scan_tasks (
        id TEXT PRIMARY KEY,
        total INTEGER,
        done INTEGER,
        percent REAL,
        state TEXT,
        message TEXT,
        ctx TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """,
]


def migrate_forward_tests(conn: sqlite3.Connection) -> None:
    """Ensure the forward_tests table has the expected columns."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='forward_tests'"
    )
    if cur.fetchone() is None:
        return
    cur.execute("PRAGMA table_info(forward_tests)")
    cols = {row[1] for row in cur.fetchall()}

    def rename(old: str, new: str):
        if old in cols and new not in cols:
            cur.execute(f"ALTER TABLE forward_tests RENAME COLUMN {old} TO {new}")
            cols.remove(old)
            cols.add(new)

    def add(col: str, ddl: str, backfill: Optional[str] = None):
        if col not in cols:
            cur.execute(f"ALTER TABLE forward_tests ADD COLUMN {col} {ddl}")
            if backfill:
                cur.execute(backfill)

    rename("roi", "roi_forward")
    rename("hit_pct", "hit_forward")
    rename("dd_pct", "dd_forward")
    rename("last_run", "last_run_at")

    add("status", "TEXT NOT NULL DEFAULT 'queued'", "UPDATE forward_tests SET status='queued' WHERE status IS NULL OR status='pending'")
    cur.execute("UPDATE forward_tests SET status='ok' WHERE status='done'")
    add("next_run_at", "TEXT")
    add("runs_count", "INTEGER NOT NULL DEFAULT 0")
    add("notes", "TEXT")
    add("roi_forward", "REAL", "UPDATE forward_tests SET roi_forward=0.0 WHERE roi_forward IS NULL")
    add("hit_forward", "REAL")
    add("dd_forward", "REAL")
    add("last_run_at", "TEXT")
    add("rule", "TEXT")
    add("ticker", "TEXT")
    add("direction", "TEXT")
    add("interval", "TEXT")
    add("created_at", "TEXT", "UPDATE forward_tests SET created_at=entry_ts WHERE created_at IS NULL")
    add("updated_at", "TEXT", "UPDATE forward_tests SET updated_at=created_at WHERE updated_at IS NULL")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forward_tests_fav ON forward_tests(fav_id)")
    conn.commit()


def migrate_favorites(conn: sqlite3.Connection) -> None:
    """Ensure favorites table has snapshot columns."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='favorites'"
    )
    if cur.fetchone() is None:
        return
    cur.execute("PRAGMA table_info(favorites)")
    cols = {row[1] for row in cur.fetchall()}

    def add(col: str, ddl: str):
        if col not in cols:
            cur.execute(f"ALTER TABLE favorites ADD COLUMN {col} {ddl}")

    add("roi_snapshot", "REAL")
    add("hit_pct_snapshot", "REAL")
    add("dd_pct_snapshot", "REAL")
    add("rule_snapshot", "TEXT")
    add("settings_json_snapshot", "TEXT")
    add("snapshot_at", "TEXT")
    conn.commit()


def migrate_runs(conn: sqlite3.Connection) -> None:
    """Ensure runs table has settings_json column."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    if cur.fetchone() is None:
        return
    cur.execute("PRAGMA table_info(runs)")
    cols = {row[1] for row in cur.fetchall()}
    if "settings_json" not in cols:
        cur.execute("ALTER TABLE runs ADD COLUMN settings_json TEXT")
    conn.commit()


def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.executescript(stmt)
        migrate_forward_tests(conn)
        migrate_favorites(conn)
        migrate_runs(conn)
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

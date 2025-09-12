import logging
import sqlite3
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import settings

try:  # pragma: no cover - prefer real Alembic if available
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
except ImportError:  # pragma: no cover - fallback stub
    from alembic_stub import command as alembic_command  # type: ignore[no-redef]
    from alembic_stub.config import Config as AlembicConfig  # type: ignore[assignment,no-redef]

from utils import now_et

logger = logging.getLogger(__name__)

DB_PATH = "patternfinder.db"
# Connection URL used by SQLAlchemy.  Defaults to the local SQLite file but can
# be overridden with e.g. ``postgresql+psycopg2://user:pass@host/db`` for
# production deployments.  Using SQLAlchemy here keeps the code database
# agnostic between SQLite (tests) and Postgres (prod).

_ENGINE: Optional[Engine] = None


def _get_database_url() -> str:
    env_url = settings.database_url
    return env_url or f"sqlite:///{DB_PATH}"


def get_engine() -> Engine:
    """Return a module-level SQLAlchemy engine, recreating if DB path changes."""
    global _ENGINE
    url = _get_database_url()
    if _ENGINE is None or str(_ENGINE.url) != url:
        _ENGINE = create_engine(url, future=True)
    return _ENGINE


def get_schema_status() -> dict:
    conn = get_engine().raw_connection()
    try:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        idx_list = conn.execute("PRAGMA index_list('bars_15m')").fetchall()
        has_idx = any(r[1] == "idx_bars_symbol_ts" for r in idx_list)
        return {
            "journal_mode": journal,
            "synchronous": synchronous,
            "has_index": bool(has_idx),
        }
    finally:
        conn.close()


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
      (
        id,
        smtp_user,
        smtp_pass,
        recipients,
        scheduler_enabled,
        throttle_minutes,
        last_boundary,
        last_run_at
      )
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
    # 15-minute bars for market data
    """
    CREATE TABLE IF NOT EXISTS bars_15m (
        symbol TEXT NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume BIGINT,
        PRIMARY KEY(symbol, ts)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars_15m(symbol, ts);",
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


def run_migrations() -> None:
    """Apply database migrations using Alembic."""
    cfg = AlembicConfig(str(Path(__file__).with_name("alembic.ini")))
    cfg.set_main_option("sqlalchemy.url", _get_database_url())
    alembic_command.upgrade(cfg, "head")


def init_db():
    run_migrations()


def get_db():
    """Yield a database cursor with rows as dictionaries."""
    url = _get_database_url()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    else:  # pragma: no cover - used only in prod with non-SQLite DBs
        conn = get_engine().raw_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
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

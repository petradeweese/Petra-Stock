import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

try:  # pragma: no cover - prefer real Alembic if available
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
except ImportError:  # pragma: no cover - fallback stub
    from alembic_stub import command as alembic_command  # type: ignore[no-redef]
    from alembic_stub.config import Config as AlembicConfig  # type: ignore[assignment,no-redef]

from utils import now_et

logger = logging.getLogger(__name__)

# Derive an absolute path for the default SQLite database so we don't depend on
# the process working directory (e.g. when run via systemd).
DB_PATH = str(Path(__file__).resolve().with_name("patternfinder.db"))
# Connection URL used by SQLAlchemy.  Defaults to the local SQLite file but can
# be overridden with e.g. ``postgresql+psycopg2://user:pass@host/db`` for
# production deployments.  Using SQLAlchemy here keeps the code database
# agnostic between SQLite (tests) and Postgres (prod).

_ENV_DATABASE_URL = os.getenv("DATABASE_URL", "")

_ENGINE: Optional[Engine] = None


def _get_database_url() -> str:
    return _ENV_DATABASE_URL or f"sqlite:///{DB_PATH}"


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
        idx_list = conn.execute("PRAGMA index_list('bars')").fetchall()
        has_idx = any(r[1] == "idx_bars_sym_int_ts" for r in idx_list)
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
        smtp_host TEXT,
        smtp_port INTEGER DEFAULT 587,
        smtp_user TEXT,
        smtp_pass TEXT,
        mail_from TEXT,
        recipients TEXT,
        scanner_recipients TEXT,
        scheduler_enabled INTEGER DEFAULT 0,
        throttle_minutes INTEGER DEFAULT 60,
        last_boundary TEXT,
        last_run_at TEXT,
        greeks_profile_json TEXT DEFAULT '{}'
    );
    """,
    """
    INSERT OR IGNORE INTO settings
      (
        id,
        smtp_host,
        smtp_port,
        smtp_user,
        smtp_pass,
        mail_from,
        recipients,
        scanner_recipients,
        scheduler_enabled,
        throttle_minutes,
        last_boundary,
        last_run_at
      )
    VALUES
      (1, '', 587, '', '', '', '', '', 0, 60, '', '');
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
        support_snapshot TEXT,
        rule_snapshot TEXT,
        settings_json_snapshot TEXT,
        snapshot_at TEXT,
        greeks_override_json TEXT
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
        version INTEGER NOT NULL DEFAULT 1,
        entry_price REAL NOT NULL,
        target_pct REAL NOT NULL,
        stop_pct REAL NOT NULL,
        window_minutes INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        roi_forward REAL,
        hit_forward REAL,
        dd_forward REAL,
        roi_1 REAL,
        roi_3 REAL,
        roi_5 REAL,
        roi_expiry REAL,
        mae REAL,
        mfe REAL,
        time_to_hit REAL,
        time_to_stop REAL,
        option_expiry TEXT,
        option_strike REAL,
        option_delta REAL,
        option_roi_proxy REAL,
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
    # Guardrail skip log
    """
    CREATE TABLE IF NOT EXISTS guardrail_skips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    CREATE TABLE IF NOT EXISTS bars (
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume BIGINT,
        PRIMARY KEY(symbol, interval, ts)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_bars_sym_int_ts ON bars(symbol, interval, ts);",
    "CREATE INDEX IF NOT EXISTS bars_symbol_ts ON bars(symbol, ts);",
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
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT
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
    # Create a new connection for each request and allow it to be used from
    # the request-handling thread even though the connection is created in the
    # dependency threadpool.  When running against SQLite (the common test
    # setup) we use the sqlite3 module directly so row_factory works as
    # expected; otherwise we fall back to SQLAlchemy's engine.
    if _ENV_DATABASE_URL:
        conn = get_engine().raw_connection()
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
    if hasattr(conn, "row_factory"):
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


def _ensure_scanner_column(db: sqlite3.Cursor) -> None:
    db.execute("PRAGMA table_info(settings)")
    cols = [r[1] for r in db.fetchall()]
    if "scanner_recipients" not in cols:
        db.execute("ALTER TABLE settings ADD COLUMN scanner_recipients TEXT DEFAULT ''")
        db.connection.commit()
    if "greeks_profile_json" not in cols:
        db.execute(
            "ALTER TABLE settings ADD COLUMN greeks_profile_json TEXT DEFAULT '{}'"
        )
        db.connection.commit()
    if "smtp_host" not in cols:
        db.execute("ALTER TABLE settings ADD COLUMN smtp_host TEXT DEFAULT ''")
        db.connection.commit()
    if "smtp_port" not in cols:
        db.execute("ALTER TABLE settings ADD COLUMN smtp_port INTEGER DEFAULT 587")
        db.connection.commit()
    if "mail_from" not in cols:
        db.execute("ALTER TABLE settings ADD COLUMN mail_from TEXT DEFAULT ''")
        db.connection.commit()


def row_to_dict(
    row: Mapping[str, Any] | sqlite3.Row | Sequence[Any] | None,
    cursor: sqlite3.Cursor | None = None,
) -> dict[str, Any]:
    """Convert a database row into a plain dict.

    The DB API used by the application varies between environments.  When the
    underlying driver returns simple tuples (e.g. via ``raw_connection`` in
    SQLAlchemy), ``row`` will not expose ``keys`` like ``sqlite3.Row`` does.  In
    those cases we can derive the column names from ``cursor.description``.
    """

    if row is None:
        return {}
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}  # type: ignore[index]
    if cursor is not None and getattr(cursor, "description", None):
        cols = [c[0] for c in cursor.description]
        return {col: row[idx] for idx, col in enumerate(cols)}
    try:
        return dict(row)
    except Exception:
        return {str(i): v for i, v in enumerate(row)}


def get_settings(db: sqlite3.Cursor) -> dict:
    _ensure_scanner_column(db)
    db.execute("SELECT * FROM settings WHERE id=1")
    row = db.fetchone()
    return row_to_dict(row, db)


def set_last_run(boundary_iso: str, db: sqlite3.Cursor):
    db.execute(
        "UPDATE settings SET last_boundary=?, last_run_at=? WHERE id=1",
        (boundary_iso, now_et().isoformat()),
    )
    db.connection.commit()

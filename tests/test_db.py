import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db


def test_init_db_creates_settings(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    gen = db.get_db()
    cur = next(gen)
    settings = db.get_settings(cur)
    assert settings["id"] == 1
    gen.close()


def test_forward_tests_table_exists(tmp_path):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(forward_tests)")
    cols = {r[1] for r in cur.fetchall()}
    conn.close()
    assert {"status", "roi_1", "roi_expiry", "option_roi_proxy"}.issubset(cols)


def test_forward_tests_migration_adds_exit_columns(tmp_path):
    db.DB_PATH = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db.DB_PATH)
    conn.executescript(
        """
        CREATE TABLE forward_tests (
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
            last_run_at TEXT,
            next_run_at TEXT,
            runs_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db.init_db()

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(forward_tests)")
    columns = {row[1] for row in cur.fetchall()}
    conn.close()

    expected = {
        "exit_reason",
        "bars_to_exit",
        "max_drawdown_pct",
        "max_runup_pct",
        "r_multiple",
        "option_roi_proxy",
    }
    assert expected.issubset(columns)


def test_favorites_migration_adds_hit_pct_snapshot(tmp_path):
    db.DB_PATH = str(tmp_path / "favorites_legacy.db")
    conn = sqlite3.connect(db.DB_PATH)
    conn.executescript(
        """
        CREATE TABLE favorites (
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
            roi_snapshot TEXT,
            dd_pct_snapshot REAL,
            support_snapshot TEXT,
            rule_snapshot TEXT,
            settings_json_snapshot TEXT,
            snapshot_at TEXT,
            greeks_override_json TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db.init_db()

    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(favorites)")
    columns = {row[1] for row in cur.fetchall()}
    conn.close()

    assert "hit_pct_snapshot" in columns

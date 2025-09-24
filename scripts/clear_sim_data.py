"""Remove synthetic artifacts produced during DEBUG_SIMULATION runs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from db import DB_PATH

ALERTS_SQLITE = Path(__file__).resolve().parent.parent / "alerts.sqlite"
EXPORT_FILES = [
    Path(__file__).resolve().parent.parent / "sim_export.csv",
    Path(__file__).resolve().parent.parent / "sim_export.json",
]


def _clear_forward_runs() -> int:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM forward_runs WHERE simulated=1")
        deleted = cur.rowcount or 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def _clear_alert_dedupe() -> int:
    if not ALERTS_SQLITE.exists():
        return 0
    conn = sqlite3.connect(ALERTS_SQLITE, check_same_thread=False)
    try:
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM sent_alerts WHERE simulated=1")
        except sqlite3.OperationalError:
            return 0
        deleted = cur.rowcount or 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def _remove_exports() -> list[Path]:
    removed: list[Path] = []
    for path in EXPORT_FILES:
        try:
            if path.exists():
                path.unlink()
                removed.append(path)
        except OSError:
            continue
    return removed


def main() -> None:
    cleared_forward = _clear_forward_runs()
    cleared_alerts = _clear_alert_dedupe()
    removed = _remove_exports()
    print(
        f"Cleared {cleared_forward} forward runs, {cleared_alerts} alert dedupe records, "
        f"removed {len(removed)} export files"
    )


if __name__ == "__main__":
    main()

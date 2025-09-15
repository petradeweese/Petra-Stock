import os
import sys
from pathlib import Path

os.environ.setdefault("SCAN_EXECUTOR_MODE", "thread")

# Ensure the project root is on the import path when running ``pytest`` directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import DB_PATH, run_migrations


def pytest_sessionstart(session):
    """Ensure database schema exists before tests run."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    run_migrations()

import os
import sys
from pathlib import Path

os.environ.setdefault("SCAN_EXECUTOR_MODE", "thread")
os.environ.setdefault("SCHWAB_CLIENT_ID", "test-client")
os.environ.setdefault("SCHWAB_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SCHWAB_REDIRECT_URI", "https://example.com/callback")
os.environ.setdefault("SCHWAB_ACCOUNT_ID", "00000000")
os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-refresh")

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

import os

from db import DB_PATH, run_migrations


def pytest_sessionstart(session):
    """Ensure database schema exists before tests run."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    run_migrations()

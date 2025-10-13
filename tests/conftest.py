import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SCAN_EXECUTOR_MODE", "thread")
os.environ.setdefault("SCHWAB_CLIENT_ID", "test-client")
os.environ.setdefault("SCHWAB_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SCHWAB_REDIRECT_URI", "https://example.com/callback")
os.environ.setdefault("SCHWAB_ACCOUNT_ID", "00000000")
os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-refresh")
os.environ.setdefault("SCHWAB_REFRESH_BACKOFF_SECONDS", "1")

# Ensure the project root is on the import path when running ``pytest`` directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import http_client, schwab_client
from services.schwab_client import SchwabAuthError

from db import DB_PATH, run_migrations


def pytest_sessionstart(session):
    """Ensure database schema exists before tests run."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    run_migrations()


@pytest.fixture(autouse=True)
def _disable_real_schwab_requests():
    """Prevent tests from making live Schwab API calls while allowing overrides."""

    original_request = http_client.request

    async def _guarded_request(method, url, **kwargs):
        current = http_client.request
        if current is not _guarded_request:
            return await current(method, url, **kwargs)
        if "schwabapi.com" in url:
            raise SchwabAuthError("schwab requests disabled in tests", status_code=400)
        return await original_request(method, url, **kwargs)

    http_client.request = _guarded_request
    try:
        yield
    finally:
        http_client.request = original_request
        schwab_client.disable(reason="tests_reset", ttl=0)


@pytest.fixture(autouse=True)
def _ensure_config_module():
    import config
    sys.modules["config"] = config
    for module in list(sys.modules.values()):
        if module is None:
            continue
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        if getattr(cfg, "__file__", None) == getattr(config, "__file__", None) and cfg is not config:
            setattr(module, "config", config)
    yield

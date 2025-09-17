import os
from dataclasses import dataclass


_FALSEY = {"0", "false", "", "no", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    """Return ``True`` if the environment variable is truthy."""

    default_val = "1" if default else "0"
    return os.getenv(name, default_val).strip().lower() not in _FALSEY


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", ""}


@dataclass
class Settings:
    run_migrations: bool = _bool("RUN_MIGRATIONS", "true")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///patternfinder.db")
    http_max_concurrency: int = int(os.getenv("HTTP_MAX_CONCURRENCY", "1"))
    job_timeout: int = int(os.getenv("JOB_TIMEOUT", "60"))
    metrics_enabled: bool = _bool("METRICS_ENABLED", "false")
    clamp_market_closed: bool = _bool("CLAMP_MARKET_CLOSED", "true")
    backfill_chunk_days: int = int(os.getenv("BACKFILL_CHUNK_DAYS", "1"))
    fetch_retry_max: int = int(os.getenv("FETCH_RETRY_MAX", "4"))
    fetch_retry_base_ms: int = int(os.getenv("FETCH_RETRY_BASE_MS", "300"))
    fetch_retry_cap_ms: int = int(os.getenv("FETCH_RETRY_CAP_MS", "5000"))

    # Scanner feature flags; all additive and safe to tweak at runtime.
    scan_max_concurrency: int = int(os.getenv("SCAN_MAX_CONCURRENCY", "8"))
    scan_rps: float = float(os.getenv("SCAN_RPS", "0"))
    scan_http_timeout: float = float(os.getenv("SCAN_HTTP_TIMEOUT", "10"))
    scan_status_poll_ms: int = int(os.getenv("SCAN_STATUS_POLL_MS", "2000"))
    scan_progress_flush_items: int = int(os.getenv("SCAN_PROGRESS_FLUSH_ITEMS", "10"))
    scan_status_flush_ms: int = int(os.getenv("SCAN_STATUS_FLUSH_MS", "500"))
    scan_fetch_concurrency: int = int(os.getenv("SCAN_FETCH_CONCURRENCY", "8"))
    scan_coverage_batch_size: int = int(os.getenv("SCAN_COVERAGE_BATCH_SIZE", "200"))
    scan_symbols_per_task: int = int(os.getenv("SCAN_SYMBOLS_PER_TASK", "1"))
    scan_minimal_near_now: bool = _bool("SCAN_MINIMAL_NEAR_NOW", "1")


settings = Settings()

USE_SCHWAB_PRIMARY = env_bool("USE_SCHWAB_PRIMARY", True)
DATA_PROVIDERS_PRIMARY: list[str] = []


def _apply_provider_priority() -> None:
    DATA_PROVIDERS_PRIMARY.clear()
    if USE_SCHWAB_PRIMARY:
        DATA_PROVIDERS_PRIMARY.extend(["schwab", "yahoo"])
    else:
        DATA_PROVIDERS_PRIMARY.extend(["yahoo", "schwab"])


_apply_provider_priority()


def set_use_schwab_primary(value: bool) -> None:
    """Update provider priority at runtime based on ``value``."""

    global USE_SCHWAB_PRIMARY
    USE_SCHWAB_PRIMARY = bool(value)
    os.environ["USE_SCHWAB_PRIMARY"] = "1" if USE_SCHWAB_PRIMARY else "0"
    _apply_provider_priority()


SCHWAB_ENABLED = env_bool("SCHWAB_ENABLED", True)
SCHWAB_RPS = int(os.getenv("SCHWAB_RPS", "4"))
ADJUST_BARS = env_bool("ADJUST_BARS", True)
_SESSION_RAW = os.getenv("SESSION", "RTH").strip().upper()
SESSION = _SESSION_RAW if _SESSION_RAW in {"RTH", "ETH"} else "RTH"

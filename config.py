import os
from dataclasses import dataclass


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", ""}


@dataclass
class Settings:
    run_migrations: bool = _bool("RUN_MIGRATIONS", "true")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///patternfinder.db")
    http_max_concurrency: int = int(os.getenv("HTTP_MAX_CONCURRENCY", "10"))
    job_timeout: int = int(os.getenv("JOB_TIMEOUT", "30"))
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


settings = Settings()

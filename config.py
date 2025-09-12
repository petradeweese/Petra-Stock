import os
from dataclasses import dataclass


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", ""}


@dataclass
class Settings:
    run_migrations: bool = _bool("RUN_MIGRATIONS", "true")
    # Default to an empty string so tests can override ``DB_PATH`` without the
    # environment variable taking precedence.
    database_url: str = os.getenv("DATABASE_URL", "")
    http_max_concurrency: int = int(os.getenv("HTTP_MAX_CONCURRENCY", "10"))
    job_timeout: int = int(os.getenv("JOB_TIMEOUT", "30"))
    metrics_enabled: bool = _bool("METRICS_ENABLED", "false")


settings = Settings()

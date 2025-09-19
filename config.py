import os
from dataclasses import dataclass
from typing import Tuple


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", ""}


def _csv(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    if not raw:
        return ()
    parts = [segment.strip() for segment in raw.split(",")]
    return tuple(part for part in parts if part)


def _choice(name: str, default: str, *, allowed: Tuple[str, ...] | None = None) -> str:
    raw = os.getenv(name, default)
    value = raw if raw is not None else default
    value = str(value).strip() or default
    if allowed:
        lowered = value.lower()
        for option in allowed:
            if lowered == option.lower():
                return option
        return default
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip() or default)
    except (TypeError, ValueError):
        return float(default)


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

    # Favorites alert delivery
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_from_number: str = os.getenv("TWILIO_FROM_NUMBER", "")
    alert_sms_to: Tuple[str, ...] = _csv("ALERT_SMS_TO")
    alert_channel: str = _choice("ALERT_CHANNEL", "Email", allowed=("Email", "MMS"))
    alert_outcomes: str = _choice("ALERT_OUTCOMES", "hit", allowed=("hit", "all"))
    forward_recency_mode: str = _choice(
        "FORWARD_RECENCY_MODE", "off", allowed=("off", "exp")
    )
    forward_recency_halflife_days: float = _float(
        "FORWARD_RECENCY_HALFLIFE_DAYS", 30.0
    )


settings = Settings()
settings.alert_channel = settings.alert_channel or "Email"
settings.alert_outcomes = (settings.alert_outcomes or "hit").lower()
setattr(settings, "ALERT_CHANNEL", settings.alert_channel)
setattr(settings, "ALERT_OUTCOMES", settings.alert_outcomes)
mode_value = (getattr(settings, "forward_recency_mode", "off") or "off").lower()
if mode_value not in {"off", "exp"}:
    mode_value = "off"
settings.forward_recency_mode = mode_value
setattr(settings, "FORWARD_RECENCY_MODE", mode_value)
try:
    half_life_value = float(getattr(settings, "forward_recency_halflife_days", 30.0))
except (TypeError, ValueError):
    half_life_value = 30.0
if half_life_value <= 0:
    half_life_value = 30.0
settings.forward_recency_halflife_days = half_life_value
setattr(settings, "FORWARD_RECENCY_HALFLIFE_DAYS", half_life_value)

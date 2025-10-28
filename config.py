import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


DEFAULT_DB_PATH = "/home/ubuntu/Petra-Stock/patternfinder.db"
DEFAULT_DATABASE_URL = f"sqlite:////home/ubuntu/Petra-Stock/patternfinder.db"
DEFAULT_SCHWAB_TOKENS_PATH = "/etc/petra/secure/schwab_tokens.json"


logger = logging.getLogger(__name__)


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


def _env_files() -> Iterable[Path]:
    """Yield candidate environment files in priority order."""

    paths: list[str] = []
    override = os.getenv("PETRA_ENV_FILE")
    if override:
        paths.append(override)
    paths.append("/etc/petra/petra.env")

    seen: set[Path] = set()
    for raw in paths:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _load_environment_from_file(path: Path) -> None:
    """Load KEY=VALUE pairs from *path* into ``os.environ`` if missing."""

    try:
        text = path.read_text()
    except FileNotFoundError:
        return
    except OSError:
        # Ignore unreadable files so a missing optional file does not abort import.
        return

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        lexer = shlex.shlex(raw_line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError:
            # Skip malformed lines.
            continue

        for token in tokens:
            if token == "export" or "=" not in token:
                continue
            name, value = token.split("=", 1)
            name = name.strip()
            if not name:
                continue
            value = value.strip()
            current = os.getenv(name)
            if current is None or (isinstance(current, str) and not current.strip()):
                os.environ[name] = value


def _load_environment() -> None:
    for candidate in _env_files():
        _load_environment_from_file(candidate)


_load_environment()


def _load_refresh_token_from_file(path: str) -> str | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    try:
        text = candidate.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "config schwab_tokens_unreadable path=%s",
            candidate,
        )
        return None

    try:
        payload = json.loads(text)
    except ValueError:
        logger.warning(
            "config schwab_tokens_invalid_json path=%s",
            candidate,
        )
        return None

    token = payload.get("refresh_token")
    if isinstance(token, str):
        token = token.strip()
    if not token:
        return None
    return token


@dataclass
class Settings:
    run_migrations: bool = _bool("RUN_MIGRATIONS", "true")
    database_url: str = os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    http_max_concurrency: int = int(os.getenv("HTTP_MAX_CONCURRENCY", "1"))
    job_timeout: int = int(os.getenv("JOB_TIMEOUT", "60"))
    metrics_enabled: bool = _bool("METRICS_ENABLED", "false")
    clamp_market_closed: bool = _bool("CLAMP_MARKET_CLOSED", "true")
    backfill_chunk_days: int = int(os.getenv("BACKFILL_CHUNK_DAYS", "1"))
    fetch_retry_max: int = int(os.getenv("FETCH_RETRY_MAX", "4"))
    fetch_retry_base_ms: int = int(os.getenv("FETCH_RETRY_BASE_MS", "300"))
    fetch_retry_cap_ms: int = int(os.getenv("FETCH_RETRY_CAP_MS", "5000"))
    forecast_allow_network: bool = _bool("FORECAST_ALLOW_NETWORK", "1")

    # Market data provider configuration
    data_provider: str = _choice(
        "DATA_PROVIDER",
        os.getenv("PF_DATA_PROVIDER", "schwab"),
        allowed=("schwab", "yahoo", "db"),
    )
    schwab_client_id: str = os.getenv("SCHWAB_CLIENT_ID", "")
    schwab_client_secret: str = os.getenv("SCHWAB_CLIENT_SECRET", "")
    schwab_redirect_uri: str = os.getenv("SCHWAB_REDIRECT_URI", "")
    schwab_account_id: str = os.getenv("SCHWAB_ACCOUNT_ID", "")
    schwab_refresh_token: str = os.getenv("SCHWAB_REFRESH_TOKEN", "")
    schwab_refresh_backoff_seconds: int = int(
        os.getenv("SCHWAB_REFRESH_BACKOFF_SECONDS", "180")
    )
    schwab_token_path: str = (
        os.getenv("SCHWAB_TOKENS_PATH")
        or os.getenv("SCHWAB_TOKEN_PATH")
        or DEFAULT_SCHWAB_TOKENS_PATH
    )

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
    twilio_verify_service_sid: str = os.getenv("TWILIO_VERIFY_SERVICE_SID", "")
    alert_sms_to: Tuple[str, ...] = _csv("ALERT_SMS_TO")
    alert_channel: str = _choice("ALERT_CHANNEL", "Email", allowed=("Email", "MMS", "SMS"))
    alert_outcomes: str = _choice("ALERT_OUTCOMES", "hit", allowed=("hit", "all"))
    forward_recency_mode: str = _choice(
        "FORWARD_RECENCY_MODE", "off", allowed=("off", "exp")
    )
    forward_recency_halflife_days: float = _float(
        "FORWARD_RECENCY_HALFLIFE_DAYS", 30.0
    )


settings = Settings()

_file_refresh_token = _load_refresh_token_from_file(settings.schwab_token_path)
if _file_refresh_token and _file_refresh_token != (
    getattr(settings, "schwab_refresh_token", "") or ""
):
    logger.info(
        "config schwab_refresh_token_loaded path=%s",
        settings.schwab_token_path,
    )
    settings.schwab_refresh_token = _file_refresh_token
    os.environ["SCHWAB_REFRESH_TOKEN"] = _file_refresh_token

setattr(settings, "SCHWAB_REFRESH_TOKEN", settings.schwab_refresh_token)

settings.alert_channel = settings.alert_channel or "Email"
settings.alert_outcomes = (settings.alert_outcomes or "hit").lower()
setattr(settings, "ALERT_CHANNEL", settings.alert_channel)
setattr(settings, "ALERT_OUTCOMES", settings.alert_outcomes)
settings.data_provider = settings.data_provider or "schwab"
setattr(settings, "DATA_PROVIDER", settings.data_provider)
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
setattr(settings, "SCHWAB_TOKEN_PATH", settings.schwab_token_path)
setattr(settings, "SCHWAB_TOKENS_PATH", settings.schwab_token_path)

logger.info(
    "config startup resolved_paths db_path=%s database_url=%s schwab_tokens_path=%s",
    DEFAULT_DB_PATH,
    settings.database_url,
    settings.schwab_token_path,
)

if settings.data_provider.lower() == "schwab":
    required = {
        "SCHWAB_CLIENT_ID": settings.schwab_client_id,
        "SCHWAB_CLIENT_SECRET": settings.schwab_client_secret,
        "SCHWAB_REDIRECT_URI": settings.schwab_redirect_uri,
        "SCHWAB_ACCOUNT_ID": settings.schwab_account_id,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required Schwab configuration: "
            f"{joined}. Set the environment variables or change DATA_PROVIDER."
        )

    if not settings.schwab_refresh_token:
        # Allow the application to continue without a refresh token so the
        # interactive OAuth flow can populate it later.
        setattr(settings, "SCHWAB_REFRESH_TOKEN", "")

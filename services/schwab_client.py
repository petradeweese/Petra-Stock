import asyncio
import base64
import datetime as dt
import grp
import json
import logging
import os
import pwd
import tempfile
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import settings
from services import http_client
from services.oauth_tokens import latest_refresh_token
from prometheus_client import Counter, Histogram  # type: ignore

logger = logging.getLogger(__name__)

TOKEN_URL = os.getenv(
    "SCHWAB_TOKEN_URL", "https://api.schwabapi.com/v1/oauth/token"
)
API_BASE_URL = os.getenv(
    "SCHWAB_API_BASE", "https://api.schwabapi.com/marketdata/v1"
)
PRICE_HISTORY_URL = os.getenv(
    "SCHWAB_PRICE_HISTORY_URL",
    "https://api.schwabapi.com/marketdata/v1/pricehistory",
)
OAUTH_CLIENT_ID_SUFFIX = os.getenv("SCHWAB_OAUTH_CLIENT_ID_SUFFIX", "@AMER.OAUTHAP")

HTTP_400_DISABLE_SECONDS = int(os.getenv("SCHWAB_HTTP_400_DISABLE_SECONDS", "600"))

REFRESH_BACKOFF_SECONDS = max(1, int(settings.schwab_refresh_backoff_seconds or 0))

token_refresh_duration = Histogram(
    "schwab_token_refresh_duration_seconds",
    "Duration of Schwab OAuth token refresh requests",
)
token_refresh_total = Counter(
    "schwab_token_refresh_total",
    "Total Schwab OAuth token refresh attempts",
)
token_refresh_success = Counter(
    "schwab_token_refresh_success_total",
    "Successful Schwab OAuth token refreshes",
)
token_refresh_failure_total = Counter(
    "schwab_token_refresh_failure_total",
    "Total failed Schwab OAuth token refresh attempts",
)
token_refresh_storm_prevented = Counter(
    "schwab_token_refresh_storm_prevented_total",
    "Refresh attempts that were coalesced by singleflight protection",
)
provider_disabled_total = Counter(
    "schwab_provider_disabled_total",
    "Times the Schwab provider was disabled due to authentication errors",
)
_provider_disabled_counters: Dict[str, Counter] = {}
_token_refresh_failure_by_code: Dict[str, Counter] = {}

SENSITIVE_KEYS = {"refresh_token", "access_token", "client_secret", "id_token"}


def _clean_str(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_metric_key(value: str) -> str:
    cleaned = (value or "unknown").strip().lower().replace(" ", "_")
    safe = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return safe or "unknown"


def _increment_refresh_error(status: Optional[int]) -> None:
    token_refresh_failure_total.inc()
    key = _normalize_metric_key(str(status) if status is not None else "unknown")
    counter = _token_refresh_failure_by_code.get(key)
    if counter is None:
        counter = Counter(
            f"schwab_token_refresh_failure_{key}_total",
            "Failed Schwab OAuth token refreshes grouped by status code",
        )
        _token_refresh_failure_by_code[key] = counter
    counter.inc()


def _increment_provider_disabled(reason: str) -> None:
    key = _normalize_metric_key(reason)
    counter = _provider_disabled_counters.get(key)
    if counter is None:
        counter = Counter(
            f"schwab_provider_disabled_{key}_total",
            "Times the Schwab provider was disabled grouped by reason",
        )
        _provider_disabled_counters[key] = counter
    counter.inc()


class SchwabAuthError(RuntimeError):
    """Raised when authentication credentials are missing or invalid."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        error_description: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_description = error_description


class SchwabAPIError(RuntimeError):
    """Raised when the Schwab API returns an unexpected response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class _Token:
    access_token: str
    expires_at: dt.datetime


class SchwabClient:
    """Thin async client for the Charles Schwab market data API."""

    def __init__(self) -> None:
        self._client_id = _clean_str(settings.schwab_client_id)
        settings.schwab_client_id = self._client_id
        self._oauth_client_id = self._format_oauth_client_id(self._client_id)
        self._client_secret = _clean_str(settings.schwab_client_secret)
        settings.schwab_client_secret = self._client_secret
        self._redirect_uri = _clean_str(settings.schwab_redirect_uri)
        settings.schwab_redirect_uri = self._redirect_uri
        self._refresh_token = _clean_str(settings.schwab_refresh_token)
        settings.schwab_refresh_token = self._refresh_token
        self._account_id = _clean_str(settings.schwab_account_id)
        settings.schwab_account_id = self._account_id
        self._token: Optional[_Token] = None
        self._token_lock = asyncio.Lock()
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._refresh_task: Optional[asyncio.Task[_Token]] = None
        self._refresh_task_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_status: Optional[int] = None
        self._disabled_until: float = 0.0
        self._disabled_reason: str = ""
        self._disabled_status: Optional[int] = None
        self._disabled_error: Optional[SchwabAuthError] = None
        missing_core = [
            ("client_id", self._oauth_client_id),
            ("client_secret", self._client_secret),
            ("redirect_uri", self._redirect_uri),
        ]
        missing_labels = [label for label, value in missing_core if not value]
        if missing_labels:
            detail = ", ".join(missing_labels)
            err = SchwabAuthError(
                f"Missing Schwab OAuth configuration ({detail})"
            )
            self.disable(reason="missing_config", status_code=None, error=err)
        elif not self._refresh_token:
            logger.info("schwab_refresh_token_deferred")

    # ------------------------------------------------------------------
    # Token handling
    async def _refresh_access_token(self) -> _Token:
        blocked, reason, status, error = self.disabled_state()
        if blocked:
            remaining = max(0.0, self._disabled_until - time.monotonic())
            logger.warning(
                "schwab_token_refresh_blocked reason=%s status=%s remaining=%.2f",
                reason or "unknown",
                status if status is not None else "unknown",
                remaining,
            )
            raise error or SchwabAuthError(
                "Token refresh currently disabled", status_code=status
            )

        refresh_token = _clean_str(self._current_refresh_token())
        if not all(
            [
                self._client_id,
                self._client_secret,
                self._redirect_uri,
                refresh_token,
            ]
        ):
            raise SchwabAuthError("Missing Schwab OAuth configuration")

        redirect_uri = _clean_str(settings.schwab_redirect_uri) or self._redirect_uri
        if redirect_uri != self._redirect_uri and redirect_uri:
            logger.info(
                "schwab_refresh_redirect_uri_updated old=%s new=%s",
                self._redirect_uri,
                redirect_uri,
            )
            self._redirect_uri = redirect_uri

        auth_mode = self._resolve_auth_mode()
        base_payload = OrderedDict(
            (
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("redirect_uri", self._redirect_uri),
            )
        )
        payload = OrderedDict(
            (key, value)
            for key, value in base_payload.items()
            if value is not None and value != ""
        )

        send_payload: OrderedDict[str, str] = payload
        headers: Dict[str, str]
        request_kwargs: Dict[str, Any]

        if auth_mode == "basic":
            basic_token = base64.b64encode(
                f"{self._oauth_client_id}:{self._client_secret}".encode()
            ).decode()
            headers = {
                "Authorization": f"Basic {basic_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            request_kwargs = {
                "data": dict(send_payload),
                "timeout": settings.scan_http_timeout,
            }
        else:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            body_payload = OrderedDict(send_payload)
            body_payload["client_id"] = self._oauth_client_id
            body_payload["client_secret"] = self._client_secret
            send_payload = OrderedDict(
                (key, value)
                for key, value in body_payload.items()
                if value is not None and value != ""
            )
            form_body = urllib.parse.urlencode(send_payload)
            request_kwargs = {"content": form_body}

        body_keys = ",".join(send_payload.keys()) or ""
        logger.info("schwab_refresh attempt mode=%s body_keys=%s", auth_mode, body_keys)

        safe_payload = OrderedDict(
            (
                key,
                "***REDACTED***"
                if key in {"refresh_token", "client_secret"}
                else value,
            )
            for key, value in send_payload.items()
        )
        logger.debug(
            "schwab_token_refresh_request body=%s",
            urllib.parse.urlencode(safe_payload),
        )

        t0 = time.monotonic()
        resp = await http_client.request("POST", TOKEN_URL, headers=headers, **request_kwargs)
        duration = time.monotonic() - t0
        token_refresh_total.inc()
        if resp.status_code >= 400:
            err = self._handle_refresh_failure(resp, duration)
            _increment_refresh_error(resp.status_code)
            token_refresh_duration.observe(duration)
            raise err

        token_refresh_success.inc()
        token_refresh_duration.observe(duration)

        try:
            data: Dict[str, Any] = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 0) or 0)
        if not token:
            raise SchwabAuthError("Token refresh response missing access_token")

        new_refresh = str(data.get("refresh_token") or "").strip()
        if new_refresh:
            self.set_refresh_token(new_refresh)
            settings.schwab_refresh_token = new_refresh
            setattr(settings, "SCHWAB_REFRESH_TOKEN", new_refresh)

        stored_payload = dict(data)
        if "refresh_token" not in stored_payload:
            stored_payload["refresh_token"] = new_refresh or refresh_token
        stored_payload.setdefault("expires_in", expires_in)
        stored_payload.setdefault("access_token", token)
        stored_payload["obtained_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._write_token_file(stored_payload)

        skewed = max(0, expires_in - 60)
        expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=skewed)
        self._clear_disable()
        logger.info(
            "schwab_token_refreshed expires_in=%s duration=%.2f", expires_in, duration
        )
        return _Token(access_token=token, expires_at=expires_at)

    def _resolve_auth_mode(self) -> str:
        mode = str(getattr(settings, "schwab_auth_mode", "basic") or "basic").lower()
        return "body" if mode == "body" else "basic"

    def _format_failure_body(self, data: Dict[str, Any], resp: Any) -> str:
        raw_text: Optional[str] = None
        try:
            raw_text = resp.text[:512]
        except Exception:  # pragma: no cover - defensive logging
            raw_text = None
        return self._sanitize_error_body(data, raw_text)

    def _sanitize_error_body(
        self, data: Dict[str, Any], raw_text: Optional[str]
    ) -> str:
        if data:
            sanitized: Dict[str, Any] = {}
            for key, value in data.items():
                if key in SENSITIVE_KEYS:
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = value
            try:
                return json.dumps(sanitized)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return str(sanitized)
        if raw_text:
            return raw_text
        return "{}"

    def _write_token_file(self, payload: Dict[str, Any]) -> None:
        path_value = getattr(settings, "schwab_token_path", "") or ""
        if not path_value:
            return
        path = Path(path_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("schwab_token_file_write_failed path=%s error=%s", path, exc)
            return

        tmp_fd: Optional[int] = None
        tmp_name = ""
        try:
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.stem or 'token'}.",
                suffix=".tmp",
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                tmp_fd = None
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.chmod(tmp_name, 0o600)
            self._apply_owner(tmp_name)
            os.replace(tmp_name, path)
            os.chmod(path, 0o600)
            self._apply_owner(path)
        except Exception as exc:
            logger.error("schwab_token_file_write_failed path=%s error=%s", path, exc)
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _apply_owner(self, target: str) -> None:
        try:
            uid = pwd.getpwnam("root").pw_uid
            gid = grp.getgrnam("ubuntu").gr_gid
        except KeyError:
            logger.debug("schwab_token_owner_lookup_failed")
            return
        try:
            os.chown(target, uid, gid)
        except PermissionError:
            logger.debug("schwab_token_owner_permission_denied path=%s", target)
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug(
                "schwab_token_owner_error path=%s error=%s", target, exc
            )

    @staticmethod
    def _format_oauth_client_id(client_id: str) -> str:
        value = (client_id or "").strip()
        suffix = (OAUTH_CLIENT_ID_SUFFIX or "").strip()
        if not value:
            return value
        if suffix and value.endswith(suffix):
            return value
        if suffix and "@" not in value:
            logger.info("schwab_client_id_suffix_applied suffix=%s", suffix)
            return f"{value}{suffix}"
        return value

    def _handle_refresh_failure(self, resp: Any, duration: float) -> SchwabAuthError:
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}

        safe_body = self._format_failure_body(data, resp)
        logger.error(
            "schwab_refresh_failed status=%s body=%s",
            resp.status_code,
            safe_body,
        )

        error_code_raw = data.get("error")
        error_description_raw = data.get("error_description")
        error_code = str(error_code_raw or "").strip()
        error_description = str(error_description_raw or "").strip()
        combined = f"{error_code} {error_description}".strip().lower()
        logger.warning(
            "schwab_token_error status=%s duration=%.2f error=%s description=%s",
            resp.status_code,
            duration,
            error_code or "unknown",
            error_description or "unknown",
        )
        if resp.status_code == 400 and (
            "invalid_grant" in combined or "invalid_refresh_token" in combined
        ):
            logger.warning("schwab_refresh_token_invalid; clearing cached token")
            self._invalidate_refresh_token()
        reason = (error_code or f"status_{resp.status_code}").strip()
        message = f"Token refresh failed ({resp.status_code})"
        if error_code:
            message = f"{message} {error_code}".strip()
        if error_description and error_description not in message:
            message = f"{message}: {error_description}".strip()
        err = SchwabAuthError(
            message,
            status_code=resp.status_code,
            error_code=error_code or None,
            error_description=error_description or None,
        )
        if resp.status_code == 400:
            self.disable(
                reason=reason or "http_400",
                status_code=resp.status_code,
                ttl=float(HTTP_400_DISABLE_SECONDS),
                error=err,
            )
        else:
            self._disabled_error = None
        return err

    def disable(
        self,
        *,
        reason: str,
        status_code: Optional[int] = None,
        ttl: Optional[float] = None,
        error: Optional[SchwabAuthError] = None,
    ) -> None:
        ttl_value = ttl if ttl is not None else float(REFRESH_BACKOFF_SECONDS)
        ttl_value = max(0.0, ttl_value)
        if ttl_value == 0:
            self._clear_disable()
            return
        reason_value = _normalize_metric_key(reason)
        provider_disabled_total.inc()
        _increment_provider_disabled(reason_value)
        self._disabled_until = time.monotonic() + ttl_value
        self._disabled_reason = reason_value
        self._disabled_status = status_code
        self._disabled_error = error
        self._refresh_task = None
        self._refresh_task_loop = None
        self.clear_cached_token()
        logger.info(
            "schwab_provider_disabled reason=%s status=%s ttl=%.2f",
            self._disabled_reason,
            status_code if status_code is not None else "unknown",
            ttl_value,
        )

    def _clear_disable(self) -> None:
        self._disabled_until = 0.0
        self._disabled_reason = ""
        self._disabled_status = None
        self._disabled_error = None

    def disabled_state(
        self,
    ) -> Tuple[bool, Optional[str], Optional[int], Optional[SchwabAuthError]]:
        if self._disabled_until and time.monotonic() < self._disabled_until:
            return True, self._disabled_reason or None, self._disabled_status, self._disabled_error
        if self._disabled_until and time.monotonic() >= self._disabled_until:
            self._clear_disable()
        return False, None, None, None
    def _invalidate_refresh_token(self) -> None:
        self.set_refresh_token("")
        settings.schwab_refresh_token = ""
        setattr(settings, "SCHWAB_REFRESH_TOKEN", "")

    def _current_refresh_token(self) -> str:
        if self._refresh_token:
            return self._refresh_token
        if settings.schwab_refresh_token:
            self._refresh_token = _clean_str(settings.schwab_refresh_token)
            settings.schwab_refresh_token = self._refresh_token
            return self._refresh_token
        token = latest_refresh_token("schwab")
        if token:
            self._refresh_token = _clean_str(token)
        return self._refresh_token

    async def _ensure_token(self) -> str:
        loop = asyncio.get_running_loop()
        if getattr(self, "_lock_loop", None) is not loop:
            self._token_lock = asyncio.Lock()
            self._lock_loop = loop
        task: Optional[asyncio.Task[_Token]] = None
        async with self._token_lock:
            now = dt.datetime.now(dt.timezone.utc)
            if self._token and self._token.expires_at > now:
                return self._token.access_token

            current = self._refresh_task
            if current is not None:
                if current.done():
                    self._refresh_task = None
                    self._refresh_task_loop = None
                elif self._refresh_task_loop is loop:
                    task = current
                    token_refresh_storm_prevented.inc()
                else:
                    self._refresh_task = None
                    self._refresh_task_loop = None

            if task is None:
                task = asyncio.create_task(self._refresh_access_token())
                self._refresh_task = task
                self._refresh_task_loop = loop

        assert task is not None  # for type-checkers
        try:
            refreshed = await task
        except Exception:
            async with self._token_lock:
                if self._refresh_task is task:
                    self._refresh_task = None
                    self._refresh_task_loop = None
            raise

        async with self._token_lock:
            self._token = refreshed
            if self._refresh_task is task:
                self._refresh_task = None
                self._refresh_task_loop = None
            return refreshed.access_token

    def clear_cached_token(self) -> None:
        self._token = None

    def last_status(self) -> Optional[int]:
        return self._last_status

    def set_refresh_token(self, refresh_token: str) -> None:
        self._refresh_token = _clean_str(refresh_token)
        self.clear_cached_token()
        self._clear_disable()

    # ------------------------------------------------------------------
    # Helpers
    async def _authed_request(
        self,
        method: str,
        path_or_url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout_ctx: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], int]:
        token = await self._ensure_token()
        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
            log_target = path_or_url
        else:
            base = API_BASE_URL.rstrip("/")
            url = f"{base}/{path_or_url.lstrip('/')}"
            log_target = path_or_url
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        t0 = time.monotonic()
        resp = await http_client.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout_ctx=timeout_ctx,
        )
        duration = time.monotonic() - t0
        self._last_status = resp.status_code
        logger.info(
            "schwab_http method=%s path=%s status=%s duration=%.2f",
            method,
            log_target,
            resp.status_code,
            duration,
        )
        if resp.status_code == 401:
            logger.info("schwab_token_expired path=%s", log_target)
            self.clear_cached_token()
            token = await self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            t0 = time.monotonic()
            resp = await http_client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout_ctx=timeout_ctx,
            )
            duration = time.monotonic() - t0
            self._last_status = resp.status_code
            logger.info(
                "schwab_http method=%s path=%s status=%s duration=%.2f retry=1",
                method,
                log_target,
                resp.status_code,
                duration,
            )

        if resp.status_code >= 400:
            body: Optional[str]
            try:
                body = resp.text[:256]
            except Exception:
                body = None
            raise SchwabAPIError(
                f"Schwab request failed ({resp.status_code})",
                status_code=resp.status_code,
                body=body,
            )

        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            raise SchwabAPIError(
                "Schwab response was not valid JSON",
                status_code=resp.status_code,
            )
        return data, resp.status_code

    # ------------------------------------------------------------------
    # Public API
    async def get_price_history(
        self,
        symbol: str,
        start: Optional[dt.datetime],
        end: Optional[dt.datetime],
        interval: str,
        *,
        timeout_ctx: Optional[dict] = None,
    ) -> pd.DataFrame:
        def _normalize_ts(ts: dt.datetime) -> int:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            else:
                ts = ts.astimezone(dt.timezone.utc)
            return int(ts.timestamp() * 1000)

        def _interval_to_frequency(value: str) -> Tuple[str, int]:
            raw = (value or "").strip().lower()
            aliases = {
                "m": ("minute", 1),
                "1m": ("minute", 1),
                "one_minute": ("minute", 1),
                "minute": ("minute", 1),
                "minutes": ("minute", 1),
                "d": ("daily", 1),
                "1d": ("daily", 1),
                "day": ("daily", 1),
                "daily": ("daily", 1),
                "w": ("weekly", 1),
                "1w": ("weekly", 1),
                "week": ("weekly", 1),
                "weekly": ("weekly", 1),
                "mo": ("monthly", 1),
                "1mo": ("monthly", 1),
                "1mon": ("monthly", 1),
                "1month": ("monthly", 1),
                "month": ("monthly", 1),
                "monthly": ("monthly", 1),
            }
            if raw in aliases:
                return aliases[raw]
            if raw.endswith("min"):
                digits = "".join(ch for ch in raw if ch.isdigit())
                freq = int(digits) if digits else 1
                return "minute", max(1, freq)
            if raw.endswith("m"):
                try:
                    freq = int(raw[:-1])
                except ValueError:
                    freq = 1
                return "minute", max(1, freq)
            if raw.endswith("h"):
                try:
                    hours = int(raw[:-1])
                except ValueError:
                    hours = 1
                return "minute", max(1, hours * 60)
            if raw.endswith("d"):
                try:
                    days = int(raw[:-1])
                except ValueError:
                    days = 1
                return "daily", max(1, days)
            if raw.endswith("w"):
                try:
                    weeks = int(raw[:-1])
                except ValueError:
                    weeks = 1
                return "weekly", max(1, weeks)
            if raw.endswith("mo") or raw.endswith("mon"):
                digits = "".join(ch for ch in raw if ch.isdigit())
                freq = int(digits) if digits else 1
                return "monthly", max(1, freq)
            return "minute", 1

        frequency_type, frequency = _interval_to_frequency(interval)

        if frequency_type == "minute":
            allowed = [1, 5, 10, 15, 30]
            if frequency not in allowed:
                # Clamp to the nearest supported Schwab intraday frequency.
                frequency = min(allowed, key=lambda opt: abs(opt - frequency))

        params: Dict[str, Any] = {
            "symbol": symbol,
            "frequencyType": frequency_type,
            "frequency": frequency,
        }

        if start is not None and end is not None:
            params["startDate"] = _normalize_ts(start)
            params["endDate"] = _normalize_ts(end)
        else:
            if frequency_type == "minute":
                params["periodType"] = "day"
                params["period"] = 10
            else:
                params["periodType"] = "year"
                params["period"] = 1

        params["needExtendedHoursData"] = "true"
        params["needPreviousClose"] = "false"

        data, _ = await self._authed_request(
            "GET",
            PRICE_HISTORY_URL,
            params=params,
            timeout_ctx=timeout_ctx,
        )
        candles = (
            data.get("candles")
            or data.get("data", {}).get("candles")
            or data.get("priceHistory", {}).get("candles")
            or []
        )
        records = []
        for candle in candles:
            ts = candle.get("datetime") or candle.get("timestamp")
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)):
                    unit = "ms" if ts > 10**10 else "s"
                    stamp = pd.to_datetime(ts, unit=unit, utc=True)
                else:
                    stamp = pd.to_datetime(ts, utc=True)
            except Exception:
                continue
            records.append(
                {
                    "ts": stamp,
                    "Open": candle.get("open"),
                    "High": candle.get("high"),
                    "Low": candle.get("low"),
                    "Close": candle.get("close"),
                    "Volume": candle.get("volume"),
                }
            )

        if not records:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )

        df = pd.DataFrame(records).set_index("ts").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize(dt.timezone.utc)
        else:
            df.index = df.index.tz_convert(dt.timezone.utc)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        # Preserve the expected OHLCV column order for downstream consumers.
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        return df

    async def get_quote(
        self,
        symbol: str,
        *,
        timeout_ctx: Optional[dict] = None,
    ) -> Dict[str, Any]:
        params = {"symbols": symbol}
        if self._account_id:
            params.setdefault("accountId", self._account_id)
        data, _ = await self._authed_request(
            "GET",
            f"{symbol}/quotes",
            params=params,
            timeout_ctx=timeout_ctx,
        )
        quote = (
            data.get("quote")
            or data.get("quotes", {}).get(symbol)
            or (data.get("data", {}).get("quotes", {}) if isinstance(data, dict) else {})
        )
        if isinstance(quote, dict) and symbol in quote:
            quote = quote[symbol]

        if not isinstance(quote, dict):
            return {}

        price = (
            quote.get("lastPrice")
            or quote.get("mark")
            or quote.get("close")
            or quote.get("regularMarketPrice")
        )
        ts_value = (
            quote.get("timestamp")
            or quote.get("quoteTimeInLong")
            or quote.get("regularMarketTime")
        )
        ts: Optional[dt.datetime] = None
        if ts_value is not None:
            try:
                if isinstance(ts_value, (int, float)):
                    unit = "ms" if ts_value > 10**10 else "s"
                    ts = dt.datetime.fromtimestamp(ts_value / (1000 if unit == "ms" else 1), dt.timezone.utc)
                else:
                    ts = pd.to_datetime(ts_value, utc=True).to_pydatetime()
            except Exception:
                ts = None

        if price is None:
            return {}

        return {
            "symbol": symbol,
            "price": float(price),
            "timestamp": ts,
            "source": "schwab",
        }


_client = SchwabClient()


async def get_price_history(
    symbol: str,
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
    interval: str,
    *,
    timeout_ctx: Optional[dict] = None,
) -> pd.DataFrame:
    return await _client.get_price_history(
        symbol, start, end, interval, timeout_ctx=timeout_ctx
    )


def disabled_state() -> Tuple[bool, Optional[str], Optional[int], Optional[SchwabAuthError]]:
    return _client.disabled_state()


def disable(
    *,
    reason: str,
    status_code: Optional[int] = None,
    ttl: Optional[float] = None,
    error: Optional[SchwabAuthError] = None,
) -> None:
    _client.disable(reason=reason, status_code=status_code, ttl=ttl, error=error)


async def get_quote(
    symbol: str,
    *,
    timeout_ctx: Optional[dict] = None,
) -> Dict[str, Any]:
    return await _client.get_quote(symbol, timeout_ctx=timeout_ctx)


def clear_cached_token() -> None:
    _client.clear_cached_token()


def last_status() -> Optional[int]:
    return _client.last_status()


def update_refresh_token(refresh_token: str) -> None:
    value = refresh_token or ""
    settings.schwab_refresh_token = value
    setattr(settings, "SCHWAB_REFRESH_TOKEN", value)
    _client.set_refresh_token(value)

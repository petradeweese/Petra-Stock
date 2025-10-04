import asyncio
import base64
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import settings
from services import http_client
from services.oauth_tokens import latest_refresh_token

logger = logging.getLogger(__name__)

TOKEN_URL = os.getenv(
    "SCHWAB_TOKEN_URL", "https://api.schwabapi.com/v1/oauth/token"
)
API_BASE_URL = os.getenv(
    "SCHWAB_API_BASE", "https://api.schwab.com/trader/v1/marketdata"
)


class SchwabAuthError(RuntimeError):
    """Raised when authentication credentials are missing or invalid."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        self._client_id = settings.schwab_client_id
        self._client_secret = settings.schwab_client_secret
        self._redirect_uri = settings.schwab_redirect_uri
        self._refresh_token = settings.schwab_refresh_token
        self._account_id = settings.schwab_account_id
        self._token: Optional[_Token] = None
        self._token_lock = asyncio.Lock()
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_status: Optional[int] = None

    # ------------------------------------------------------------------
    # Token handling
    async def _refresh_access_token(self) -> _Token:
        refresh_token = self._current_refresh_token()
        if not all(
            [
                self._client_id,
                self._client_secret,
                self._redirect_uri,
                refresh_token,
            ]
        ):
            raise SchwabAuthError("Missing Schwab OAuth configuration")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
        }
        basic_token = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic_token}",
        }

        t0 = time.monotonic()
        resp = await http_client.request(
            "POST", TOKEN_URL, data=payload, headers=headers
        )
        duration = time.monotonic() - t0
        if resp.status_code >= 400:
            logger.warning(
                "schwab_token_error status=%s duration=%.2f", resp.status_code, duration
            )
            raise SchwabAuthError(
                f"Token refresh failed ({resp.status_code})",
                status_code=resp.status_code,
            )

        try:
            data: Dict[str, Any] = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 0) or 0)
        if not token:
            raise SchwabAuthError("Token refresh response missing access_token")

        skewed = max(0, expires_in - 60)
        expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=skewed)
        logger.info(
            "schwab_token_refreshed expires_in=%s duration=%.2f", expires_in, duration
        )
        return _Token(access_token=token, expires_at=expires_at)

    def _current_refresh_token(self) -> str:
        if self._refresh_token:
            return self._refresh_token
        if settings.schwab_refresh_token:
            self._refresh_token = settings.schwab_refresh_token
            return self._refresh_token
        token = latest_refresh_token("schwab")
        if token:
            self._refresh_token = token
        return self._refresh_token

    async def _ensure_token(self) -> str:
        loop = asyncio.get_running_loop()
        if getattr(self, "_lock_loop", None) is not loop:
            self._token_lock = asyncio.Lock()
            self._lock_loop = loop
        async with self._token_lock:
            if self._token and self._token.expires_at > dt.datetime.now(dt.timezone.utc):
                return self._token.access_token
            self._token = await self._refresh_access_token()
            return self._token.access_token

    def clear_cached_token(self) -> None:
        self._token = None

    def last_status(self) -> Optional[int]:
        return self._last_status

    def set_refresh_token(self, refresh_token: str) -> None:
        self._refresh_token = refresh_token or ""
        self.clear_cached_token()

    # ------------------------------------------------------------------
    # Helpers
    async def _authed_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout_ctx: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], int]:
        token = await self._ensure_token()
        url = f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
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
            path,
            resp.status_code,
            duration,
        )
        if resp.status_code == 401:
            logger.info("schwab_token_expired path=%s", path)
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
                path,
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
        start: dt.datetime,
        end: dt.datetime,
        interval: str,
        *,
        timeout_ctx: Optional[dict] = None,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "start": int(start.timestamp() * 1000),
            "end": int(end.timestamp() * 1000),
        }
        if self._account_id:
            params.setdefault("accountId", self._account_id)

        data, _ = await self._authed_request(
            "GET",
            f"{symbol}/pricehistory",
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
    start: dt.datetime,
    end: dt.datetime,
    interval: str,
    *,
    timeout_ctx: Optional[dict] = None,
) -> pd.DataFrame:
    return await _client.get_price_history(
        symbol, start, end, interval, timeout_ctx=timeout_ctx
    )


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

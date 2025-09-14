import asyncio
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import polygon_client


def test_retry_success(monkeypatch, caplog):
    attempts = 0

    async def fake_get_json(url, headers=None):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            resp = httpx.Response(500, request=httpx.Request("GET", url))
            raise httpx.HTTPStatusError("boom", request=resp.request, response=resp)
        return {"results": []}

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(
        polygon_client, "http_client", SimpleNamespace(get_json=fake_get_json)
    )
    monkeypatch.setattr(polygon_client.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        polygon_client,
        "settings",
        SimpleNamespace(fetch_retry_max=4, fetch_retry_base_ms=1, fetch_retry_cap_ms=2),
    )

    start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)
    with caplog.at_level("WARNING"):
        asyncio.run(polygon_client._fetch_single("SPY", start, end))
    assert attempts == 3
    msgs = [r.message for r in caplog.records if "retry attempt" in r.message]
    assert len(msgs) == 2
    assert "status=500" in msgs[0]


def test_retry_exhaust(monkeypatch):
    attempts = 0

    async def fake_get_json(url, headers=None):
        nonlocal attempts
        attempts += 1
        resp = httpx.Response(500, request=httpx.Request("GET", url))
        raise httpx.HTTPStatusError("boom", request=resp.request, response=resp)

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(
        polygon_client, "http_client", SimpleNamespace(get_json=fake_get_json)
    )
    monkeypatch.setattr(polygon_client.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        polygon_client,
        "settings",
        SimpleNamespace(fetch_retry_max=2, fetch_retry_base_ms=1, fetch_retry_cap_ms=2),
    )

    start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(polygon_client._fetch_single("SPY", start, end))
    assert attempts == 2

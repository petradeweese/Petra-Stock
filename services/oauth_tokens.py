from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from db import get_db

logger = logging.getLogger(__name__)


@contextmanager
def _db_context(db_cursor):
    if db_cursor is not None:
        yield db_cursor
        return

    gen: Iterator = get_db()
    cursor = next(gen)
    try:
        yield cursor
        try:
            next(gen)
        except StopIteration:
            pass
    finally:
        gen.close()


def store_refresh_token(
    provider: str,
    refresh_token: str,
    *,
    account_id: str | None = None,
    db_cursor=None,
) -> None:
    if not provider:
        raise ValueError("provider is required")
    if not refresh_token:
        raise ValueError("refresh_token is required")

    timestamp = datetime.now(timezone.utc).isoformat()
    with _db_context(db_cursor) as cursor:
        cursor.execute(
            """
            INSERT INTO oauth_tokens(provider, created_at, refresh_token, account_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                created_at=excluded.created_at,
                refresh_token=excluded.refresh_token,
                account_id=excluded.account_id
            """,
            (provider, timestamp, refresh_token, account_id),
        )
    logger.info("oauth_token_saved provider=%s", provider)


def latest_refresh_token(
    provider: str,
    *,
    db_cursor=None,
) -> str:
    if not provider:
        return ""

    with _db_context(db_cursor) as cursor:
        cursor.execute(
            "SELECT refresh_token FROM oauth_tokens WHERE provider=? ORDER BY created_at DESC LIMIT 1",
            (provider,),
        )
        row = cursor.fetchone()
        if not row:
            return ""
        if isinstance(row, dict):  # pragma: no cover - sqlite3.Row in tests
            return str(row.get("refresh_token") or "")
        try:
            return str(row[0] or "")
        except (IndexError, TypeError):  # pragma: no cover - defensive
            return ""

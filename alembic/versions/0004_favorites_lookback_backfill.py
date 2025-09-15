"""Backfill favorites lookback values.

Revision ID: 0004
Revises: 0003
Create Date: 2024-09-01 00:00:00.000000
"""
from __future__ import annotations

import logging
import sqlite3

try:  # pragma: no cover - prefer real Alembic if available
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

try:  # pragma: no cover - prefer SQLAlchemy if available
    from sqlalchemy.exc import OperationalError as SAOperationalError  # type: ignore
except Exception:  # pragma: no cover - fallback when SQLAlchemy missing
    SAOperationalError = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_operation_errors: list[type[BaseException]] = []
if SAOperationalError is not None:
    _operation_errors.append(SAOperationalError)
if hasattr(sqlite3, "OperationalError"):
    _operation_errors.append(sqlite3.OperationalError)
OPERATIONAL_ERRORS: tuple[type[BaseException], ...] = tuple(_operation_errors)


def _is_operational_error(exc: BaseException) -> bool:
    if not OPERATIONAL_ERRORS:
        return False
    return isinstance(exc, OPERATIONAL_ERRORS)


def _safe_execute(sql: str, context: str) -> None:
    try:
        op.execute(sql)
    except Exception as exc:  # pragma: no cover - best effort logging only
        if _is_operational_error(exc):
            logger.warning(
                "Migration 0004 skipped %s due to operational error: %s",
                context,
                exc,
            )
            return
        logger.exception("Migration 0004 failed while executing %s", context)
        raise


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    cols = {
        row[1]
        for row in conn.exec_driver_sql("PRAGMA table_info(favorites)").fetchall()
    }

    if "lookback_years" not in cols:
        logger.info("Adding favorites.lookback_years column during migration 0004")
        op.execute("ALTER TABLE favorites ADD COLUMN lookback_years REAL")

    support_exists = "support_snapshot" in cols
    params_exists = "params_json" in cols

    if not support_exists:
        try:
            op.execute("ALTER TABLE favorites ADD COLUMN support_snapshot TEXT")
            logger.info(
                "Added missing favorites.support_snapshot column during migration 0004"
            )
            support_exists = True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "favorites.support_snapshot column missing; skipping JSON backfill: %s",
                exc,
            )

    if support_exists:
        _safe_execute(
            """
            UPDATE favorites
               SET lookback_years = CAST(json_extract(support_snapshot, '$.lookback_years') AS REAL)
             WHERE (lookback_years IS NULL OR lookback_years IN (0, 0.2))
               AND json_type(json_extract(support_snapshot, '$.lookback_years')) IN ('integer','real');
            """,
            "support_snapshot lookback backfill",
        )
    else:
        logger.warning(
            "Skipping favorites.support_snapshot backfill; column is not available"
        )

    if params_exists:
        fallback_sql = """
            UPDATE favorites
               SET lookback_years = CAST(json_extract(params_json, '$.lookback_years') AS REAL)
             WHERE (lookback_years IS NULL OR lookback_years IN (0, 0.2))
               AND json_type(json_extract(params_json, '$.lookback_years')) IN ('integer','real')
        """
        if support_exists:
            fallback_sql += """
               AND (
                    support_snapshot IS NULL
                    OR TRIM(COALESCE(CAST(support_snapshot AS TEXT), '')) = ''
                    OR COALESCE(
                        json_type(json_extract(support_snapshot, '$.lookback_years')),
                        ''
                    ) NOT IN ('integer','real')
               )
            """
        fallback_sql += ";"
        _safe_execute(fallback_sql, "params_json lookback fallback")


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")


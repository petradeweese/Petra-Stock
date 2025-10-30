"""Add forecast recipients column to settings

Revision ID: 0013
Revises: 0012
Create Date: 2024-10-05 00:00:00.000000
"""
from __future__ import annotations

try:  # pragma: no cover - prefer real Alembic when available
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op  # type: ignore

from sqlalchemy import inspect, text

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def _has_column(inspector, table: str, column: str) -> bool:
    try:
        columns = inspector.get_columns(table)
    except Exception:
        return False
    return any((col.get("name") or "").lower() == column.lower() for col in columns)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not _has_column(inspector, "settings", "forecast_recipients"):
        op.execute("ALTER TABLE settings ADD COLUMN forecast_recipients TEXT")
        bind.execute(
            text(
                "UPDATE settings SET forecast_recipients = '' "
                "WHERE forecast_recipients IS NULL"
            )
        )


def downgrade() -> None:  # pragma: no cover - destructive downgrade
    raise RuntimeError("downgrade not supported")

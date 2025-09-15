"""add deleted_at to overnight batches

Revision ID: 0005
Revises: 0004
Create Date: 2024-09-15 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE overnight_batches ADD COLUMN deleted_at TEXT")


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

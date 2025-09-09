"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from db import SCHEMA
    for stmt in SCHEMA:
        op.execute(stmt)


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

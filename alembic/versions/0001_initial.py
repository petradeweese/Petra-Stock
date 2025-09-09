"""Initial schema

Revision ID: 0001
Revises: 
Create Date: 2025-09-09 08:30:00.000000
"""
try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op


def upgrade() -> None:
    from db import SCHEMA
    for stmt in SCHEMA:
        op.execute(stmt)


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

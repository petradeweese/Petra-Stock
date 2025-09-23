"""Add OAuth token storage table

Revision ID: 0008
Revises: 0007
Create Date: 2024-11-01 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            provider TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            refresh_token TEXT NOT NULL,
            account_id TEXT
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_tokens_created_at ON oauth_tokens(created_at);"
    )


def downgrade() -> None:  # pragma: no cover - destructive downgrade
    raise RuntimeError("downgrade not supported")

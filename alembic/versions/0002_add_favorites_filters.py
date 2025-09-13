"""Add favorites filter columns to settings

Revision ID: 0002
Revises: 0001
Create Date: 2024-07-30 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind() if hasattr(op, "get_bind") else op.conn
    if hasattr(conn, "exec_driver_sql"):
        info = conn.exec_driver_sql("PRAGMA table_info(settings)")
    else:  # pragma: no cover - sqlite3 connection in tests
        info = conn.execute("PRAGMA table_info(settings)")
    existing = {row[1] for row in info}

    if "fav_filter_liquidity" not in existing:
        op.execute("ALTER TABLE settings ADD COLUMN fav_filter_liquidity INTEGER DEFAULT 1")
    if "fav_filter_trend" not in existing:
        op.execute("ALTER TABLE settings ADD COLUMN fav_filter_trend INTEGER DEFAULT 1")
    if "fav_filter_earnings" not in existing:
        op.execute("ALTER TABLE settings ADD COLUMN fav_filter_earnings INTEGER DEFAULT 1")

    op.execute(
        "UPDATE settings SET fav_filter_liquidity=1, fav_filter_trend=1, fav_filter_earnings=1 WHERE id=1"
    )


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

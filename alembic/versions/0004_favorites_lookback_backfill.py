"""Backfill favorites lookback values

Revision ID: 0004
Revises: 0003
Create Date: 2024-09-01 00:00:00.000000
"""
from __future__ import annotations

try:  # pragma: no cover - prefer real Alembic if available
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(favorites)").fetchall()]
    if "lookback_years" not in cols:
        op.execute("ALTER TABLE favorites ADD COLUMN lookback_years REAL")

    op.execute(
        """
        UPDATE favorites
           SET lookback_years = CAST(json_extract(support_snapshot, '$.lookback_years') AS REAL)
         WHERE (lookback_years IS NULL OR lookback_years IN (0, 0.2))
           AND json_type(json_extract(support_snapshot, '$.lookback_years')) IN ('integer','real');
        """
    )


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")


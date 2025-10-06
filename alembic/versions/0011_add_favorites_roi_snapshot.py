"""Add roi_snapshot column to favorites

Revision ID: 0011
Revises: 0010
Create Date: 2024-12-12 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("favorites")}

    if "roi_snapshot" not in columns:
        op.add_column(
            "favorites",
            sa.Column(
                "roi_snapshot",
                sa.JSON().with_variant(sa.Text(), "sqlite"),
                nullable=True,
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("favorites")}

    if "roi_snapshot" in columns:
        op.drop_column("favorites", "roi_snapshot")

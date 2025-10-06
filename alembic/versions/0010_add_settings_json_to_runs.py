"""Add settings_json to runs

Revision ID: 0010
Revises: 0009
Create Date: 2024-11-19 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("runs")}

    if "settings_json" not in columns:
        op.add_column(
            "runs",
            sa.Column(
                "settings_json",
                sa.JSON().with_variant(
                    postgresql.JSONB(astext_type=sa.Text()), "postgresql"
                ),
                nullable=True,
            ),
        )

    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX IF NOT EXISTS ix_runs_settings_json ON runs USING GIN (settings_json)"
            )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute("DROP INDEX IF EXISTS ix_runs_settings_json")
    op.drop_column("runs", "settings_json")

"""add greeks profile columns

Revision ID: 0002
Revises: 0001
Create Date: 2024-05-13 00:00:00.000000
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
    conn = op.get_bind()
    cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(settings)").fetchall()]
    if "greeks_profile_json" not in cols:
        op.execute(
            "ALTER TABLE settings ADD COLUMN greeks_profile_json TEXT DEFAULT '{}'"
        )
    cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(favorites)").fetchall()]
    if "greeks_override_json" not in cols:
        op.execute("ALTER TABLE favorites ADD COLUMN greeks_override_json TEXT")


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

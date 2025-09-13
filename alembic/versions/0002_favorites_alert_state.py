"""favorites alert state

Revision ID: 0002
Revises: 0001_initial
Create Date: 2024-05-31
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    cols = [r[1] for r in conn.execute(sa.text("PRAGMA table_info('favorites')")).fetchall()]
    if "alerts_enabled" not in cols:
        op.add_column("favorites", sa.Column("alerts_enabled", sa.Integer(), server_default="0", nullable=False))
        op.add_column("favorites", sa.Column("cooldown_minutes", sa.Integer(), server_default="30", nullable=False))
        op.add_column("favorites", sa.Column("last_notified_ts", sa.Text(), server_default="", nullable=False))
        op.add_column("favorites", sa.Column("last_signal_bar", sa.Text(), server_default="", nullable=False))
    cols = [r[1] for r in conn.execute(sa.text("PRAGMA table_info('settings')")).fetchall()]
    if "fav_cooldown_minutes" not in cols:
        op.add_column("settings", sa.Column("fav_cooldown_minutes", sa.Integer(), server_default="30", nullable=False))



def downgrade() -> None:
    op.drop_column("favorites", "last_signal_bar")
    op.drop_column("favorites", "last_notified_ts")
    op.drop_column("favorites", "cooldown_minutes")
    op.drop_column("favorites", "alerts_enabled")
    op.drop_column("settings", "fav_cooldown_minutes")

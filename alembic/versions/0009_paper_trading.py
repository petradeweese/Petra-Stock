"""Paper trading tables

Revision ID: 0009
Revises: 0008
Create Date: 2024-11-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table("paper_settings"):
        op.create_table(
            "paper_settings",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("starting_balance", sa.Float, nullable=False, server_default="10000"),
            sa.Column("max_pct", sa.Float, nullable=False, server_default="10"),
            sa.Column("started_at", sa.Text, nullable=True),
            sa.Column("status", sa.Text, nullable=False, server_default="inactive"),
            sa.CheckConstraint("id = 1"),
        )
    else:
        settings_cols = {col["name"] for col in inspector.get_columns("paper_settings")}
        if "started_at" not in settings_cols:
            op.add_column("paper_settings", sa.Column("started_at", sa.Text))
        if "status" not in settings_cols:
            op.add_column("paper_settings", sa.Column("status", sa.Text, server_default="inactive"))

    op.execute(
        """
        INSERT OR IGNORE INTO paper_settings(id, starting_balance, max_pct, started_at, status)
        VALUES(1, 10000, 10, NULL, 'inactive')
        """
    )

    if not inspector.has_table("paper_equity"):
        op.create_table(
            "paper_equity",
            sa.Column("ts", sa.Text, primary_key=True),
            sa.Column("balance", sa.Float, nullable=False),
        )

    if not inspector.has_table("paper_trades"):
        op.create_table(
            "paper_trades",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("ticker", sa.Text, nullable=False),
            sa.Column("call_put", sa.Text, nullable=False),
            sa.Column("strike", sa.Float, nullable=True),
            sa.Column("expiry", sa.Text, nullable=True),
            sa.Column("qty", sa.Integer, nullable=False),
            sa.Column("interval", sa.Text, nullable=True),
            sa.Column("entry_time", sa.Text, nullable=False),
            sa.Column("executed_at", sa.Text, nullable=False),
            sa.Column("entry_price", sa.Float, nullable=False),
            sa.Column("exit_time", sa.Text, nullable=True),
            sa.Column("exit_price", sa.Float, nullable=True),
            sa.Column("roi_pct", sa.Float, nullable=True),
            sa.Column("status", sa.Text, nullable=False),
            sa.Column("source_alert_id", sa.Text, nullable=True),
            sa.Column("price_source", sa.Text, nullable=True),
        )
    else:
        trade_cols = {col["name"] for col in inspector.get_columns("paper_trades")}
        if "interval" not in trade_cols:
            op.add_column("paper_trades", sa.Column("interval", sa.Text))
        if "executed_at" not in trade_cols:
            op.add_column("paper_trades", sa.Column("executed_at", sa.Text))
            op.execute("UPDATE paper_trades SET executed_at = entry_time WHERE executed_at IS NULL")
        if "status" not in trade_cols:
            op.add_column("paper_trades", sa.Column("status", sa.Text, server_default="open"))

    op.execute("CREATE INDEX IF NOT EXISTS ix_paper_trades_status ON paper_trades(status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_paper_trades_ticker ON paper_trades(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_paper_trades_executed_at ON paper_trades(executed_at)")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_trades_source_alert
        ON paper_trades(source_alert_id) WHERE source_alert_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_paper_trades_source_alert")
    op.execute("DROP INDEX IF EXISTS ix_paper_trades_executed_at")
    op.execute("DROP INDEX IF EXISTS ix_paper_trades_ticker")
    op.execute("DROP INDEX IF EXISTS ix_paper_trades_status")
    if inspect(op.get_bind()).has_table("paper_trades"):
        op.drop_table("paper_trades")
    if inspect(op.get_bind()).has_table("paper_equity"):
        op.drop_table("paper_equity")
    if inspect(op.get_bind()).has_table("paper_settings"):
        op.drop_table("paper_settings")

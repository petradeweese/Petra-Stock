"""Add forward test exit metadata columns

Revision ID: 0006
Revises: 0005
Create Date: 2024-09-16 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_COLUMNS_TO_ADD = {
    "exit_reason": "TEXT",
    "bars_to_exit": "INTEGER",
    "max_drawdown_pct": "REAL",
    "max_runup_pct": "REAL",
    "r_multiple": "REAL",
    "option_roi_proxy": "REAL",
}


def _existing_forward_columns() -> set[str]:
    get_bind = getattr(op, "get_bind", None)
    if callable(get_bind):
        conn = get_bind()
        try:
            result = conn.exec_driver_sql("PRAGMA table_info('forward_tests')")
        except AttributeError:
            result = conn.execute("PRAGMA table_info('forward_tests')")
    else:
        result = op.execute("PRAGMA table_info('forward_tests')")
    return {row[1] for row in result.fetchall()}


def upgrade() -> None:
    existing = _existing_forward_columns()
    for name, column_type in _COLUMNS_TO_ADD.items():
        if name not in existing:
            op.execute(f"ALTER TABLE forward_tests ADD COLUMN {name} {column_type}")


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

"""add overnight tables

Revision ID: 0003
Revises: 0002
Create Date: 2024-08-19 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS overnight_batches (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            starts_at     TEXT,
            status        TEXT NOT NULL DEFAULT 'queued',
            label         TEXT,
            note          TEXT,
            start_override INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS overnight_items (
            id           TEXT PRIMARY KEY,
            batch_id     TEXT NOT NULL REFERENCES overnight_batches(id) ON DELETE CASCADE,
            position     INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'queued',
            created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at   TEXT,
            finished_at  TEXT,
            run_id       INTEGER,
            error        TEXT
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_overnight_items_batch_pos
        ON overnight_items(batch_id, position);
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS overnight_prefs (
            id                         INTEGER PRIMARY KEY CHECK (id=1),
            enabled                    INTEGER NOT NULL DEFAULT 1,
            window_start               TEXT NOT NULL DEFAULT '01:00',
            window_end                 TEXT NOT NULL DEFAULT '08:00',
            timezone                   TEXT NOT NULL DEFAULT 'SERVER',
            sleep_ms_between_items     INTEGER NOT NULL DEFAULT 500,
            max_failures               INTEGER NOT NULL DEFAULT 3,
            send_mms_on_completion     INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def downgrade() -> None:
    raise RuntimeError("downgrade not supported")

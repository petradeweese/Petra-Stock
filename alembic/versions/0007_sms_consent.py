"""Add SMS consent and delivery log tables

Revision ID: 0007
Revises: 0006
Create Date: 2024-10-01 00:00:00.000000
"""
from __future__ import annotations

try:
    from alembic import op  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from alembic_stub import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sms_consent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            phone_e164 TEXT NOT NULL,
            consent_text TEXT NOT NULL,
            consent_at TEXT NOT NULL,
            ip TEXT,
            user_agent TEXT,
            revoked_at TEXT,
            method TEXT NOT NULL DEFAULT 'settings',
            verification_id TEXT
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sms_consent_user ON sms_consent(user_id, revoked_at);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sms_consent_phone ON sms_consent(phone_e164, consent_at DESC);"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sms_delivery_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            phone_e164 TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            message_type TEXT,
            body_hash TEXT
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sms_delivery_user ON sms_delivery_log(user_id, sent_at);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sms_delivery_phone ON sms_delivery_log(phone_e164, sent_at DESC);"
    )


def downgrade() -> None:  # pragma: no cover - destructive downgrade
    raise RuntimeError("downgrade not supported")

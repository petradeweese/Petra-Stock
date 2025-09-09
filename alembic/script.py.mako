"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma, if_=down_revision}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

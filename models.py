"""SQLAlchemy ORM models used by Petra Stock services."""

from __future__ import annotations

from sqlalchemy import Integer, JSON, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class Run(Base):
    """Archive run metadata stored in the ``runs`` table."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[str | None] = mapped_column(Text)
    scan_type: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[str | None] = mapped_column(Text)
    universe: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[str | None] = mapped_column(Text)
    hit_count: Mapped[int | None] = mapped_column(Integer)
    settings_json: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB(astext_type=Text()), "postgresql"), nullable=True
    )

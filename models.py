"""SQLAlchemy ORM models used by Petra Stock services."""

from __future__ import annotations

from sqlalchemy import Float, Integer, JSON, Text
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


class Favorite(Base):
    """Favorites saved from the scanner UI."""

    __tablename__ = "favorites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    interval: Mapped[str | None] = mapped_column(Text, nullable=False, default="15m")
    rule: Mapped[str] = mapped_column(Text, nullable=False)
    target_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_unit: Mapped[str | None] = mapped_column(Text, nullable=True)
    ref_avg_dd: Mapped[float | None] = mapped_column(Float, nullable=True)
    lookback_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_support: Mapped[int | None] = mapped_column(Integer, nullable=True)
    support_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    roi_snapshot: Mapped[dict | list | float | int | str | None] = mapped_column(
        JSON().with_variant(Text(), "sqlite"),
        nullable=True,
    )
    hit_pct_snapshot: Mapped[float | None] = mapped_column(Float, nullable=True)
    dd_pct_snapshot: Mapped[float | None] = mapped_column(Float, nullable=True)
    rule_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings_json_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_at: Mapped[str | None] = mapped_column(Text, nullable=True)

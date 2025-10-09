"""Scrub persisted contact info and update business phone.

Revision ID: 0012
Revises: 0011
Create Date: 2025-01-15 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


PHONE_VALUE = "+1 4705584503"
LEGACY_PHONE = "+1 (555) 555-1212"


PHONE_COLUMNS: dict[str, tuple[str, ...]] = {
    "settings": ("phone", "business_phone", "company_phone"),
    "companies": ("phone", "business_phone", "company_phone"),
    "profiles": ("phone", "business_phone", "company_phone"),
    "favorites": ("phone", "contact_phone"),
}


ADDRESS_COLUMNS: dict[str, tuple[tuple[str, bool], ...]] = {}


LEGACY_ADDRESS_DEFAULTS: dict[str, str] = {
    "address": "123 Example Street",
    "address_1": "123 Example Street",
    "address_2": "Suite 100",
    "city": "City",
    "region": "ST",
    "state": "ST",
    "postal": "00000",
    "zip": "00000",
}


def _address_columns_for_table(inspector: Inspector, table: str) -> tuple[tuple[str, bool], ...]:
    if table in ADDRESS_COLUMNS:
        return ADDRESS_COLUMNS[table]

    try:
        columns = inspector.get_columns(table)
    except NoSuchTableError:
        ADDRESS_COLUMNS[table] = tuple()
        return tuple()
    except Exception:
        ADDRESS_COLUMNS[table] = tuple()
        return tuple()

    address_like = []
    for col in columns:
        name = col.get("name") or ""
        if any(token in name.lower() for token in ("address", "street", "city", "region", "state", "postal", "zip")):
            address_like.append((name, bool(col.get("nullable", True))))

    ADDRESS_COLUMNS[table] = tuple(address_like)
    return ADDRESS_COLUMNS[table]


def _get_columns(inspector: Inspector, table: str) -> set[str]:
    try:
        return {col["name"] for col in inspector.get_columns(table)}
    except NoSuchTableError:
        return set()
    except Exception:
        return set()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table, candidates in PHONE_COLUMNS.items():
        existing = _get_columns(inspector, table)
        if not existing:
            continue
        for column in candidates:
            if column in existing:
                bind.execute(
                    text(
                        f"UPDATE {table} SET {column} = :phone "
                        f"WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) != ''"
                    ),
                    {"phone": PHONE_VALUE},
                )

    for table in PHONE_COLUMNS.keys():
        columns = _address_columns_for_table(inspector, table)
        if not columns:
            continue
        for column, nullable in columns:
            if nullable:
                bind.execute(
                    text(
                        f"UPDATE {table} SET {column} = NULL "
                        f"WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) != ''"
                    )
                )
            else:
                bind.execute(
                    text(
                        f"UPDATE {table} SET {column} = '' "
                        f"WHERE TRIM(CAST({column} AS TEXT)) != ''"
                    )
                )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table, candidates in PHONE_COLUMNS.items():
        existing = _get_columns(inspector, table)
        if not existing:
            continue
        for column in candidates:
            if column in existing:
                bind.execute(
                    text(
                        f"UPDATE {table} SET {column} = :phone "
                        f"WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) != ''"
                    ),
                    {"phone": LEGACY_PHONE},
                )

    for table in PHONE_COLUMNS.keys():
        columns = _address_columns_for_table(inspector, table)
        if not columns:
            continue
        for column, nullable in columns:
            key = column.lower()
            value = LEGACY_ADDRESS_DEFAULTS.get(key)
            if value is None:
                value = LEGACY_ADDRESS_DEFAULTS.get(next((k for k in LEGACY_ADDRESS_DEFAULTS if k in key), ""))
            if value is None:
                value = ""
            if not nullable and value == "":
                value = "N/A"
            bind.execute(
                text(
                    f"UPDATE {table} SET {column} = :value "
                    f"WHERE {column} IS NULL OR TRIM(CAST({column} AS TEXT)) = ''"
                ),
                {"value": value},
            )

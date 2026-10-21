"""Add the REPORT status source.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-21

``REPORT`` marks availability read from a published injury table (BasketNews's
daily EuroLeague report): one row per player with an editor's status, so it is
trusted like a manual entry rather than thresholded like a news reading.

**The downgrade deletes data**, as in 0005: PostgreSQL cannot drop an enum
value, so the type is rebuilt, and rows still holding ``REPORT`` are removed
first.
"""

from __future__ import annotations

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | None = None
depends_on: str | None = None

_PREVIOUS_VALUES: tuple[str, ...] = ("PROVIDER", "NEWS", "MANUAL")
_TABLES = ("player_status_events", "starting_lineup_entries")


def upgrade() -> None:
    op.execute("ALTER TYPE statussourcetype ADD VALUE IF NOT EXISTS 'REPORT'")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DELETE FROM {table} WHERE source_type::text = 'REPORT'")
    values = ", ".join(f"'{value}'" for value in _PREVIOUS_VALUES)
    op.execute("ALTER TYPE statussourcetype RENAME TO statussourcetype_old")
    op.execute(f"CREATE TYPE statussourcetype AS ENUM ({values})")
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN source_type DROP DEFAULT")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN source_type TYPE statussourcetype "
            "USING source_type::text::statussourcetype"
        )
        op.execute(f"ALTER TABLE {table} ALTER COLUMN source_type SET DEFAULT 'PROVIDER'")
    op.execute("DROP TYPE statussourcetype_old")

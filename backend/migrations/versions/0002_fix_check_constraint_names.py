"""Rename doubly-prefixed check constraints.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-13

The naming convention is ``ck_%(table_name)s_%(constraint_name)s``. Revision 0001
passed constraint names that were *already* prefixed, so the convention applied
its prefix a second time::

    ck_news_events_ck_news_events_confidence_in_unit_interval

and where that exceeded PostgreSQL's 63-byte limit, Alembic truncated and
appended a hash::

    ck_player_game_stats_ck_player_game_stats_seconds_playe_1438

The ORM renders the correct single-prefixed names, so any database created by the
original 0001 has drifted from the models. ``test_no_pending_schema_changes``
caught this on its first ever run.

Revision 0001 is corrected at source, so a fresh database gets the right names
and this revision finds nothing to do. An existing database gets the renames.
Both paths end identically, which is the property that matters.

Renaming rather than dropping and recreating: a rename is a catalogue update that
holds no lock on the data and cannot fail validation against existing rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: ``(table, correct constraint name)``. The incorrect names are matched by
#: pattern rather than listed, because truncation made several of them
#: unpredictable -- they carry a hash suffix derived from the full string.
_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("news_events", "ck_news_events_confidence_in_unit_interval"),
    ("player_game_stats", "ck_player_game_stats_seconds_played_non_negative"),
    ("player_status_events", "ck_player_status_events_confidence_in_unit_interval"),
    ("player_team_stints", "ck_player_team_stints_effective_interval_valid"),
    ("team_coach_stints", "ck_team_coach_stints_effective_interval_valid"),
)

# Finds a CHECK constraint on the table whose name carries the duplicated
# "ck_<table>_ck_" prefix, and renames it to the convention-correct name.
# Wrapped in a DO block so it is a no-op on a database that never had the bug.
_RENAME = """
DO $$
DECLARE
    wrong_name text;
BEGIN
    SELECT con.conname INTO wrong_name
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
    WHERE rel.relname = '{table}'
      AND nsp.nspname = current_schema()
      AND con.contype = 'c'
      AND con.conname LIKE 'ck\\_{table}\\_ck\\_%'
    LIMIT 1;

    IF wrong_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE {table} RENAME CONSTRAINT %I TO %I',
            wrong_name, '{correct}'
        );
        RAISE NOTICE 'renamed % to {correct}', wrong_name;
    END IF;
END $$;
"""

# The reverse: restore the doubled name, so downgrade is a true inverse.
_REVERT = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        WHERE rel.relname = '{table}' AND con.conname = '{correct}'
    ) THEN
        EXECUTE 'ALTER TABLE {table} RENAME CONSTRAINT {correct} TO {doubled}';
    END IF;
END $$;
"""


def upgrade() -> None:
    for table, correct in _CONSTRAINTS:
        op.execute(_RENAME.format(table=table, correct=correct))


def downgrade() -> None:
    for table, correct in _CONSTRAINTS:
        # Reconstruct the doubled form, truncated to PostgreSQL's limit exactly
        # as the original defect produced it.
        doubled = f"ck_{table}_{correct}"[:63]
        op.execute(_REVERT.format(table=table, correct=correct, doubled=doubled))

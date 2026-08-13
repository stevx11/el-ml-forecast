"""Every database identifier must fit PostgreSQL's 63-byte limit.

This is a unit test, not an integration test, on purpose: it needs no database
and runs in milliseconds, so an over-long name is caught before anyone waits on
a migration to fail.

The bug it exists to prevent: SQLAlchemy's naming convention builds foreign key
names as ``fk_<table>_<column>_<referred_table>``. With long table names --
``predictions`` referencing ``player_feature_snapshots`` -- that produced a
66-character identifier. PostgreSQL's limit is 63 bytes, and SQLAlchemy raises
rather than silently truncating, so ``alembic upgrade head`` died partway through
creating the schema.

Constraints are only rendered when the metadata is inspected, so this failure is
invisible to import-time checks and to every test that does not actually build
the schema.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, UniqueConstraint

# euroleague_ai.models must be imported for its tables to register on the metadata.
import euroleague_ai.models  # noqa: F401
from euroleague_ai.db.base import Base

# PostgreSQL's NAMEDATALEN is 64, giving 63 usable bytes.
MAX_IDENTIFIER_LENGTH = 63

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"


def _named_constraints() -> list[tuple[str, str, str]]:
    """Return ``(table, kind, identifier)`` for every named schema object."""
    found: list[tuple[str, str, str]] = []

    for table_name, table in Base.metadata.tables.items():
        found.append((table_name, "table", table_name))

        for column in table.columns:
            found.append((table_name, "column", column.name))

        for constraint in table.constraints:
            if constraint.name is None:
                continue
            kind = {
                ForeignKeyConstraint: "foreign key",
                UniqueConstraint: "unique",
                CheckConstraint: "check",
            }.get(type(constraint), "constraint")
            found.append((table_name, kind, str(constraint.name)))

        for index in table.indexes:
            if index.name is not None:
                found.append((table_name, "index", str(index.name)))

    return found


class TestOrmIdentifiers:
    def test_every_identifier_fits_postgres(self) -> None:
        """Names are rendered through the naming convention, then measured."""
        offenders = [
            (table, kind, name, len(name))
            for table, kind, name in _named_constraints()
            if len(name.encode("utf-8")) > MAX_IDENTIFIER_LENGTH
        ]

        assert not offenders, "identifiers exceeding PostgreSQL's 63-byte limit:\n" + "\n".join(
            f"  {table}.{kind}: {name} ({length} chars, over by "
            f"{length - MAX_IDENTIFIER_LENGTH})"
            for table, kind, name, length in offenders
        )

    def test_the_known_long_foreign_key_is_named_explicitly(self) -> None:
        """Regression: predictions -> player_feature_snapshots.

        Left to the naming convention this is 66 characters. If someone removes
        the explicit name, the generic test above catches it -- but this one says
        precisely which constraint and why.
        """
        predictions = Base.metadata.tables["predictions"]
        fk = next(
            constraint
            for constraint in predictions.constraints
            if isinstance(constraint, ForeignKeyConstraint)
            and "player_feature_snapshot_id" in [c.name for c in constraint.columns]
        )
        assert fk.name == "fk_predictions_player_feature_snapshot_id"
        assert len(str(fk.name)) <= MAX_IDENTIFIER_LENGTH


class TestMigrationIdentifiers:
    """The migration hardcodes names, so it needs checking independently.

    The ORM and the migration can disagree: fixing one and forgetting the other
    is exactly how this bug survived its first fix attempt.
    """

    @pytest.fixture(scope="class")
    def migration_sources(self) -> str:
        return "\n".join(
            path.read_text(encoding="utf-8") for path in MIGRATIONS_DIR.glob("*.py")
        )

    def test_no_migration_identifier_is_too_long(self, migration_sources: str) -> None:
        names = set(re.findall(r'name="([a-z0-9_]+)"', migration_sources))
        names |= set(re.findall(r'op\.create_index\(\s*"([a-z0-9_]+)"', migration_sources))
        names |= set(re.findall(r'op\.create_table\(\s*\n?\s*"([a-z0-9_]+)"', migration_sources))

        assert names, "no identifiers found -- the extraction patterns have drifted"

        offenders = sorted(
            (name for name in names if len(name.encode("utf-8")) > MAX_IDENTIFIER_LENGTH),
            key=len,
            reverse=True,
        )
        assert not offenders, "migration identifiers over 63 bytes:\n" + "\n".join(
            f"  {name} ({len(name)} chars)" for name in offenders
        )

    def test_migration_and_orm_agree_on_the_shortened_name(
        self, migration_sources: str
    ) -> None:
        """Both sides must carry the same explicit name, or the schema drifts."""
        assert "fk_predictions_player_feature_snapshot_id" in migration_sources
        assert (
            "fk_predictions_player_feature_snapshot_id_player_feature_snapshots"
            not in migration_sources
        )

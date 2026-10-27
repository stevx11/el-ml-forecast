"""Schema drift check.

The rest of the suite already runs against the real migration -- the ``engine``
fixture applies it rather than calling ``create_all``. What this file adds is the
check that the migration and the ORM models still *agree*: a model change that
never got a migration, or a migration that drifted from the models, shows up here
as a non-empty autogenerate diff.
"""

from __future__ import annotations

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Connection, inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from euroleague_ai.db.base import Base
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

# Diff entries that are noise rather than drift.
#
# Alembic reports server-side defaults as differences whenever the rendered SQL
# text differs from the Python-side default, even when the two are semantically
# identical ("false" vs sa.false()). Comparing those would fail permanently for
# no benefit, so only structural differences are treated as drift.
_IGNORED_DIFF_KINDS = frozenset({"modify_default", "modify_comment"})


def _structural_diffs(raw: list) -> list:
    """Flatten Alembic's diff output and drop the noisy entries."""
    flattened: list = []
    for entry in raw:
        # Column-level diffs arrive as a list of tuples; table-level as a tuple.
        candidates = entry if isinstance(entry, list) else [entry]
        for item in candidates:
            if isinstance(item, tuple) and item and item[0] not in _IGNORED_DIFF_KINDS:
                flattened.append(item)
    return flattened


def _diff(connection: Connection) -> list:
    context = MigrationContext.configure(
        connection,
        opts={"compare_type": True, "compare_server_default": False},
    )
    return _structural_diffs(compare_metadata(context, Base.metadata))


class TestMigrationMatchesModels:
    async def test_migration_creates_every_model_table(self, engine: AsyncEngine) -> None:
        async with engine.connect() as connection:
            tables = await connection.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )

        missing = set(Base.metadata.tables) - tables
        assert not missing, f"migration does not create: {sorted(missing)}"

    async def test_no_pending_schema_changes(self, engine: AsyncEngine) -> None:
        """Autogenerate against the migrated database must find nothing to do.

        If this fails, the models and the migration have diverged. Run
        ``alembic revision --autogenerate`` and review what it proposes -- but
        read it before applying: a renamed column is indistinguishable from a
        drop plus an add, and applying that blindly deletes data.
        """
        async with engine.connect() as connection:
            differences = await connection.run_sync(_diff)

        assert not differences, "models and migration have drifted apart:\n" + "\n".join(
            f"  {item}" for item in differences
        )

    async def test_alembic_version_is_stamped(self, engine: AsyncEngine) -> None:
        """A migrated database must know which revision it is on."""
        async with engine.connect() as connection:
            tables = await connection.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
        assert "alembic_version" in tables

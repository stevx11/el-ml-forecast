"""Add the PLAYS prediction target.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-21

``PLAYS`` is the probability that a projected player takes the floor at all.
Every stat model predicts a line *given that he plays*, and inference projects
the whole candidate list, which always includes a few players who will not
appear. Storing that probability as an ordinary prediction row means it reuses
the registry, the activation gate, scoring and the API unchanged, instead of
growing a parallel set of tables.

``ALTER TYPE ... ADD VALUE`` is allowed inside a transaction on PostgreSQL 12
and later. The new value just cannot be used until the transaction commits,
and nothing in this migration uses it.

**The downgrade deletes data.** PostgreSQL cannot drop a value from an enum, so
the type is rebuilt without it, and any row still holding ``PLAYS`` would make
that cast fail. Those rows are deleted first: outcomes, then predictions, then
model versions, in foreign-key order. Downgrading past this revision is a
deliberate act and should be one.
"""

from __future__ import annotations

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | None = None
depends_on: str | None = None

#: The enum as it stood before this revision, needed to rebuild it on downgrade.
_PREVIOUS_VALUES: tuple[str, ...] = (
    "MINUTES",
    "POINTS",
    "REBOUNDS",
    "ASSISTS",
    "THREE_POINTERS_MADE",
    "STEALS",
    "BLOCKS",
    "TURNOVERS",
    "FIELD_GOALS_MADE",
    "FIELD_GOALS_ATTEMPTED",
    "THREE_POINTERS_ATTEMPTED",
    "FREE_THROWS_MADE",
    "FREE_THROWS_ATTEMPTED",
    "VALUATION",
)


def upgrade() -> None:
    op.execute("ALTER TYPE predictiontarget ADD VALUE IF NOT EXISTS 'PLAYS'")


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM prediction_outcomes
        WHERE prediction_id IN (
            SELECT id FROM predictions WHERE target::text = 'PLAYS'
        )
        """
    )
    op.execute("DELETE FROM predictions WHERE target::text = 'PLAYS'")
    op.execute(
        """
        DELETE FROM predictions WHERE model_version_id IN (
            SELECT id FROM model_versions WHERE target::text = 'PLAYS'
        )
        """
    )
    op.execute("DELETE FROM model_versions WHERE target::text = 'PLAYS'")

    values = ", ".join(f"'{value}'" for value in _PREVIOUS_VALUES)
    op.execute("ALTER TYPE predictiontarget RENAME TO predictiontarget_old")
    op.execute(f"CREATE TYPE predictiontarget AS ENUM ({values})")
    for table in ("model_versions", "predictions"):
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN target TYPE predictiontarget "
            "USING target::text::predictiontarget"
        )
    op.execute("DROP TYPE predictiontarget_old")

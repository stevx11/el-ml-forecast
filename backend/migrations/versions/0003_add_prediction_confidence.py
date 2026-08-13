"""Add predictions.confidence.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-13

Item 06 gives every prediction a distribution, and a distribution has a width.
``confidence`` records how tight this particular prediction is relative to a
typical one -- which is the difference between projecting a well-known starter
and projecting a player with no EuroLeague box scores at all, where the model
falls back to a constant and we should say so.

Nullable rather than defaulted. A prediction written by a model trained before
the distribution work has no confidence to report, and inventing 0.5 for it
would be indistinguishable from a real measurement of average confidence.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column(
            "confidence",
            sa.Numeric(4, 3),
            nullable=True,
            comment=(
                "Relative tightness of the predicted distribution, in (0, 1]. "
                "NULL for models trained before the distribution work."
            ),
        ),
    )
    # Written as explicit DDL rather than op.create_check_constraint.
    #
    # The naming convention is "ck_%(table_name)s_%(constraint_name)s", and
    # whether Alembic applies it to a bare name here depends on how the
    # migration context was configured. Getting that wrong produces
    # "ck_predictions_ck_predictions_..." -- which is precisely the drift
    # migration 0002 exists to repair. Spelling out the final name removes the
    # question entirely, and it is the name the ORM will render for the same
    # constraint, so autogenerate stays quiet.
    op.execute(
        "ALTER TABLE predictions "
        "ADD CONSTRAINT ck_predictions_confidence_in_unit_interval "
        "CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE predictions "
        "DROP CONSTRAINT IF EXISTS ck_predictions_confidence_in_unit_interval"
    )
    op.drop_column("predictions", "confidence")

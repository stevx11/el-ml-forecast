"""Add prediction_outcomes.did_not_appear.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-13

Scoring a prediction needs three outcomes, not two. A player can produce a real
line, or dress and never be used, or not be in the squad at all. The last two
both score as zero, and lumping them in with genuine performances would blame
the points model for a coach's rotation decision -- while dropping them silently
would flatter projections we did in fact publish.

One boolean covers both non-appearance cases, so reports can include or exclude
them deliberately rather than by accident.

Server default so the column can be NOT NULL against existing rows. There are
none in practice -- nothing has been scored yet -- but a migration that only
works on an empty table is a migration that fails the first time it matters.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "prediction_outcomes",
        sa.Column(
            "did_not_appear",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
            comment="Player did not take the floor; scored as zero but excluded from MAE by default.",
        ),
    )
    # Drop the server default once the backfill is done. Keeping it would let
    # application code omit the value and silently get a false, which is exactly
    # the kind of default that hides a bug in the scoring job.
    op.alter_column("prediction_outcomes", "did_not_appear", server_default=None)


def downgrade() -> None:
    op.drop_column("prediction_outcomes", "did_not_appear")

"""Add predictions.expected_value.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-21

``predicted_value`` is the median, which is what the L1 models are trained for
and what the MAE gate judges. ``expected_value`` is the mean of the same
prediction, conditional on the player taking the floor: what he *averages* in
this spot rather than what he usually does. For skewed box-score stats the two
differ enough that summed medians fall eight points short of a real team.

Kept beside the median rather than replacing it. Scoring, intervals and
threshold probabilities are all built around the median, and swapping the
column's meaning under them would silently invalidate every stored outcome.

Nullable: a target whose expectation failed its gate, the ``PLAYS`` probability,
and every prediction written before this revision have none.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column(
            "expected_value",
            sa.Float(),
            nullable=True,
            comment="Mean outcome given the player plays. predicted_value is the median.",
        ),
    )


def downgrade() -> None:
    op.drop_column("predictions", "expected_value")

"""The versioned feature set.

One place that says what a feature vector contains and what version it is. Every
snapshot records ``feature_set_version``, so a model trained on ``v2`` can never
be served ``v1`` inputs by accident, and a change to any family is a new version
rather than a silent redefinition of existing rows.

Families:

===================  =========================================================
``form``             Rolling, weighted and trend statistics over the player's
                     own prior box scores, **crossing season boundaries**.
``role``             Starter status and per-40 usage rates.
``schedule``         Rest, congestion, recent load.
``team``             The player's team's ratings, pace and form.
``opponent``         The opposing team's ratings, pace and form.
``availability``     Teammate context inferred from prior appearances.
===================  =========================================================
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.features.availability import build_availability_features
from euroleague_ai.features.player_form import build_player_form_features
from euroleague_ai.features.team_context import build_team_context_features

#: Bump this whenever a feature's name, meaning or computation changes.
#: Existing snapshots keep their old version and stay valid for the models
#: trained on them.
#:
#: v3: rolling windows cross season boundaries. Same feature *names* as v2 but
#: different *meaning* -- ``last5_points_mean`` now spans the summer break -- so
#: v2 snapshots and v3 snapshots must never be mixed in one training set. That is
#: exactly what this version string prevents.
FEATURE_SET_VERSION = "v3"

FEATURE_FAMILIES = ("form", "role", "schedule", "team", "opponent", "availability")


async def build_player_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    opponent_team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
    is_home: bool | None = None,
) -> dict[str, float | None]:
    """Build the complete feature vector for one player-game as of one instant.

    ``as_of`` is mandatory and keyword-only. Every family underneath reads the
    database only through
    :mod:`euroleague_ai.repositories.history`, which applies the
    ``observed_at <= as_of`` predicate -- so there is no path by which a feature
    can see past the cutoff.

    Args:
        player_id: Player being predicted.
        team_id: Team they are playing for in the upcoming game.
        opponent_team_id: Team they are playing against.
        as_of: Cutoff. Nothing recorded after this instant may contribute.
        season_id: Restricts season-scoped features to one season.
        is_home: Whether the upcoming game is at home.

    Returns:
        Feature name to value. ``None`` denotes genuinely missing data and is
        passed through to the model as NaN rather than being imputed here --
        LightGBM handles missingness natively and learns a split for it, which
        is strictly better than a guess made in this function.
    """
    as_of = ensure_utc(as_of, field="as_of")

    features: dict[str, float | None] = {}

    # Form, role and schedule.
    features.update(
        await build_player_form_features(
            session, player_id=player_id, team_id=team_id, as_of=as_of, season_id=season_id
        )
    )

    # Team and opponent strength.
    features.update(
        await build_team_context_features(
            session,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            as_of=as_of,
            season_id=season_id,
            is_home=is_home,
        )
    )

    # Teammate availability, inferred from prior appearances.
    features.update(
        await build_availability_features(
            session, player_id=player_id, team_id=team_id, as_of=as_of, season_id=season_id
        )
    )

    return features


async def feature_names(session: AsyncSession) -> list[str]:
    """Return the feature vector's key names, sorted.

    Used by training code to pin column order. Derived from a real build against
    non-existent ids -- which exercises every family's empty-history branch --
    rather than a hardcoded list, so it cannot drift out of sync with what the
    builders actually emit.
    """
    absent = uuid.UUID(int=0)
    vector = await build_player_features(
        session,
        player_id=absent,
        team_id=absent,
        opponent_team_id=absent,
        as_of=datetime.now(UTC),
    )
    return sorted(vector)

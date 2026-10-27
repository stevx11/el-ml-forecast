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
``game``             Expected total, margin and blowout risk for the
                     fixture -- how the game itself is likely to unfold.
``position``         Listed position, and where the player ranks among
                     teammates competing for the same minutes.
``appearances``      (v6) Share of the team's last five games he played, and
                     whether he has debuted for the club.
``opponent_position``  (v6) What the opponent allows to his position group.
``coach``            (v6) Current coach's games in charge.
``schedule_venue``   (v6) Next fixture, postseason flag, home/away split.
===================  =========================================================
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.features.availability import SQUAD_WINDOW, summarise_squad
from euroleague_ai.features.context_v6 import build_v6_features
from euroleague_ai.features.player_form import build_player_form_features
from euroleague_ai.features.position import COMPETITION_WINDOW, summarise_position
from euroleague_ai.features.team_context import build_team_context_features
from euroleague_ai.repositories.history import TeamHistoryRepository

#: Bump this whenever a feature's name, meaning or computation changes.
#: Existing snapshots keep their old version and stay valid for the models
#: trained on them.
#:
#: v5: position and positional competition. Five guards cannot all play
#: twenty minutes; predicting each player in isolation had no way to know
#: that. The signal is the competition features -- rank and minutes share
#: within the position group -- not the one-hot flags.
#:
#: v4: game-script features -- expected total, margin and blowout risk. Added
#: because the chain experiment measured minutes MAE at 4.43 on a ~20 minute
#: mean, and nothing in the feature set knew whether a game would be close.
#:
#: v3: rolling windows cross season boundaries. Same feature *names* as v2 but
#: different *meaning* -- ``last5_points_mean`` now spans the summer break -- so
#: v2 snapshots and v3 snapshots must never be mixed in one training set. That is
#: exactly what this version string prevents.
#: v6: recent appearances, opponent defence by position, coach tenure, next
#: fixture, postseason and home/away split (features/context_v6.py). Additive
#: only: every v5 feature keeps its name and meaning, which is what lets v5
#: models be served from v6 vectors until the backfill and retrain are done.
FEATURE_SET_VERSION = "v6"

#: Which model feature sets the current builder can serve. A model picks its
#: own columns out of the vector by name, so a later set that only *adds*
#: features can serve models trained on the earlier one. A version that changes
#: any existing feature's meaning (v3 did) must not be listed here.
SERVES: dict[str, frozenset[str]] = {
    "v6": frozenset({"v5", "v6"}),
}


def can_serve(model_feature_set: str, current: str = FEATURE_SET_VERSION) -> bool:
    """Whether a model trained on ``model_feature_set`` can be served now."""
    return model_feature_set in SERVES.get(current, frozenset({current}))


FEATURE_FAMILIES = (
    "form",
    "role",
    "schedule",
    "team",
    "opponent",
    "availability",
    "game",
    "position",
    "appearances",
    "opponent_position",
    "coach",
    "schedule_venue",
)


async def build_player_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    opponent_team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
    is_home: bool | None = None,
    cache: dict[Any, Any] | None = None,
    game: Any = None,
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
        game: The game being predicted, for the schedule and phase features.
            Optional so a vector can be built without one (feature names, ad
            hoc checks); those features are then missing.
        cache: Optional per-game memo. Three of the four families -- team
            context, availability and position -- depend only on
            ``(team, opponent, as_of)`` and are therefore identical for every
            player on a side. Without a cache they are recomputed once per
            player, which is roughly a dozen redundant round trips per team per
            game and the reason a full backfill takes as long as it does.
            Callers processing a whole game should pass a fresh dict; callers
            building a single vector can leave it as ``None``.

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

    # Team and opponent strength. Shared by every player on this side, so it is
    # computed once per (team, opponent, as_of) when a cache is supplied.
    key = ("team", team_id, opponent_team_id, as_of, is_home)
    if cache is not None and key in cache:
        features.update(cache[key])
    else:
        team_features = await build_team_context_features(
            session,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            as_of=as_of,
            season_id=season_id,
            is_home=is_home,
        )
        if cache is not None:
            cache[key] = team_features
        features.update(team_features)

    # Teammate availability and positional competition. Both read the *squad*
    # rather than the player, so the queries are identical for everyone on a
    # side -- but the results are not, because each describes where this
    # particular player sits within that squad. So the cache holds the raw
    # squad data and the per-player arithmetic still runs.
    #
    # This is where the backfill's time actually went: three squad queries per
    # player, twelve players a side, for data that changes once per game.
    squad_key = ("squad", team_id, as_of, season_id)
    if cache is not None and squad_key in cache:
        appearances, competition, positions = cache[squad_key]
    else:
        teams = TeamHistoryRepository(session)
        appearances = await teams.squad_appearances(
            team_id, as_of=as_of, games=SQUAD_WINDOW, season_id=season_id
        )
        # A separate fetch, because the two families look back over different
        # windows: five games is the right horizon for "who is missing", ten for
        # "who is ahead of whom in the rotation".
        competition = (
            appearances
            if COMPETITION_WINDOW == SQUAD_WINDOW
            else await teams.squad_appearances(
                team_id, as_of=as_of, games=COMPETITION_WINDOW, season_id=season_id
            )
        )
        positions = await teams.squad_positions(
            team_id, as_of=as_of, season_id=season_id
        )
        if cache is not None:
            cache[squad_key] = (appearances, competition, positions)

    features.update(summarise_squad(appearances, player_id=player_id))
    features.update(
        summarise_position(
            player_id=player_id, positions=positions, appearances=competition
        )
    )

    features.update(
        await build_v6_features(
            session,
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            as_of=as_of,
            season_id=season_id,
            game=game,
            is_home=is_home,
            position=positions.get(player_id),
            cache=cache,
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

"""Persisting feature snapshots.

A snapshot is written once and never updated. That immutability is what lets a
prediction made in October still be explained in April with the exact inputs it
consumed, even after the underlying box scores have been corrected upstream.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock, ensure_utc
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import NotFoundError
from euroleague_ai.features.registry import FEATURE_SET_VERSION, build_player_features
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game


def default_prediction_timestamp(
    tip_off_utc: datetime, settings: Settings | None = None
) -> datetime:
    """Return the default cutoff for a fixture: ``tip_off - PREDICTION_LEAD_MINUTES``.

    Training rows and served predictions use the same offset deliberately. If
    training rows were generated closer to tip-off than serving allows, offline
    metrics would be inflated by information the live system never has, and the
    model would disappoint in exactly the way that is hardest to diagnose.
    """
    resolved = settings or get_settings()
    return ensure_utc(tip_off_utc, field="tip_off_utc") - timedelta(
        minutes=resolved.prediction_lead_minutes
    )


async def build_and_store_player_snapshot(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    game_id: uuid.UUID,
    team_id: uuid.UUID,
    prediction_timestamp: datetime | None = None,
    clock: Clock | None = None,
    settings: Settings | None = None,
) -> PlayerFeatureSnapshot:
    """Build a feature snapshot and persist it.

    Idempotent: an existing snapshot for the same
    ``(player, game, feature_set_version, prediction_timestamp)`` is returned
    unchanged rather than recomputed, preserving immutability.
    """
    clock = clock or SystemClock()

    game = (await session.execute(select(Game).where(Game.id == game_id))).scalar_one_or_none()
    if game is None:
        raise NotFoundError("Game", game_id)

    cutoff = (
        ensure_utc(prediction_timestamp, field="prediction_timestamp")
        if prediction_timestamp is not None
        else default_prediction_timestamp(game.tip_off_utc, settings)
    )

    opponent_team_id = game.away_team_id if team_id == game.home_team_id else game.home_team_id

    existing = (
        await session.execute(
            select(PlayerFeatureSnapshot).where(
                PlayerFeatureSnapshot.player_id == player_id,
                PlayerFeatureSnapshot.game_id == game_id,
                PlayerFeatureSnapshot.feature_set_version == FEATURE_SET_VERSION,
                PlayerFeatureSnapshot.prediction_timestamp == cutoff,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    features = await build_player_features(
        session,
        player_id=player_id,
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        as_of=cutoff,
        season_id=game.season_id,
        is_home=team_id == game.home_team_id,
    )

    snapshot = PlayerFeatureSnapshot(
        id=uuid.uuid4(),
        player_id=player_id,
        game_id=game_id,
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        prediction_timestamp=cutoff,
        feature_set_version=FEATURE_SET_VERSION,
        features=features,
        computed_at=clock.now(),
    )
    session.add(snapshot)
    await session.flush()
    return snapshot

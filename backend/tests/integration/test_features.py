"""Feature snapshot construction and persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.features.player_form import FEATURE_SET_VERSION
from euroleague_ai.features.snapshot import (
    build_and_store_player_snapshot,
    default_prediction_timestamp,
)
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.models.enums import EntityType
from euroleague_ai.models.game import Game
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

CLOCK = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))


async def _internal_id(
    session: AsyncSession, entity_type: EntityType, external_id: str
) -> uuid.UUID:
    stmt = select(ExternalRef.internal_id).where(
        ExternalRef.entity_type == entity_type, ExternalRef.external_id == external_id
    )
    return (await session.execute(stmt)).scalar_one()


class TestPredictionTimestamp:
    def test_default_is_the_configured_lead_before_tip_off(self) -> None:
        tip_off = datetime(2025, 10, 31, 18, 0, tzinfo=UTC)
        cutoff = default_prediction_timestamp(tip_off)
        assert cutoff < tip_off
        assert (tip_off - cutoff) <= timedelta(hours=24)

    def test_naive_tip_off_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            default_prediction_timestamp(datetime(2025, 10, 31, 18, 0))


class TestSnapshots:
    async def test_snapshot_is_persisted_with_its_cutoff(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        game = (await session.execute(select(Game).where(Game.game_code == 5))).scalar_one()
        player_id = await _internal_id(session, EntityType.PLAYER, "011001")
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")

        snapshot = await build_and_store_player_snapshot(
            session, player_id=player_id, game_id=game.id, team_id=team_id, clock=CLOCK
        )

        assert snapshot.feature_set_version == FEATURE_SET_VERSION
        assert snapshot.prediction_timestamp < game.tip_off_utc
        assert snapshot.opponent_team_id == game.away_team_id
        assert snapshot.features["season_points_mean"] == pytest.approx(13.0)

    async def test_snapshots_are_immutable_and_idempotent(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Rebuilding returns the original row rather than recomputing.

        Immutability is what lets an October prediction still be explained in
        April with the exact inputs it consumed.
        """
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        game = (await session.execute(select(Game).where(Game.game_code == 5))).scalar_one()
        player_id = await _internal_id(session, EntityType.PLAYER, "011001")
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")

        first = await build_and_store_player_snapshot(
            session, player_id=player_id, game_id=game.id, team_id=team_id, clock=CLOCK
        )
        second = await build_and_store_player_snapshot(
            session, player_id=player_id, game_id=game.id, team_id=team_id, clock=CLOCK
        )
        assert first.id == second.id

    async def test_missing_history_is_null_not_zero(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """A player with no prior games has unknown form, not zero form."""
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        game = (await session.execute(select(Game).where(Game.game_code == 1))).scalar_one()
        player_id = await _internal_id(session, EntityType.PLAYER, "011001")
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")

        snapshot = await build_and_store_player_snapshot(
            session, player_id=player_id, game_id=game.id, team_id=team_id, clock=CLOCK
        )
        assert snapshot.features["season_points_mean"] is None
        assert snapshot.features["games_played_season"] == 0

"""Database connectivity, schema integrity, and temporal invariants."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from euroleague_ai.db.base import Base
from euroleague_ai.models.competition import League, Season, Team
from euroleague_ai.models.people import Player, PlayerTeamStint
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


class TestConnectivity:
    async def test_the_database_is_reachable(self, session: AsyncSession) -> None:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1

    async def test_every_model_table_exists(self, engine: AsyncEngine) -> None:
        async with engine.connect() as connection:
            tables = await connection.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
        missing = set(Base.metadata.tables) - tables
        assert not missing, f"tables missing from the database: {sorted(missing)}"

    async def test_timestamps_round_trip_as_utc(self, session: AsyncSession) -> None:
        """Every datetime column is timestamptz; naive values must never appear."""
        league = League(id=uuid.uuid4(), code="TEST", name="Test League")
        session.add(league)
        await session.flush()
        await session.refresh(league)
        assert league.created_at.tzinfo is not None


class TestTemporalInvariants:
    async def _scaffold(self, session: AsyncSession) -> tuple[Player, Team, Season]:
        league = League(id=uuid.uuid4(), code="TL", name="Test League")
        session.add(league)
        await session.flush()

        season = Season(
            id=uuid.uuid4(), league_id=league.id, code="T2025", name="Test 2025", start_year=2025
        )
        team = Team(id=uuid.uuid4(), code="TTM", name="Test Team")
        player = Player(id=uuid.uuid4(), full_name="TEST, PLAYER")
        session.add_all([season, team, player])
        await session.flush()
        return player, team, season

    async def test_inverted_intervals_are_rejected(self, session: AsyncSession) -> None:
        """``effective_until`` before ``effective_from`` is nonsense and must not persist."""
        player, team, season = await self._scaffold(session)
        now = datetime(2025, 10, 1, tzinfo=UTC)

        session.add(
            PlayerTeamStint(
                id=uuid.uuid4(),
                player_id=player.id,
                team_id=team.id,
                season_id=season.id,
                effective_from=now,
                effective_until=now - timedelta(days=1),
                observed_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_open_intervals_are_allowed(self, session: AsyncSession) -> None:
        """A NULL upper bound means "still current"."""
        player, team, season = await self._scaffold(session)
        now = datetime(2025, 10, 1, tzinfo=UTC)

        session.add(
            PlayerTeamStint(
                id=uuid.uuid4(),
                player_id=player.id,
                team_id=team.id,
                season_id=season.id,
                effective_from=now,
                effective_until=None,
                observed_at=now,
            )
        )
        await session.flush()

        stored = (await session.execute(select(PlayerTeamStint))).scalar_one()
        assert stored.effective_until is None

    async def test_two_stints_can_coexist_for_one_player(self, session: AsyncSession) -> None:
        """A transfer appends; it does not overwrite.

        If the schema forced one row per player, ingesting a January move would
        destroy the record of where they played in December.
        """
        player, team, season = await self._scaffold(session)
        other = Team(id=uuid.uuid4(), code="OTH", name="Other Team")
        session.add(other)
        await session.flush()

        start = datetime(2025, 9, 1, tzinfo=UTC)
        session.add_all(
            [
                PlayerTeamStint(
                    id=uuid.uuid4(),
                    player_id=player.id,
                    team_id=team.id,
                    season_id=season.id,
                    effective_from=start,
                    effective_until=start + timedelta(days=100),
                    observed_at=start,
                ),
                PlayerTeamStint(
                    id=uuid.uuid4(),
                    player_id=player.id,
                    team_id=other.id,
                    season_id=season.id,
                    effective_from=start + timedelta(days=101),
                    effective_until=None,
                    observed_at=start + timedelta(days=101),
                ),
            ]
        )
        await session.flush()

        stints = (await session.execute(select(PlayerTeamStint))).scalars().all()
        assert len(stints) == 2


class TestConstraints:
    async def test_league_codes_are_unique(self, session: AsyncSession) -> None:
        session.add_all(
            [
                League(id=uuid.uuid4(), code="DUP", name="First"),
                League(id=uuid.uuid4(), code="DUP", name="Second"),
            ]
        )
        with pytest.raises(IntegrityError):
            await session.flush()

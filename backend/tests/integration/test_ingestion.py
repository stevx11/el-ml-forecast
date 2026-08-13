"""End-to-end ingestion against the offline fixture provider.

The fixture provider satisfies the same protocol as the live adapter, so this
exercises identity resolution, temporal upserts, and derived statistics with no
network access and no flakiness.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.core.errors import ProviderError
from euroleague_ai.ingestion.jobs import (
    MAX_CONSECUTIVE_FAILURES,
    ingest_all,
    ingest_game_stats,
    ingest_games,
    ingest_players,
    ingest_teams,
)
from euroleague_ai.models.competition import Team, TeamSeason
from euroleague_ai.models.enums import GameStatus, IngestionStatus, TeamSide
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.models.ingestion import IngestionRun
from euroleague_ai.models.people import Player, PlayerTeamStint
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

CLOCK = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))


class TestFullIngestion:
    async def test_ingests_the_whole_fixture_season(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        assert (await session.execute(select(func.count(Team.id)))).scalar_one() == 3
        assert (await session.execute(select(func.count(Player.id)))).scalar_one() == 8
        assert (await session.execute(select(func.count(Game.id)))).scalar_one() == 6

        # Five games have box scores; game 5 is still scheduled.
        finished = (
            await session.execute(
                select(func.count(Game.id)).where(Game.status == GameStatus.FINISHED)
            )
        ).scalar_one()
        assert finished == 5

    async def test_is_idempotent(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Re-running must not duplicate anything.

        Nightly ingestion re-reads the same season every time; if that grew the
        tables, every form average would silently double-count.
        """
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()
        before = {
            "teams": (await session.execute(select(func.count(Team.id)))).scalar_one(),
            "players": (await session.execute(select(func.count(Player.id)))).scalar_one(),
            "games": (await session.execute(select(func.count(Game.id)))).scalar_one(),
            "stats": (
                await session.execute(select(func.count(PlayerGameStats.id)))
            ).scalar_one(),
            "stints": (
                await session.execute(select(func.count(PlayerTeamStint.id)))
            ).scalar_one(),
        }

        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        after = {
            "teams": (await session.execute(select(func.count(Team.id)))).scalar_one(),
            "players": (await session.execute(select(func.count(Player.id)))).scalar_one(),
            "games": (await session.execute(select(func.count(Game.id)))).scalar_one(),
            "stats": (
                await session.execute(select(func.count(PlayerGameStats.id)))
            ).scalar_one(),
            "stints": (
                await session.execute(select(func.count(PlayerTeamStint.id)))
            ).scalar_one(),
        }
        assert before == after


class TestIdentityResolution:
    async def test_external_refs_are_registered(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        refs = (await session.execute(select(ExternalRef))).scalars().all()
        assert refs
        assert all(ref.provider_key == "fixture" for ref in refs)

    async def test_one_player_appears_once_despite_two_stints(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """The transferred player is one person with two registrations."""
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        players = (
            await session.execute(select(Player).where(Player.last_name == "NIKOLIC"))
        ).scalars().all()
        assert len(players) == 1

        stints = (
            await session.execute(
                select(PlayerTeamStint).where(PlayerTeamStint.player_id == players[0].id)
            )
        ).scalars().all()
        assert len(stints) == 2


class TestSeasonNaming:
    async def test_sponsored_name_lives_on_the_season_registration(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Club identity is stable; the sponsored name is season-scoped."""
        await ingest_teams(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        team = (
            await session.execute(select(Team).where(Team.code == "BAS"))
        ).scalar_one()
        assert team.name == "Saski Baskonia"

        registration = (
            await session.execute(
                select(TeamSeason).where(TeamSeason.team_id == team.id)
            )
        ).scalar_one()
        assert registration.display_name == "Saski Baskonia"


class TestGames:
    async def test_each_game_has_exactly_two_sides(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        rows = (
            await session.execute(
                select(GameTeam.game_id, func.count(GameTeam.id)).group_by(GameTeam.game_id)
            )
        ).all()
        assert rows
        assert all(count == 2 for _, count in rows)

        sides = (
            await session.execute(
                select(GameTeam.side, func.count(GameTeam.id)).group_by(GameTeam.side)
            )
        ).all()
        assert dict(sides) == {TeamSide.HOME: 6, TeamSide.AWAY: 6}

    async def test_rescheduling_updates_in_place(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """A postponement changes the fixture; it does not create a second one."""
        await ingest_games(session, fixture_provider, season, clock=CLOCK)
        await session.flush()
        first = (await session.execute(select(func.count(Game.id)))).scalar_one()

        await ingest_games(session, fixture_provider, season, clock=CLOCK)
        await session.flush()
        assert (await session.execute(select(func.count(Game.id)))).scalar_one() == first


class TestDerivedTeamStats:
    async def test_possessions_and_pace_are_computed_at_ingest(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Stored so opponent features need not re-aggregate box scores."""
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        rows = (await session.execute(select(TeamGameStats))).scalars().all()
        assert rows
        for row in rows:
            assert row.possessions is not None and row.possessions > 0
            assert row.pace is not None and row.pace > 0


class TestObservedAt:
    async def test_every_ingested_row_is_stamped_from_the_injected_clock(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """One ingestion run means one coherent observation time.

        Stamping each row with ``now()`` would spread a single run across
        milliseconds and make ``observed_at <= as_of`` boundary behaviour
        untestable.
        """
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        for stats in (await session.execute(select(PlayerGameStats))).scalars().all():
            assert stats.observed_at == CLOCK.now()
        for game in (await session.execute(select(Game))).scalars().all():
            assert game.observed_at == CLOCK.now()


class _FailingProvider:
    """Wraps a provider and fails box score fetches for chosen games.

    Models what the live API actually does under load: most requests succeed,
    some return 429, and the failures are not evenly distributed.
    """

    def __init__(self, inner: FixtureProvider, fail_codes: set[int]) -> None:
        self._inner = inner
        self._fail_codes = fail_codes
        self.attempted: list[int] = []

    provider_key = "fixture"

    async def fetch_teams(self, season: SeasonRef):  # type: ignore[no-untyped-def]
        return await self._inner.fetch_teams(season)

    async def fetch_players(self, season: SeasonRef):  # type: ignore[no-untyped-def]
        return await self._inner.fetch_players(season)

    async def fetch_games(self, season: SeasonRef, *, round_number: int | None = None):  # type: ignore[no-untyped-def]
        return await self._inner.fetch_games(season, round_number=round_number)

    async def fetch_game_stats(self, season: SeasonRef, game_code: int):  # type: ignore[no-untyped-def]
        self.attempted.append(game_code)
        if game_code in self._fail_codes:
            raise ProviderError("fixture", f"rate limited on game {game_code}")
        return await self._inner.fetch_game_stats(season, game_code)

    async def aclose(self) -> None:
        await self._inner.aclose()


class TestBoxScoreResilience:
    """A season backfill is hundreds of rate-limited requests. Something fails."""

    async def test_one_failed_game_does_not_abort_the_run(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Regression: a single 429 used to roll back the entire backfill.

        Every box score fetched before the failure was lost, because the whole
        job ran in one transaction and the exception propagated out of it.
        """
        await ingest_teams(session, fixture_provider, season, clock=CLOCK)
        await ingest_players(session, fixture_provider, season, clock=CLOCK)
        await ingest_games(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        flaky = _FailingProvider(fixture_provider, fail_codes={2})
        result = await ingest_game_stats(session, flaky, season, clock=CLOCK)  # type: ignore[arg-type]
        await session.flush()

        assert result.details.get("failed") == 1
        # The other four games were still written.
        written = (
            await session.execute(select(func.count(PlayerGameStats.id)))
        ).scalar_one()
        assert written > 0

        stats_by_game = (
            await session.execute(
                select(func.count(func.distinct(GameTeam.game_id)))
                .select_from(PlayerGameStats)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            )
        ).scalar_one()
        assert stats_by_game == 4

    async def test_rerunning_resumes_and_retries_only_what_is_missing(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Resume is what makes a 400-game backfill practical.

        Re-fetching everything after a partial run would trip the same rate
        limit at the same point, forever.
        """
        await ingest_teams(session, fixture_provider, season, clock=CLOCK)
        await ingest_players(session, fixture_provider, season, clock=CLOCK)
        await ingest_games(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        flaky = _FailingProvider(fixture_provider, fail_codes={2})
        await ingest_game_stats(session, flaky, season, clock=CLOCK)  # type: ignore[arg-type]
        await session.flush()

        # Second pass: the API has recovered.
        healthy = _FailingProvider(fixture_provider, fail_codes=set())
        result = await ingest_game_stats(session, healthy, season, clock=CLOCK)  # type: ignore[arg-type]
        await session.flush()

        # Only the previously-failed game is retried.
        assert healthy.attempted == [2]
        assert result.details.get("failed") is None

        stats_by_game = (
            await session.execute(
                select(func.count(func.distinct(GameTeam.game_id)))
                .select_from(PlayerGameStats)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            )
        ).scalar_one()
        assert stats_by_game == 5

    async def test_a_completed_run_has_nothing_left_to_do(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        healthy = _FailingProvider(fixture_provider, fail_codes=set())
        await ingest_game_stats(session, healthy, season, clock=CLOCK)  # type: ignore[arg-type]
        assert healthy.attempted == []

    async def test_refresh_forces_a_refetch(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """After an upstream stat correction you do want everything again."""
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        healthy = _FailingProvider(fixture_provider, fail_codes=set())
        await ingest_game_stats(
            session, healthy, season, refresh_existing=True, clock=CLOCK  # type: ignore[arg-type]
        )
        assert sorted(healthy.attempted) == [1, 2, 3, 4, 6]

    async def test_circuit_breaker_stops_a_doomed_run(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """When the API is down, stop rather than grinding through the schedule."""
        await ingest_teams(session, fixture_provider, season, clock=CLOCK)
        await ingest_players(session, fixture_provider, season, clock=CLOCK)
        await ingest_games(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        dead = _FailingProvider(fixture_provider, fail_codes={1, 2, 3, 4, 6})
        result = await ingest_game_stats(session, dead, season, clock=CLOCK)  # type: ignore[arg-type]

        assert result.rows_written == 0
        assert result.details.get("failed") == len(dead.attempted)
        # The fixture season is smaller than the breaker threshold, so every game
        # is attempted here; the guarantee under test is that the job returns
        # instead of raising.
        assert len(dead.attempted) <= MAX_CONSECUTIVE_FAILURES


class TestAuditTrail:
    async def test_runs_are_recorded_with_counts(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await ingest_all(session, fixture_provider, season, clock=CLOCK)
        await session.flush()

        runs = (await session.execute(select(IngestionRun))).scalars().all()
        assert {run.job_name for run in runs} == {"teams", "players", "games", "game_stats"}
        assert all(run.status is IngestionStatus.SUCCEEDED for run in runs)
        assert all(run.finished_at is not None for run in runs)
        assert any(run.rows_written > 0 for run in runs)

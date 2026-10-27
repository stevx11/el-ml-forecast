"""Feature set v6 through the real schema: values, and the leaks each family must not have.

Fixture calendar (Partizan): game 1 Oct 3 home v Madrid, 2 Oct 10 at Baskonia,
3 Oct 17 home v Baskonia, 4 Oct 24 at Madrid, game 5 Oct 31 home v Madrid (the
one predicted, cutoff 17:00 UTC), game 6 Nov 7. Petrovic, a guard, scored 10,
12, 14 and 16 in games 1-4.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select, update

from euroleague_ai.features import context_v6
from euroleague_ai.features.context_v6 import V6_FEATURE_NAMES
from euroleague_ai.features.registry import SERVES, build_player_features, can_serve
from euroleague_ai.models.enums import EntityType
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.people import Coach
from tests.conftest import requires_db
from tests.integration.test_no_leakage import CUTOFF, PLAYER_CODE, _internal_id, _season_id, _seed

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

LATE = CUTOFF + timedelta(hours=1)


async def _game(session: AsyncSession, code: int) -> Game:
    return (await session.execute(select(Game).where(Game.game_code == code))).scalar_one()


async def _vector(session: AsyncSession) -> dict[str, float | None]:
    return await build_player_features(
        session,
        player_id=await _internal_id(session, EntityType.PLAYER, PLAYER_CODE),
        team_id=await _internal_id(session, EntityType.TEAM, "PAR"),
        opponent_team_id=await _internal_id(session, EntityType.TEAM, "MAD"),
        as_of=CUTOFF,
        season_id=await _season_id(session),
        is_home=True,
        game=await _game(session, 5),
    )


async def _arrive_late(session: AsyncSession, code: int) -> None:
    """Game ``code`` tipped off before the cutoff, but its box score arrived after."""
    game = await _game(session, code)
    sides = select(GameTeam.id).where(GameTeam.game_id == game.id)
    await session.execute(
        update(PlayerGameStats)
        .where(PlayerGameStats.game_team_id.in_(sides))
        .values(observed_at=LATE)
    )
    await session.execute(
        update(TeamGameStats).where(TeamGameStats.game_team_id.in_(sides)).values(observed_at=LATE)
    )
    await session.flush()


class TestValues:
    async def test_every_family_at_the_cutoff(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)
        v = await _vector(session)
        assert set(V6_FEATURE_NAMES) <= set(v)
        # Appearances: played all four of Partizan's games.
        assert (v["recent_played_share"], v["recent_team_games"], v["not_debuted_for_club"]) == (
            1.0,
            4.0,
            0.0,
        )
        # Madrid allowed guards 10 (game 1) and 16 (game 4): Petrovic is the guard.
        assert v["opp_pos_points_pg"] == pytest.approx(13.0)
        assert 0.0 < v["opp_pos_points_share"] < 1.0  # type: ignore[operator]
        # One coach for all four games.
        assert (v["coach_games_in_charge"], v["coach_changed_recently"]) == (4.0, 0.0)
        # Game 6 is a week later; regular season.
        assert v["days_to_next_team_game"] == pytest.approx(7.0)
        assert (v["next_game_within_3d"], v["is_postseason"]) == (0.0, 0.0)
        # Four prior games: too few for a home/away split.
        assert v["venue_points_diff"] is None

    async def test_home_split_once_there_are_enough_games(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(context_v6, "VENUE_MIN_GAMES", 2)
        await _seed(session, fixture_provider, season)
        v = await _vector(session)
        # Home: 10 and 14 (mean 12); all four: mean 13.
        assert v["venue_points_diff"] == pytest.approx(-1.0)


class TestLeaks:
    async def test_late_box_scores_do_not_count_for_opponent_defence_or_coach(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)
        await _arrive_late(session, 4)
        v = await _vector(session)
        assert v["opp_pos_points_pg"] == pytest.approx(10.0)  # game 1 only
        assert v["coach_games_in_charge"] == 3.0

    async def test_a_fixture_learned_after_the_cutoff_is_not_the_next_game(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)
        await session.execute(update(Game).where(Game.game_code == 6).values(observed_at=LATE))
        await session.flush()
        v = await _vector(session)
        assert v["days_to_next_team_game"] is None
        assert v["next_game_within_3d"] == 0.0

    async def test_a_coach_change_shows_up(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)
        new_coach = Coach(id=uuid.uuid4(), full_name="NEW, COACH")
        session.add(new_coach)
        partizan = await _internal_id(session, EntityType.TEAM, "PAR")
        game4 = await _game(session, 4)
        await session.execute(
            update(GameTeam)
            .where(GameTeam.game_id == game4.id, GameTeam.team_id == partizan)
            .values(coach_id=new_coach.id)
        )
        await session.flush()
        v = await _vector(session)
        assert (v["coach_games_in_charge"], v["coach_changed_recently"]) == (1.0, 1.0)


class TestServing:
    def test_v6_serves_v5_models_and_nothing_older(self) -> None:
        assert can_serve("v5") and can_serve("v6")
        assert not can_serve("v4")
        assert SERVES["v6"] == frozenset({"v5", "v6"})

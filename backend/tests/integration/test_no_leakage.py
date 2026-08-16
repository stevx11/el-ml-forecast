"""Data leakage guards.

**The most important tests in this repository.**

Leakage is the failure mode that produces excellent offline metrics and a useless
product, and it does so silently. These tests seed a world in which post-cutoff
rows carry deliberately extreme values, build feature snapshots at the cutoff,
and assert the extreme values never appear. A leak changes the numbers and fails
the test.

Two distinct leaks are covered, because they have different causes and either one
alone would be insufficient:

1. **Future games** -- a game that tips off after the cutoff. Guarded by the
   ``tip_off_utc < as_of`` predicate.
2. **Late-arriving knowledge** -- a game that tipped off *before* the cutoff but
   was recorded *after* it. Guarded by ``observed_at <= as_of``. This is the
   subtle one: filtering on event time alone looks correct and leaks anyway.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.features.availability import build_availability_features
from euroleague_ai.features.player_form import build_player_form_features
from euroleague_ai.features.registry import FEATURE_SET_VERSION, build_player_features
from euroleague_ai.features.team_context import build_team_context_features
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.competition import Season
from euroleague_ai.models.enums import (
    AvailabilityStatus,
    EntityType,
    GameStatus,
    StatusSourceType,
    TeamSide,
)
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from euroleague_ai.repositories.history import PlayerHistoryRepository
from tests.conftest import SEASON, requires_db

pytestmark = [pytest.mark.integration, requires_db]

# Marko Petrovic scored 10, 12, 14, 16 in the four games before the cutoff, and
# 60 in game 6, which tips off a week afterwards.
PLAYER_CODE = "011001"
PRE_CUTOFF_POINTS = [10, 12, 14, 16]
POST_CUTOFF_POINTS = 60

# Game 5 tips off at 18:00 UTC; the default lead is 60 minutes.
SEASON_CODE = SEASON.season_code

TARGET_TIP_OFF = datetime(2025, 10, 31, 18, 0, tzinfo=UTC)
CUTOFF = TARGET_TIP_OFF - timedelta(minutes=60)


async def _seed(session: AsyncSession, provider: FixtureProvider, season: SeasonRef) -> None:
    await ingest_all(
        session, provider, season, clock=FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))
    )
    await session.flush()


async def _internal_id(
    session: AsyncSession, entity_type: EntityType, external_id: str
) -> uuid.UUID:
    stmt = select(ExternalRef.internal_id).where(
        ExternalRef.entity_type == entity_type, ExternalRef.external_id == external_id
    )
    return (await session.execute(stmt)).scalar_one()


async def _season_id(session: AsyncSession, code: str = SEASON_CODE) -> uuid.UUID:
    return (await session.execute(select(Season.id).where(Season.code == code))).scalar_one()


async def _features(
    session: AsyncSession, *, as_of: datetime, season_id: uuid.UUID | None = None
) -> dict[str, float | None]:
    """Build Petrovic's form features.

    ``season_id`` defaults to the fixture season rather than to ``None``, because
    ``None`` means "do not scope" -- which makes ``season_*`` features quietly
    read the player's whole career and would let a cross-season test pass for the
    wrong reason. Production always passes a season id.
    """
    player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
    team_id = await _internal_id(session, EntityType.TEAM, "PAR")
    return await build_player_form_features(
        session,
        player_id=player_id,
        team_id=team_id,
        as_of=as_of,
        season_id=season_id if season_id is not None else await _season_id(session),
    )


class TestFutureGamesAreInvisible:
    async def test_pre_cutoff_average_excludes_the_future_spike(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """The 60-point game happens a week later and must not be averaged in.

        Correct: mean(10, 12, 14, 16) = 13.
        Leaked:  mean(10, 12, 14, 16, 60) = 22.4.
        """
        await _seed(session, fixture_provider, season)
        features = await _features(session, as_of=CUTOFF)

        expected = sum(PRE_CUTOFF_POINTS) / len(PRE_CUTOFF_POINTS)
        assert features["season_points_mean"] == pytest.approx(expected)
        assert features["games_played_season"] == pytest.approx(len(PRE_CUTOFF_POINTS))

    async def test_the_spike_appears_once_it_is_in_the_past(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """The guard must filter by time, not simply discard the row.

        Without this, a builder that always dropped the newest game would pass
        the test above while being badly wrong.
        """
        await _seed(session, fixture_provider, season)
        later = datetime(2025, 11, 30, 12, 0, tzinfo=UTC)
        features = await _features(session, as_of=later)

        all_points = [*PRE_CUTOFF_POINTS, POST_CUTOFF_POINTS]
        assert features["season_points_mean"] == pytest.approx(sum(all_points) / len(all_points))
        assert features["games_played_season"] == pytest.approx(len(all_points))

    async def test_windowed_and_derived_features_also_respect_the_cutoff(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Every derived statistic, not just the season mean."""
        await _seed(session, fixture_provider, season)
        features = await _features(session, as_of=CUTOFF)

        # Newest-first, the pre-cutoff series is [16, 14, 12, 10].
        assert features["last5_points_mean"] == pytest.approx(13.0)
        assert features["last3_points_mean"] == pytest.approx(14.0)

        # A weighted mean including a 60-point game could not be below 20.
        ewma = features["ewma_points"]
        assert ewma is not None and ewma < 20.0

        # Steadily improving form reads positive.
        trend = features["trend10_points"]
        assert trend is not None and trend > 0


class TestLateArrivingKnowledgeIsInvisible:
    """The subtle leak: the event happened first, but we learned about it later."""

    async def test_a_box_score_recorded_after_the_cutoff_is_not_visible(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Backdate the future game, but stamp it as observed after the cutoff.

        Game 6 now *tipped off* before the cutoff, so a builder filtering only on
        ``tip_off_utc`` would happily include its 60-point line. Only the
        ``observed_at <= as_of`` predicate keeps it out.
        """
        await _seed(session, fixture_provider, season)

        game = (
            await session.execute(select(Game).where(Game.game_code == 6))
        ).scalar_one()
        game.tip_off_utc = CUTOFF - timedelta(days=1)

        # ...but our system did not learn of it until an hour after the cutoff.
        late = CUTOFF + timedelta(hours=1)
        game.observed_at = late

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        stats = (
            await session.execute(
                select(PlayerGameStats)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                .where(GameTeam.game_id == game.id, PlayerGameStats.player_id == player_id)
            )
        ).scalar_one()
        stats.observed_at = late
        await session.flush()

        features = await _features(session, as_of=CUTOFF)
        expected = sum(PRE_CUTOFF_POINTS) / len(PRE_CUTOFF_POINTS)
        assert features["season_points_mean"] == pytest.approx(expected)

        # And it becomes visible once the cutoff moves past the moment we learned it.
        after = await _features(session, as_of=late + timedelta(minutes=1))
        assert after["season_points_mean"] != pytest.approx(expected)


class TestRollingWindowsCrossSeasons:
    """v3: last-N windows span the summer break; season aggregates do not.

    The motivating bug: at round one of a new season every season-scoped feature
    was null, so every player received an identical vector and the model returned
    the same 14.5 minutes and 5.2 points for a franchise scorer and a third-string
    guard. A player who averaged 20 a game in May is not an unknown quantity in
    October.

    The risk in fixing it is that widening the window widens the leak surface, so
    each test here pairs a "the past is now visible" assertion with a "the future
    still is not" assertion.
    """

    async def _add_prior_season_game(
        self,
        session: AsyncSession,
        *,
        points: int,
        tip_off: datetime,
        observed_at: datetime,
    ) -> None:
        """Give Petrovic one game in the season before the fixture season."""
        current = (
            await session.execute(select(Season).where(Season.code == SEASON_CODE))
        ).scalar_one()

        prior = Season(
            id=uuid.uuid4(),
            league_id=current.league_id,
            code="E2024",
            name="Prior season",
            start_year=current.start_year - 1,
        )
        session.add(prior)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        partizan = await _internal_id(session, EntityType.TEAM, "PAR")
        madrid = await _internal_id(session, EntityType.TEAM, "MAD")

        game = Game(
            id=uuid.uuid4(),
            season_id=prior.id,
            game_code=901,
            round_number=30,
            tip_off_utc=tip_off,
            status=GameStatus.FINISHED,
            home_team_id=partizan,
            away_team_id=madrid,
            home_score=80,
            away_score=75,
            observed_at=observed_at,
        )
        session.add(game)

        home = GameTeam(
            id=uuid.uuid4(), game_id=game.id, team_id=partizan, side=TeamSide.HOME, score=80
        )
        away = GameTeam(
            id=uuid.uuid4(), game_id=game.id, team_id=madrid, side=TeamSide.AWAY, score=75
        )
        session.add_all([home, away])

        session.add(
            PlayerGameStats(
                id=uuid.uuid4(),
                game_team_id=home.id,
                player_id=player_id,
                seconds_played=1800,
                is_starter=True,
                points=points,
                fg2m=points // 2,
                fg2a=points // 2 + 2,
                total_rebounds=5,
                assists=4,
                observed_at=observed_at,
            )
        )
        await session.flush()

    async def test_last_season_counts_toward_the_rolling_window(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """A game from the previous May is part of "the last five games"."""
        await _seed(session, fixture_provider, season)
        await self._add_prior_season_game(
            session,
            points=40,
            tip_off=datetime(2025, 5, 20, 18, 0, tzinfo=UTC),
            observed_at=datetime(2025, 5, 20, 22, 0, tzinfo=UTC),
        )

        features = await _features(session, as_of=CUTOFF)

        # Four current-season games (10, 12, 14, 16) plus one prior-season 40.
        assert features["last5_points_mean"] == pytest.approx((10 + 12 + 14 + 16 + 40) / 5)
        assert features["games_played_career_window"] == pytest.approx(5.0)

        # ...while the season aggregate stays scoped to this season. Both
        # quantities are legitimate and the model gets to see both.
        assert features["season_points_mean"] == pytest.approx(13.0)
        assert features["games_played_season"] == pytest.approx(4.0)

        # And the mix feature says how much of the window is same-season, which
        # is how the model learns when to trust which.
        assert features["career_window_is_current_season"] == pytest.approx(0.8)

    async def test_a_player_with_no_current_season_games_is_not_blank(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """The actual bug being fixed: round one is no longer a null vector."""
        await _seed(session, fixture_provider, season)
        await self._add_prior_season_game(
            session,
            points=40,
            tip_off=datetime(2025, 5, 20, 18, 0, tzinfo=UTC),
            observed_at=datetime(2025, 5, 20, 22, 0, tzinfo=UTC),
        )

        # Before any game of the new season has been played.
        preseason = datetime(2025, 9, 15, 12, 0, tzinfo=UTC)
        features = await _features(session, as_of=preseason)

        assert features["games_played_season"] == pytest.approx(0.0)
        assert features["season_points_mean"] is None

        # Under v2 every one of these was None, which is precisely why every
        # round-one projection came out identical.
        assert features["last5_points_mean"] == pytest.approx(40.0)
        assert features["ewma_points"] == pytest.approx(40.0)
        assert features["career_window_is_current_season"] == pytest.approx(0.0)

    async def test_widening_the_window_did_not_widen_the_leak(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """A prior-season game we had not yet recorded is still invisible.

        Crossing seasons must not become an excuse to bypass ``observed_at``.
        Here the game was played long before the cutoff but only entered our
        database an hour after it, so it must not appear -- exactly as it would
        not for a same-season game.
        """
        await _seed(session, fixture_provider, season)
        await self._add_prior_season_game(
            session,
            points=40,
            tip_off=datetime(2025, 5, 20, 18, 0, tzinfo=UTC),
            observed_at=CUTOFF + timedelta(hours=1),
        )

        features = await _features(session, as_of=CUTOFF)
        assert features["last5_points_mean"] == pytest.approx(13.0)
        assert features["games_played_career_window"] == pytest.approx(4.0)

        later = await _features(session, as_of=CUTOFF + timedelta(hours=2))
        assert later["games_played_career_window"] == pytest.approx(5.0)


class TestAvailabilityHistory:
    """The worked example from docs/data-model.md, asserted."""

    async def test_status_resolves_to_what_was_known_at_the_time(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """10:00 OUT, 14:00 QUESTIONABLE, 18:30 AVAILABLE.

        Asked at 17:00, the answer is QUESTIONABLE. The 18:30 upgrade exists in
        the table but was not knowable, and a model predicting at 17:00 must not
        see it.
        """
        await _seed(session, fixture_provider, season)
        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        day = datetime(2025, 10, 31, tzinfo=UTC)

        for hour, minute, status in (
            (10, 0, AvailabilityStatus.OUT),
            (14, 0, AvailabilityStatus.QUESTIONABLE),
            (18, 30, AvailabilityStatus.AVAILABLE),
        ):
            moment = day.replace(hour=hour, minute=minute)
            session.add(
                PlayerStatusEvent(
                    id=uuid.uuid4(),
                    player_id=player_id,
                    status=status,
                    source_type=StatusSourceType.PROVIDER,
                    event_timestamp=moment,
                    # Learned a few minutes after it happened, as in reality.
                    observed_at=moment + timedelta(minutes=5),
                )
            )
        await session.flush()

        repo = PlayerHistoryRepository(session)
        assert await repo.status_as_of(player_id, as_of=day.replace(hour=11)) is (
            AvailabilityStatus.OUT
        )
        assert await repo.status_as_of(player_id, as_of=day.replace(hour=17)) is (
            AvailabilityStatus.QUESTIONABLE
        )
        assert await repo.status_as_of(player_id, as_of=day.replace(hour=19)) is (
            AvailabilityStatus.AVAILABLE
        )

    async def test_unknown_is_not_available(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Absence of evidence is not evidence of availability."""
        await _seed(session, fixture_provider, season)
        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        status = await PlayerHistoryRepository(session).status_as_of(player_id, as_of=CUTOFF)
        assert status is AvailabilityStatus.UNKNOWN


class TestTransfersDoNotRewriteHistory:
    async def test_team_membership_is_resolved_as_of_a_date(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Nikolic moved to Baskonia in December.

        Asking which club he belonged to in November must still answer Partizan.
        A mutable ``current_team_id`` column would answer Baskonia and quietly
        corrupt every November feature row.
        """
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, "011003")
        partizan = await _internal_id(session, EntityType.TEAM, "PAR")
        baskonia = await _internal_id(session, EntityType.TEAM, "BAS")

        repo = PlayerHistoryRepository(session)
        november = datetime(2025, 11, 15, 12, 0, tzinfo=UTC)
        january = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

        assert await repo.team_as_of(player_id, as_of=november) == partizan
        assert await repo.team_as_of(player_id, as_of=january) == baskonia


class TestOpponentFeaturesRespectTheCutoff:
    """The classic leak: reusing a season-long opponent rating for every game.

    Computed once and applied everywhere, a team's season defensive rating
    encodes March form into an October prediction. These tests assert the
    ratings move as the cutoff moves, which they cannot do if they were
    precomputed.
    """

    async def test_opponent_rating_uses_only_prior_games(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")
        opponent_id = await _internal_id(session, EntityType.TEAM, "MAD")

        early = await build_team_context_features(
            session,
            team_id=team_id,
            opponent_team_id=opponent_id,
            as_of=datetime(2025, 10, 11, 12, 0, tzinfo=UTC),
        )
        late = await build_team_context_features(
            session, team_id=team_id, opponent_team_id=opponent_id, as_of=CUTOFF
        )

        # Real Madrid had played once by 11 October and twice by the cutoff.
        assert early["opp_games_played"] == 1.0
        assert late["opp_games_played"] == 2.0
        assert player_id is not None  # sanity: fixture wiring resolved

    async def test_ratings_ignore_games_after_the_cutoff(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Game 6 is a 99-70 blowout a week after the target game."""
        await _seed(session, fixture_provider, season)

        team_id = await _internal_id(session, EntityType.TEAM, "PAR")
        opponent_id = await _internal_id(session, EntityType.TEAM, "MAD")

        at_cutoff = await build_team_context_features(
            session, team_id=team_id, opponent_team_id=opponent_id, as_of=CUTOFF
        )
        after = await build_team_context_features(
            session,
            team_id=team_id,
            opponent_team_id=opponent_id,
            as_of=datetime(2025, 11, 30, tzinfo=UTC),
        )

        assert at_cutoff["team_games_played"] == 4.0
        assert after["team_games_played"] == 5.0
        assert at_cutoff["team_off_rating"] != after["team_off_rating"]

    async def test_naive_cutoff_rejected(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")
        with pytest.raises(ValueError, match="timezone-aware"):
            await build_team_context_features(
                session,
                team_id=team_id,
                opponent_team_id=team_id,
                as_of=datetime(2025, 10, 31, 17, 0),
            )


class TestAvailabilityProxyRespectsTheCutoff:
    async def test_squad_is_read_only_from_prior_games(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """The proxy must not peek at who played in the game being predicted."""
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")

        features = await build_availability_features(
            session, player_id=player_id, team_id=team_id, as_of=CUTOFF
        )
        # Four Partizan games precede the cutoff; the window caps at five.
        assert features["squad_games_observed"] == 4.0
        assert features["player_played_last_game"] == 1.0

    async def test_window_grows_only_as_time_passes(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")

        early = await build_availability_features(
            session,
            player_id=player_id,
            team_id=team_id,
            as_of=datetime(2025, 10, 11, 12, 0, tzinfo=UTC),
        )
        assert early["squad_games_observed"] == 2.0


class TestComposedFeatureSet:
    async def test_v2_vector_contains_every_family(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")
        opponent_id = await _internal_id(session, EntityType.TEAM, "MAD")

        vector = await build_player_features(
            session,
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_id,
            as_of=CUTOFF,
            season_id=await _season_id(session),
            is_home=True,
        )

        for marker in (
            "season_points_mean",       # form
            "starter_last_game",        # role
            "days_rest",                # schedule
            "team_off_rating",          # team
            "opp_def_rating",           # opponent
            "teammate_minutes_absent",  # availability
        ):
            assert marker in vector, f"missing family marker: {marker}"

        assert vector["season_points_mean"] == pytest.approx(13.0)
        assert FEATURE_SET_VERSION == "v5"

        # v3 exposes how much of the rolling window is same-season. The fixture
        # world is a single season, so it is entirely current.
        assert vector["career_window_is_current_season"] == pytest.approx(1.0)

    async def test_the_whole_vector_is_leak_free(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """Composition must not reintroduce what the families each avoid.

        Built at the cutoff and again after the 60-point blowout: any feature
        that is identical across both either ignores time entirely or was
        computed from data it should not have seen. Here we assert the opposite
        direction -- the ones that *should* move, do.
        """
        await _seed(session, fixture_provider, season)

        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)
        team_id = await _internal_id(session, EntityType.TEAM, "PAR")
        opponent_id = await _internal_id(session, EntityType.TEAM, "MAD")

        kwargs = {
            "player_id": player_id,
            "team_id": team_id,
            "opponent_team_id": opponent_id,
            "is_home": True,
        }
        before = await build_player_features(session, as_of=CUTOFF, **kwargs)
        after = await build_player_features(
            session, as_of=datetime(2025, 11, 30, tzinfo=UTC), **kwargs
        )

        assert before["season_points_mean"] == pytest.approx(13.0)
        assert after["season_points_mean"] == pytest.approx(22.4)
        assert before["games_played_season"] == 4.0
        assert after["games_played_season"] == 5.0


class TestTheCutoffIsMandatory:
    async def test_naive_cutoff_is_rejected(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        """A naive cutoff is a bug, and must fail loudly rather than compare wrongly."""
        await _seed(session, fixture_provider, season)
        player_id = await _internal_id(session, EntityType.PLAYER, PLAYER_CODE)

        with pytest.raises(ValueError, match="timezone-aware"):
            await PlayerHistoryRepository(session).recent_games(
                player_id, as_of=datetime(2025, 10, 31, 17, 0)
            )

    def test_builders_have_no_default_cutoff(self) -> None:
        """``as_of`` must be keyword-only with no default.

        A default would let a caller omit it and silently get "now", which for a
        historical backtest means reading the entire future.
        """
        import inspect

        signature = inspect.signature(build_player_form_features)
        as_of = signature.parameters["as_of"]
        assert as_of.kind is inspect.Parameter.KEYWORD_ONLY
        assert as_of.default is inspect.Parameter.empty

"""The daily injury table through the real schema: import, diff, clear, predict.

Monday Oct 27: Partizan host Real Madrid in game 5 (round 5, Oct 31); game 6
(round 6, Nov 7) follows. The page lists Jovanovic out for round 5, Petrovic out
for rounds 5-6, Garcia doubtful indefinitely and Martinez ready.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from euroleague_ai.ml.inference import predict_game
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, EntityType, StatusSourceType
from euroleague_ai.models.news import NewsArticle
from euroleague_ai.news.injury_report import import_injury_report
from tests.conftest import requires_db
from tests.integration.test_availability import _game, _id
from tests.integration.test_injuries import INGEST, MONDAY_BEFORE_GAME5, _models, _plays, _world
from tests.unit.test_injury_report import BLOCKS, page

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

URL = "https://example.test/injury-report.html"
TUESDAY = MONDAY_BEFORE_GAME5 + timedelta(days=1)


async def _import(
    session: AsyncSession, blocks: dict, *, at: object = MONDAY_BEFORE_GAME5, **kw: str
) -> object:
    result = await import_injury_report(
        session,
        page(blocks, **kw),
        url=URL,
        now=at,
        min_rows=1,
        min_teams=1,  # type: ignore[arg-type]
    )
    await session.flush()
    return result


async def _reports(session: AsyncSession, code: str) -> list[PlayerStatusEvent]:
    player_id = await _id(session, EntityType.PLAYER, code)
    return list(
        (
            await session.execute(
                select(PlayerStatusEvent)
                .where(PlayerStatusEvent.player_id == player_id)
                .order_by(PlayerStatusEvent.observed_at)
            )
        ).scalars()
    )


class TestImport:
    async def test_rounds_become_game_and_date_scopes(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _world(session, fixture_provider, season)
        result = await _import(session, BLOCKS)
        assert (result.matched, result.written, result.cleared) == (4, 4, 0)  # type: ignore[attr-defined]

        game5, game6 = await _game(session, 5), await _game(session, 6)
        (jovanovic,) = await _reports(session, "011002")
        assert jovanovic.source_type is StatusSourceType.REPORT
        assert (jovanovic.status, jovanovic.game_id) == (AvailabilityStatus.OUT, game5.id)
        assert jovanovic.reason == "Knee"
        assert jovanovic.observed_at == MONDAY_BEFORE_GAME5
        assert jovanovic.event_timestamp < MONDAY_BEFORE_GAME5  # page edited at 06:30 UTC

        (petrovic,) = await _reports(session, "011001")
        assert petrovic.game_id is None
        assert petrovic.expected_return_date == game6.tip_off_utc + timedelta(minutes=1)

        (garcia,) = await _reports(session, "012001")
        assert (garcia.status, garcia.game_id, garcia.expected_return_date) == (
            AvailabilityStatus.DOUBTFUL,
            None,
            None,
        )
        (martinez,) = await _reports(session, "012002")
        assert martinez.status is AvailabilityStatus.AVAILABLE

        articles = (await session.execute(select(NewsArticle))).scalars().all()
        assert len(articles) == 1 and jovanovic.source_article_id == articles[0].id

    async def test_an_unchanged_table_writes_nothing(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _world(session, fixture_provider, season)
        await _import(session, BLOCKS)
        again = await _import(session, BLOCKS, at=TUESDAY)
        assert (again.written, again.cleared, again.unchanged) == (0, 0, 4)  # type: ignore[attr-defined]
        assert len((await session.execute(select(PlayerStatusEvent))).scalars().all()) == 4
        assert len((await session.execute(select(NewsArticle))).scalars().all()) == 1

    async def test_dropping_off_the_table_clears_and_changes_are_recorded(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _world(session, fixture_provider, season)
        await _import(session, BLOCKS)
        tuesday = {
            "Partizan Mozzart Bet Belgrade": [BLOCKS["Partizan Mozzart Bet Belgrade"][1]],
            "Real Madrid": [
                ("Pablo Garcia", "Out", "Round 5", "Ankle"),
                BLOCKS["Real Madrid"][1],
            ],
        }
        result = await _import(session, tuesday, modified="2025-10-28T09:00:00+0300", at=TUESDAY)  # type: ignore[call-arg]
        assert (result.written, result.cleared) == (1, 1)  # type: ignore[attr-defined]

        jovanovic = await _reports(session, "011002")
        assert [r.status for r in jovanovic] == [
            AvailabilityStatus.OUT,
            AvailabilityStatus.AVAILABLE,
        ]
        assert jovanovic[-1].reason == "No longer on the injury report"
        garcia = await _reports(session, "012001")
        assert garcia[-1].status is AvailabilityStatus.OUT
        assert garcia[-1].game_id == (await _game(session, 5)).id


class TestConfirmation:
    async def test_every_fetch_is_recorded_even_when_nothing_changed(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        from euroleague_ai.repositories.availability import table_fetches

        await _world(session, fixture_provider, season)
        await _import(session, BLOCKS)
        await _import(session, BLOCKS, at=TUESDAY)
        fetches = await table_fetches(session, as_of=TUESDAY)
        assert [f.fetched_at for f in fetches] == [TUESDAY, MONDAY_BEFORE_GAME5]
        partizan = str(await _id(session, EntityType.TEAM, "PAR"))
        assert all(partizan in f.team_ids for f in fetches)

    async def test_a_misread_news_out_loses_to_the_table_fetched_after_it(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        """The Nunn case: the table says ready on Monday, a misread article says
        out on Tuesday morning, Tuesday afternoon's fetch still says ready."""
        await _world(session, fixture_provider, season)
        await _import(session, BLOCKS)
        martinez = await _id(session, EntityType.PLAYER, "012002")
        session.add(
            PlayerStatusEvent(
                player_id=martinez,
                team_id=await _id(session, EntityType.TEAM, "MAD"),
                game_id=(await _game(session, 5)).id,
                status=AvailabilityStatus.OUT,
                source_type=StatusSourceType.NEWS,
                confidence=0.85,
                reason="ruled out",
                event_timestamp=TUESDAY - timedelta(hours=4),
                observed_at=TUESDAY - timedelta(hours=4),
            )
        )
        unchanged = await _import(session, BLOCKS, at=TUESDAY)
        assert unchanged.written == 0  # type: ignore[attr-defined]

        await predict_game(
            session,
            await _game(session, 5),
            await _models(session),
            clock=INGEST,
            settings=settings,
        )
        plays = await _plays(session, 5)
        assert plays["012002"][0] == pytest.approx(0.9)  # not ruled out
        assert plays["011002"][0] == 0.0  # the table's own outs still apply


class TestPredictions:
    async def test_table_outs_set_plays_to_zero_and_ready_clears_news(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        # An older news reading has Martinez out; the table says he is ready.
        martinez_id = await _id(session, EntityType.PLAYER, "012002")
        team_id = await _id(session, EntityType.TEAM, "MAD")
        session.add(
            PlayerStatusEvent(
                player_id=martinez_id,
                team_id=team_id,
                status=AvailabilityStatus.OUT,
                source_type=StatusSourceType.NEWS,
                confidence=0.9,
                reason="ruled out",
                event_timestamp=MONDAY_BEFORE_GAME5 - timedelta(days=1),
                observed_at=MONDAY_BEFORE_GAME5 - timedelta(days=1),
            )
        )
        await _import(session, BLOCKS)
        await predict_game(
            session,
            await _game(session, 5),
            await _models(session),
            clock=INGEST,
            settings=settings,
        )

        plays = await _plays(session, 5)
        assert plays["011002"][0] == 0.0 and plays["011001"][0] == 0.0
        assert plays["011002"][1]["source"] == "REPORT"
        assert plays["012002"][0] == pytest.approx(0.9)  # cleared by "Ready"
        assert plays["012001"][0] == pytest.approx(0.9)  # doubtful: flagged only
        assert plays["012001"][1]["status"] == "DOUBTFUL"

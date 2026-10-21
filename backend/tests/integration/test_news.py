"""News to predictions through the real schema, with a stand-in feed.

Partizan play Real Madrid in game 5 (Oct 31). A Monday article rules
Jovanovic out with confidence; another only says Garcia "is injured", which is
too vague to act on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from euroleague_ai.ml.inference import predict_game
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.models.news import NewsArticle, NewsEvent
from euroleague_ai.news.base import ArticleDTO
from euroleague_ai.news.pipeline import collect_news, reread_articles
from tests.conftest import requires_db
from tests.integration.test_availability import _game
from tests.integration.test_injuries import INGEST, MONDAY_BEFORE_GAME5, _models, _plays, _world

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

FEED = "https://example.test/en/euroleague/feed/"


class _Feed:
    collector_key = "stub"

    def __init__(self, articles: list[ArticleDTO]) -> None:
        self.articles = articles

    async def collect(self, feed_url: str) -> list[ArticleDTO]:
        return list(self.articles)

    async def aclose(self) -> None:
        return None


ARTICLES = [
    ArticleDTO(
        url="https://example.test/en/euroleague/1/injury-report/",
        title="Injury report: Partizan without Jovanovic",
        body=(
            "Partizan will be short-handed against Real Madrid.\n"
            "Luka Jovanovic is ruled out for Friday's game with a knee problem, the club confirmed.\n"
            "Marko Petrovic scored 20 points last week."
        ),
        published_at=datetime(2025, 10, 27, 8, 0, tzinfo=UTC),
    ),
    ArticleDTO(
        url="https://example.test/en/euroleague/2/madrid-notes/",
        title="Madrid notes",
        body="Pablo Garcia is injured according to local reports.",
        published_at=datetime(2025, 10, 27, 9, 0, tzinfo=UTC),
    ),
]


async def _collect(session: AsyncSession, settings: Settings) -> object:
    return await collect_news(
        session,
        settings=settings.model_copy(update={"news_feed_urls": [FEED]}),
        now=MONDAY_BEFORE_GAME5,
        collector=_Feed(ARTICLES),
    )


class TestCollection:
    async def test_articles_and_readings_are_stored_with_fetch_time(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        result = await _collect(session, settings)

        assert result.new_articles == 2  # type: ignore[attr-defined]
        assert result.readings == 2 and result.ruled_out == 1  # type: ignore[attr-defined]
        articles = (await session.execute(select(NewsArticle))).scalars().all()
        assert {a.observed_at for a in articles} == {MONDAY_BEFORE_GAME5}

        reports = (await session.execute(select(PlayerStatusEvent))).scalars().all()
        assert {r.source_type for r in reports} == {StatusSourceType.NEWS}
        assert all(r.source_article_id is not None for r in reports)
        assert all(r.observed_at == MONDAY_BEFORE_GAME5 for r in reports)
        events = (await session.execute(select(NewsEvent))).scalars().all()
        assert any("ruled out" in e.payload["sentence"] for e in events)

        # "ruled out for Friday's game" is about that game only; "is injured" is open.
        game5 = await _game(session, 5)
        assert sorted(str(r.game_id) for r in reports) == sorted([str(game5.id), "None"])
        scopes = {e.payload["scope"]: e.payload for e in events}
        assert scopes["game"]["game_code"] == 5
        assert scopes["open"]["game_code"] is None

    async def test_a_second_run_stores_nothing_new(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        await _collect(session, settings)
        again = await _collect(session, settings)
        assert again.new_articles == 0 and again.readings == 0  # type: ignore[attr-defined]
        assert len((await session.execute(select(PlayerStatusEvent))).scalars().all()) == 2


class TestReread:
    async def test_replace_drops_old_readings_instead_of_adding_to_them(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        await _collect(session, settings)
        before = (await session.execute(select(PlayerStatusEvent.id))).scalars().all()

        articles, readings, ruled_out = await reread_articles(
            session, now=MONDAY_BEFORE_GAME5, days=7
        )
        assert (articles, readings, ruled_out) == (2, 2, 1)
        after = (await session.execute(select(PlayerStatusEvent.id))).scalars().all()
        assert len(after) == 2 and not set(before) & set(after)
        assert len((await session.execute(select(NewsEvent))).scalars().all()) == 2


class TestPredictions:
    async def test_confident_news_rules_out_and_vague_news_only_flags(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        await _collect(session, settings)
        await predict_game(session, await _game(session, 5), await _models(session),
                           clock=INGEST, settings=settings)

        plays = await _plays(session, 5)
        jovanovic, garcia = plays["011002"], plays["012001"]
        assert jovanovic[0] == 0.0
        assert jovanovic[1]["source"] == "NEWS" and jovanovic[1]["applied"] is True
        assert garcia[0] == pytest.approx(0.9)
        assert garcia[1]["status"] == str(AvailabilityStatus.OUT)
        assert garcia[1]["applied"] is False

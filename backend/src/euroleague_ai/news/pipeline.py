"""Collect feeds, store articles, read injury statuses out of them.

Runs in the scheduled pipeline before predictions, so a player reported out in
the morning's coverage is ruled out in the afternoon's projections.

Each reading is written twice, for two audiences:

* a :class:`~euroleague_ai.models.news.NewsEvent`, the audit trail: which
  article, which sentence, which phrase, which extractor version;
* a :class:`~euroleague_ai.models.availability.PlayerStatusEvent` with
  ``source_type = NEWS`` and the reading's confidence, the same table manual
  reports go to, so prediction code needs no news-specific path.

Both carry the article's ``observed_at`` (when we fetched it), never the
publisher's timestamp.

**How long a reading lasts.** Three shapes, from the wording:

* "ruled out for Efes's game at Zalgiris on Friday": that **one game**, the
  team's first fixture after the article was published;
* "sidelined for 6 to 8 weeks", "set to miss 20 days": **until** the lower bound
  of the stated absence;
* anything else ("Fournier is out"): open, until he plays again or is reported
  back, and never longer than :data:`~euroleague_ai.features.injury_status.NEWS_OPEN_MAX_AGE`.
  Long absences are restated in every round's report, which renews them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, or_, select

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.core.logging import get_logger
from euroleague_ai.ingestion.injuries import current_rosters
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, NewsEventType, StatusSourceType
from euroleague_ai.models.game import Game
from euroleague_ai.models.news import NewsArticle, NewsEvent
from euroleague_ai.news.injury_extract import (
    EXTRACTOR_VERSION,
    Extracted,
    RosterIndex,
    extract_statuses,
)
from euroleague_ai.news.store import store_articles

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.news.base import NewsCollector

logger = get_logger(__name__)

_EVENT_TYPE = {
    AvailabilityStatus.OUT: NewsEventType.INJURY,
    AvailabilityStatus.SUSPENDED: NewsEventType.AVAILABILITY_CHANGE,
    AvailabilityStatus.DOUBTFUL: NewsEventType.AVAILABILITY_CHANGE,
    AvailabilityStatus.QUESTIONABLE: NewsEventType.AVAILABILITY_CHANGE,
    AvailabilityStatus.PROBABLE: NewsEventType.AVAILABILITY_CHANGE,
    AvailabilityStatus.AVAILABLE: NewsEventType.RETURN_FROM_INJURY,
}


@dataclass(slots=True)
class NewsRunResult:
    feeds: dict[str, int] = field(default_factory=dict)
    new_articles: int = 0
    articles_read: int = 0
    readings: int = 0
    ruled_out: int = 0
    failed_feeds: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feeds": self.feeds,
            "new_articles": self.new_articles,
            "articles_read": self.articles_read,
            "readings": self.readings,
            "ruled_out": self.ruled_out,
            "failed_feeds": self.failed_feeds,
        }


#: A one-game reading attaches to the team's first fixture within this window
#: after publication. Further out, the article is probably not about that game.
GAME_LOOKAHEAD = timedelta(days=10)


async def _first_game_after(
    session: AsyncSession, team_id: uuid.UUID, after: datetime
) -> Game | None:
    return (
        await session.execute(
            select(Game)
            .where(
                or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                Game.tip_off_utc > after,
                Game.tip_off_utc <= after + GAME_LOOKAHEAD,
            )
            .order_by(Game.tip_off_utc)
            .limit(1)
        )
    ).scalar_one_or_none()


def _article_text(article: NewsArticle) -> str:
    return "\n".join(
        part for part in (article.title, article.body or article.summary or "") if part
    )


async def record_readings(
    session: AsyncSession, article: NewsArticle, readings: Sequence[Extracted]
) -> None:
    """Write the audit event and the status report for each reading."""
    observed = ensure_utc(article.observed_at, field="observed_at")
    published = article.published_at
    # When the status changed in the world: the publisher's time, but never
    # later than when we saw it, and never trusted to make it earlier than the
    # cutoff can see (observed_at still gates every read).
    event_time = (
        min(ensure_utc(published, field="published_at"), observed) if published else observed
    )
    for reading in readings:
        game: Game | None = None
        until: datetime | None = None
        if reading.duration_days is not None:
            until = event_time + timedelta(days=reading.duration_days)
        elif reading.game_scoped:
            game = await _first_game_after(session, reading.team_id, event_time)
        scope = "until" if until else "game" if game else "open"
        session.add(
            NewsEvent(
                id=uuid.uuid4(),
                article_id=article.id,
                event_type=_EVENT_TYPE[reading.status],
                player_id=reading.player_id,
                team_id=reading.team_id,
                payload={
                    "status": str(reading.status),
                    "phrase": reading.phrase,
                    "sentence": reading.sentence,
                    "scope": scope,
                    "game_code": game.game_code if game else None,
                    "until": until.date().isoformat() if until else None,
                    "published_at": event_time.isoformat(),
                },
                confidence=reading.confidence,
                extraction_model=EXTRACTOR_VERSION,
                event_timestamp=event_time,
                observed_at=observed,
            )
        )
        session.add(
            PlayerStatusEvent(
                id=uuid.uuid4(),
                player_id=reading.player_id,
                team_id=reading.team_id,
                game_id=game.id if game else None,
                status=reading.status,
                reason=reading.phrase[:512],
                expected_return_date=until,
                source_type=StatusSourceType.NEWS,
                source_article_id=article.id,
                confidence=reading.confidence,
                event_timestamp=event_time,
                observed_at=observed,
            )
        )


async def extract_articles(
    session: AsyncSession, articles: Sequence[NewsArticle], *, now: datetime
) -> tuple[int, int]:
    """Read every article; returns ``(readings, of which rule a player out)``."""
    if not articles:
        return 0, 0
    index = RosterIndex(await current_rosters(session, as_of=now))
    readings = ruled_out = 0
    for article in articles:
        found = extract_statuses(_article_text(article), index)
        await record_readings(session, article, found)
        readings += len(found)
        ruled_out += sum(1 for r in found if r.rules_out)
    await session.flush()
    return readings, ruled_out


async def unread_articles(session: AsyncSession, *, now: datetime, days: int) -> list[NewsArticle]:
    """Recent articles with no readings from the current extractor version.

    Lets improved rules be re-run over the last few weeks. Articles that simply
    mention no injuries are read again each time, which costs a regex pass.
    """
    since = ensure_utc(now, field="now") - timedelta(days=days)
    already = select(NewsEvent.article_id).where(NewsEvent.extraction_model == EXTRACTOR_VERSION)
    return list(
        (
            await session.execute(
                select(NewsArticle).where(
                    NewsArticle.observed_at >= since, NewsArticle.id.not_in(already)
                )
            )
        ).scalars()
    )


async def reread_articles(
    session: AsyncSession, *, now: datetime, days: int
) -> tuple[int, int, int]:
    """Re-read the last ``days`` of stored articles with the current rules.

    News readings are derived data: the articles are the record, so replacing
    the readings taken from them loses nothing. Earlier readings from those
    articles are deleted first, so a fixed rule cannot leave the old misreading
    in place beside the new one. Predictions already written keep the report
    they used, inside their explanation.

    Returns ``(articles, readings, ruling a player out)``.
    """
    since = ensure_utc(now, field="now") - timedelta(days=days)
    query = select(NewsArticle).where(NewsArticle.observed_at >= since)
    articles = list((await session.execute(query)).scalars())
    ids = [a.id for a in articles]
    if ids:
        await session.execute(
            delete(PlayerStatusEvent).where(
                PlayerStatusEvent.source_type == StatusSourceType.NEWS,
                PlayerStatusEvent.source_article_id.in_(ids),
            )
        )
        await session.execute(delete(NewsEvent).where(NewsEvent.article_id.in_(ids)))
    readings, ruled_out = await extract_articles(session, articles, now=now)
    return len(articles), readings, ruled_out


async def collect_news(
    session: AsyncSession,
    *,
    settings: Settings,
    now: datetime,
    collector: NewsCollector | None = None,
) -> NewsRunResult:
    """Poll every configured feed, store new articles, extract injury readings."""
    from euroleague_ai.news.rss import RssNewsCollector

    now = ensure_utc(now, field="now")
    result = NewsRunResult()
    owned = collector is None
    collector = collector or RssNewsCollector(settings)
    new: list[NewsArticle] = []
    try:
        for url in settings.news_feed_urls:
            dtos = await collector.collect(url)
            if not dtos:
                result.failed_feeds.append(url)
            stored = await store_articles(session, feed_url=url, articles=dtos, observed_at=now)
            result.feeds[url] = len(stored)
            new.extend(stored)
    finally:
        if owned:
            await collector.aclose()
    result.new_articles = len(new)
    result.articles_read = len(new)
    result.readings, result.ruled_out = await extract_articles(session, new, now=now)
    logger.info("news collected", extra=result.as_dict())
    return result

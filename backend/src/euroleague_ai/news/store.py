"""Persisting collected articles.

``observed_at`` is the fetch time, never the publisher's ``pubDate``: a story
dated 09:00 that we first saw at 18:00 was not knowable at 17:00. Articles are
keyed by URL and never rewritten. A story the publisher edits later keeps the
version we first saw, which is the version a prediction made then could have
used.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from sqlalchemy import select

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.models.news import NewsArticle, NewsSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.news.base import ArticleDTO


def source_key(feed_url: str) -> str:
    """Stable short key for a feed: its host and path, hashed to fit the column."""
    parsed = urlparse(feed_url)
    readable = f"{parsed.netloc}{parsed.path}".strip("/")
    if len(readable) <= 64:
        return readable
    return readable[:47] + "-" + hashlib.sha256(readable.encode()).hexdigest()[:16]


async def ensure_source(session: AsyncSession, feed_url: str) -> NewsSource:
    """The source row for a feed, created on first use."""
    key = source_key(feed_url)
    existing = (
        await session.execute(select(NewsSource).where(NewsSource.key == key))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    source = NewsSource(
        id=uuid.uuid4(),
        key=key,
        name=urlparse(feed_url).netloc or key,
        feed_url=feed_url,
        language="en",
        is_enabled=True,
        reliability_weight=1.0,
    )
    session.add(source)
    await session.flush()
    return source


async def store_articles(
    session: AsyncSession,
    *,
    feed_url: str,
    articles: Sequence[ArticleDTO],
    observed_at: datetime,
) -> list[NewsArticle]:
    """Insert articles not seen before. Returns only the new ones."""
    observed_at = ensure_utc(observed_at, field="observed_at")
    if not articles:
        return []
    source = await ensure_source(session, feed_url)
    urls = [a.url for a in articles]
    known = set(
        (
            await session.execute(select(NewsArticle.url).where(NewsArticle.url.in_(urls)))
        ).scalars()
    )
    known_hashes = set(
        (
            await session.execute(
                select(NewsArticle.content_hash).where(
                    NewsArticle.content_hash.in_([a.content_hash for a in articles])
                )
            )
        ).scalars()
    )
    new: list[NewsArticle] = []
    for dto in articles:
        # Same URL, or the same story syndicated under another URL.
        if dto.url in known or dto.content_hash in known_hashes:
            continue
        known.add(dto.url)
        known_hashes.add(dto.content_hash)
        article = NewsArticle(
            id=uuid.uuid4(),
            source_id=source.id,
            external_id=(dto.external_id or None) and dto.external_id[:256],
            url=dto.url[:1024],
            title=dto.title[:512],
            summary=dto.summary,
            body=dto.body,
            language=dto.language,
            author=(dto.author or None) and dto.author[:160],
            published_at=dto.published_at,
            content_hash=dto.content_hash,
            observed_at=observed_at,
        )
        session.add(article)
        new.append(article)
    await session.flush()
    return new

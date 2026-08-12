"""News sources, articles, and the structured events extracted from them.

Article storage is implemented. Event extraction is planned -- the tables exist
now so that landing it is an insert path rather than a migration.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import NewsEventType
from euroleague_ai.models.mixins import ObservedMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from euroleague_ai.models.people import Player


class NewsSource(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A feed we are permitted to poll.

    No sources ship enabled. Populate this table with feeds you have the right to
    consume; see ``NEWS_FEED_URLS`` in ``.env.example``.
    """

    __tablename__ = "news_sources"

    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    feed_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    is_enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    reliability_weight: Mapped[float] = mapped_column(
        Numeric(4, 3),
        default=1.0,
        nullable=False,
        doc="Down-weights sources with a poor track record when features aggregate events.",
    )


class NewsArticle(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """A single article.

    ``published_at`` is what the publisher claims; ``observed_at`` is when we
    fetched it. Only the latter gates features. Publishers backdate, syndication
    reorders, and trusting a publisher's clock is exactly how a news feature ends
    up leaking.
    """

    __tablename__ = "news_articles"
    __table_args__ = (
        Index("ix_news_articles_published", "published_at"),
        Index("ix_news_articles_hash", "content_hash"),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("news_sources.id"), nullable=False
    )
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    url: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    author: Mapped[str | None] = mapped_column(String(160), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(nullable=True)
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 of title+body. Deduplicates syndicated copies of one story.",
    )

    events: Mapped[list[NewsEvent]] = relationship(back_populates="article")


class NewsEvent(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """A structured basketball event extracted from an article.

    The target transformation::

        "Coach says Player X will have his minutes monitored tonight."
          -> {"event_type": "MINUTES_LIMIT", "player_id": ..., "confidence": 0.91}

    Extraction is planned. Generic sentiment analysis is explicitly not the goal:
    the useful signal is the basketball event, not the tone of the prose.
    """

    __tablename__ = "news_events"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_in_unit_interval"
        ),
        Index("ix_news_events_player_observed", "player_id", "observed_at"),
        Index("ix_news_events_type_observed", "event_type", "observed_at"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("news_articles.id"), nullable=False
    )
    event_type: Mapped[NewsEventType] = mapped_column(nullable=False)

    # Nullable: extraction may identify an event but fail to resolve the entity.
    # Storing it unresolved beats discarding it -- resolution can improve later.
    player_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=True
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=True
    )

    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc='Event-specific fields, e.g. {"expected_minutes": null}.',
    )
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    extraction_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_timestamp: Mapped[datetime] = mapped_column(nullable=False)

    article: Mapped[NewsArticle] = relationship(back_populates="events")
    player: Mapped[Player | None] = relationship()

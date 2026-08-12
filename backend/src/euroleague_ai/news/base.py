"""News collection protocol and DTO."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ArticleDTO(BaseModel):
    """One collected article, before entity resolution or extraction."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    external_id: str | None = None
    url: str
    title: str
    summary: str | None = None
    body: str | None = None
    author: str | None = None
    language: str = "en"

    # What the publisher claims. Never used as a feature cutoff -- publishers
    # backdate and syndication reorders, so only observed_at (stamped at ingest)
    # gates what a model may see.
    published_at: datetime | None = None

    @property
    def content_hash(self) -> str:
        """SHA-256 of title and body, for deduplicating syndicated copies."""
        payload = f"{self.title}\n{self.body or self.summary or ''}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class NewsCollector(Protocol):
    """Source of news articles.

    No collector ships enabled. Configure ``NEWS_FEED_URLS`` with feeds you are
    permitted to poll; see the legal note in docs/architecture.md.
    """

    collector_key: str

    async def collect(self, feed_url: str) -> list[ArticleDTO]:
        """Fetch and parse one feed."""
        ...

    async def aclose(self) -> None:
        """Release any held resources."""
        ...

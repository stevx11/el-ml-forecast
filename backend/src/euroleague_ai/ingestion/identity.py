"""Identity resolution: provider identifiers to internal entities.

Every entity that arrives from a provider passes through here exactly once. The
resolver keeps an in-memory cache for the duration of an ingestion run, because a
season backfill resolves the same twenty club codes several thousand times and a
round trip per lookup would dominate the run.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.enums import EntityType
from euroleague_ai.models.identity import ExternalRef

logger = get_logger(__name__)


class IdentityResolver:
    """Maps ``(entity_type, external_id)`` to a stable internal UUID."""

    def __init__(
        self,
        session: AsyncSession,
        provider_key: str,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._session = session
        self._provider_key = provider_key
        self._clock = clock or SystemClock()
        self._cache: dict[tuple[EntityType, str], uuid.UUID] = {}

    async def preload(self, entity_type: EntityType) -> None:
        """Load every known reference of one type into the cache.

        One query instead of N. Called at the start of each ingestion job.
        """
        stmt = select(ExternalRef.external_id, ExternalRef.internal_id).where(
            ExternalRef.provider_key == self._provider_key,
            ExternalRef.entity_type == entity_type,
        )
        for external_id, internal_id in (await self._session.execute(stmt)).all():
            self._cache[(entity_type, external_id)] = internal_id

    async def lookup(self, entity_type: EntityType, external_id: str) -> uuid.UUID | None:
        """Return the internal id for a provider reference, or ``None``."""
        key = (entity_type, external_id)
        if (cached := self._cache.get(key)) is not None:
            return cached

        stmt = select(ExternalRef.internal_id).where(
            ExternalRef.provider_key == self._provider_key,
            ExternalRef.entity_type == entity_type,
            ExternalRef.external_id == external_id,
        )
        internal_id = (await self._session.execute(stmt)).scalar_one_or_none()
        if internal_id is not None:
            self._cache[key] = internal_id
        return internal_id

    async def resolve(
        self, entity_type: EntityType, external_id: str
    ) -> tuple[uuid.UUID, bool]:
        """Return ``(internal_id, is_new)``, registering the reference if unseen.

        ``ON CONFLICT DO NOTHING`` plus a re-select handles the race where two
        concurrent ingestion jobs discover the same new player simultaneously:
        one insert wins, both end up with the winner's id, and neither crashes.
        """
        if (existing := await self.lookup(entity_type, external_id)) is not None:
            return existing, False

        new_id = uuid.uuid4()
        stmt = (
            pg_insert(ExternalRef)
            .values(
                id=uuid.uuid4(),
                provider_key=self._provider_key,
                entity_type=entity_type,
                external_id=external_id,
                internal_id=new_id,
                observed_at=self._clock.now(),
            )
            .on_conflict_do_nothing(
                index_elements=["provider_key", "entity_type", "external_id"]
            )
            .returning(ExternalRef.internal_id)
        )
        returned = (await self._session.execute(stmt)).scalar_one_or_none()

        if returned is None:
            # Another writer won the race; adopt their id.
            winner = await self.lookup(entity_type, external_id)
            if winner is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"could not resolve {entity_type}:{external_id} after conflict"
                )
            return winner, False

        self._cache[(entity_type, external_id)] = returned
        return returned, True

    def remember(self, entity_type: EntityType, external_id: str, internal_id: uuid.UUID) -> None:
        """Seed the cache without touching the database. Test support."""
        self._cache[(entity_type, external_id)] = internal_id

    @property
    def stats(self) -> dict[str, Any]:
        return {"cached_refs": len(self._cache)}

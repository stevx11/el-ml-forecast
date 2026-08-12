"""Mapping between provider identifiers and internal entity ids.

Provider ids never become primary keys. One table handles every provider and
every entity type, which means adding Sportradar or an ACB feed requires no
migration and no new columns.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import EntityType
from euroleague_ai.models.mixins import ObservedMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ExternalRef(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """A provider's identifier for an entity we hold internally.

    The alternative -- a ``euroleague_id`` column on every table -- needs a
    migration per provider and produces a wide, mostly-NULL schema. It also
    cannot express the same player appearing in three feeds under three ids,
    which is the normal case for a EuroLeague player who also plays domestically.
    """

    __tablename__ = "external_refs"
    __table_args__ = (
        UniqueConstraint(
            "provider_key",
            "entity_type",
            "external_id",
            name="uq_external_refs_provider_type_id",
        ),
        Index("ix_external_refs_internal", "entity_type", "internal_id"),
    )

    provider_key: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(nullable=False)
    external_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Normalised provider identifier. See providers.mapping.normalise_person_code.",
    )
    internal_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)

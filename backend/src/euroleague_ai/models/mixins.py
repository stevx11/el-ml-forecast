"""Reusable column mixins.

The three timestamp concepts in this schema are deliberately separated into three
mixins so that a model declaration states, in one glance, which temporal
semantics apply to it.

    TimestampMixin    -> row bookkeeping.        Never used in features.
    ObservedMixin     -> when *we* learned it.   THE leakage filter.
    EffectiveMixin    -> when it was true.       Interval-valued state.

See ``docs/data-model.md`` for why conflating these is the most dangerous bug
available in this system.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, declarative_mixin, mapped_column


@declarative_mixin
class UUIDPrimaryKeyMixin:
    """UUID primary key generated application-side.

    Client-side generation lets ingestion assemble an object graph -- a game, its
    two sides, and twenty-four box score rows -- and assign foreign keys before
    any of it is flushed. With server-side sequences that requires a flush per
    level, which is both slower and more awkward to test.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


@declarative_mixin
class TimestampMixin:
    """Row creation and modification bookkeeping.

    Audit only. These columns must never appear in a feature query: they reflect
    when a *row* was touched, which has no basketball meaning and drifts whenever
    a backfill rewrites unrelated columns.
    """

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


@declarative_mixin
class ObservedMixin:
    """When this application learned the fact.

    **This is the column every feature query filters on.**

    Consider a report published at 19:00 stating a player was injured at 10:00. A
    model predicting at 17:00 could not possibly have known. Filtering on the
    10:00 event time would leak; filtering on ``observed_at`` -- roughly 19:00,
    when ingestion actually saw it -- does not.

    Indexed because ``WHERE observed_at <= :as_of`` appears in effectively every
    feature query in the system.
    """

    observed_at: Mapped[datetime] = mapped_column(
        nullable=False,
        index=True,
        doc="When this application recorded the fact. The leakage cutoff column.",
    )


@declarative_mixin
class EffectiveMixin:
    """Half-open validity interval ``[effective_from, effective_until)``.

    Used for state that changes over time but is not event-shaped: roster
    membership, jersey numbers, coaching appointments.

    Superseding a fact closes the current row and inserts a new one. Values are
    never overwritten, because overwriting destroys the answer to "what was true
    in December?" the moment a January transfer is ingested.
    """

    effective_from: Mapped[datetime] = mapped_column(nullable=False, index=True)
    effective_until: Mapped[datetime | None] = mapped_column(
        nullable=True,
        index=True,
        doc="NULL means the interval is still open.",
    )


def effective_interval_check() -> CheckConstraint:
    """Constraint guaranteeing a non-empty validity interval.

    Returned as a factory rather than declared on :class:`EffectiveMixin` via
    ``declared_attr``: a mixin-supplied ``__table_args__`` is silently shadowed
    when the model declares its own, and every model here declares its own
    indexes. Requiring an explicit call keeps the constraint visible at the model
    it protects.
    """
    return CheckConstraint(
        "effective_until IS NULL OR effective_until > effective_from",
        name="effective_interval_valid",
    )

"""Availability history and lineup announcements.

Both tables here are **append-only**. Nothing is ever updated in place, because
the entire point is to be able to reconstruct what was known at an arbitrary past
instant.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.models.mixins import ObservedMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from euroleague_ai.models.people import Player


class PlayerStatusEvent(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """One observation of a player's availability. Append-only.

    This table is the reason the whole schema is shaped this way. Consider a game
    day on which a player's status moves::

        10:00  OUT
        14:00  QUESTIONABLE
        18:30  AVAILABLE

    That is three rows, not one row updated twice. A model predicting at 17:00
    reads the first two, resolves to ``QUESTIONABLE``, and correctly does not know
    about the 18:30 upgrade.

    Had this been a mutable ``players.current_status`` column, that prediction
    would be unreproducible -- and any backtest of it would be quietly dishonest,
    because the backtest would credit the model with information it never had.
    """

    __tablename__ = "player_status_events"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_in_unit_interval",
        ),
        # The dominant query: "latest status for this player as of T".
        Index(
            "ix_player_status_events_player_observed",
            "player_id",
            "observed_at",
            "event_timestamp",
        ),
        Index("ix_player_status_events_game", "game_id"),
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=True
    )
    game_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("games.id"),
        nullable=True,
        doc="Set when the observation is scoped to a specific fixture.",
    )

    status: Mapped[AvailabilityStatus] = mapped_column(nullable=False)
    reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    injury_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    expected_return_date: Mapped[datetime | None] = mapped_column(nullable=True)

    source_type: Mapped[StatusSourceType] = mapped_column(
        default=StatusSourceType.PROVIDER, nullable=False
    )
    source_article_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("news_articles.id"), nullable=True
    )
    confidence: Mapped[float | None] = mapped_column(
        Numeric(4, 3),
        nullable=True,
        doc="0-1 for inferred observations. NULL when the source is authoritative.",
    )

    # When the status changed in the world. Distinct from observed_at, which is
    # when we learned it. Features filter on observed_at; this column is for
    # ordering and for display.
    event_timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)

    player: Mapped[Player] = relationship()


class StartingLineupEntry(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """An announced starter for a game. Append-only.

    Projected lineups appear hours before tip-off and are later superseded by the
    confirmed sheet. Both are kept: ``is_confirmed = False`` is itself a strong
    pre-game feature in EuroLeague, where lineups leak to the press well in
    advance and the difference between "reported" and "confirmed" carries
    information.
    """

    __tablename__ = "starting_lineup_entries"
    __table_args__ = (
        Index("ix_starting_lineup_game_team_observed", "game_id", "team_id", "observed_at"),
    )

    game_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("games.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )

    is_confirmed: Mapped[bool] = mapped_column(default=False, nullable=False)
    announced_at: Mapped[datetime] = mapped_column(nullable=False)
    source_type: Mapped[StatusSourceType] = mapped_column(
        default=StatusSourceType.PROVIDER, nullable=False
    )

    player: Mapped[Player] = relationship()

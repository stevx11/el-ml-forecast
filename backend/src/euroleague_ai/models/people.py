"""Players, coaches, and their time-bounded affiliations.

The split here is the important part. :class:`Player` holds only genuinely
immutable biography. Everything that changes -- team, jersey number, declared
position -- lives in an interval table, so that a February transfer does not
retroactively rewrite what was true in November.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import PlayerPosition
from euroleague_ai.models.mixins import (
    EffectiveMixin,
    ObservedMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    effective_interval_check,
)

if TYPE_CHECKING:
    from euroleague_ai.models.competition import Season, Team


class Player(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A person who plays basketball.

    Deliberately absent: current team, current jersey number, current status.
    Those are all time-varying, and a mutable column holding them would sooner or
    later be read by a feature builder, leaking the present into the past.
    """

    __tablename__ = "players"
    __table_args__ = (Index("ix_players_last_name", "last_name"),)

    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    last_name: Mapped[str | None] = mapped_column(String(96), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(96), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(nullable=True)
    country_code: Mapped[str | None] = mapped_column(String(3), nullable=True)
    birth_country_code: Mapped[str | None] = mapped_column(String(3), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(nullable=True)
    position: Mapped[PlayerPosition] = mapped_column(
        default=PlayerPosition.UNKNOWN,
        nullable=False,
        doc="Primary listed position. Per-stint position is on PlayerTeamStint.",
    )

    stints: Mapped[list[PlayerTeamStint]] = relationship(back_populates="player")


class PlayerTeamStint(Base, UUIDPrimaryKeyMixin, ObservedMixin, EffectiveMixin, TimestampMixin):
    """A spell of a player being registered with a team. Interval-valued.

    This is the transfer history. "Which team was this player on at time T?" is a
    query over intervals rather than a lookup of a mutable column, which is what
    makes historical feature computation honest.

    The EuroLeague ``/people`` endpoint supplies ``startDate`` and ``endDate``
    per registration, so this maps directly onto the upstream data.
    """

    __tablename__ = "player_team_stints"
    __table_args__ = (
        effective_interval_check(),
        Index("ix_player_team_stints_player_from", "player_id", "effective_from"),
        Index("ix_player_team_stints_team_season", "team_id", "season_id"),
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("seasons.id"), nullable=False
    )

    jersey_number: Mapped[str | None] = mapped_column(String(8), nullable=True)
    position: Mapped[PlayerPosition] = mapped_column(default=PlayerPosition.UNKNOWN, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
        doc="The provider's own active flag, retained for reconciliation against intervals.",
    )
    previous_team_name: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
        doc="Free-text prior club as reported upstream. Useful for transfer detection.",
    )

    player: Mapped[Player] = relationship(back_populates="stints")
    team: Mapped[Team] = relationship(back_populates="player_stints")
    season: Mapped[Season] = relationship()


class Coach(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A head coach.

    Tracked because coaching changes are a genuine EuroLeague signal: rotations
    tighten or widen sharply after a sacking, and the change date must be
    recoverable for features to reflect it.
    """

    __tablename__ = "coaches"

    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(3), nullable=True)


class TeamCoachStint(Base, UUIDPrimaryKeyMixin, ObservedMixin, EffectiveMixin, TimestampMixin):
    """A spell of a coach leading a team. Interval-valued."""

    __tablename__ = "team_coach_stints"
    __table_args__ = (
        effective_interval_check(),
        Index("ix_team_coach_stints_team_from", "team_id", "effective_from"),
    )

    coach_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("coaches.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("seasons.id"), nullable=False
    )

    coach: Mapped[Coach] = relationship()

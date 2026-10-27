"""Competition structure: leagues, seasons, teams, and season registration.

Nothing here is EuroLeague-specific. Adding ABA, ACB, or a domestic league is a
row in ``leagues``, not a migration.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.people import PlayerTeamStint


class League(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A competition. EuroLeague is one row among several."""

    __tablename__ = "leagues"

    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    country_code: Mapped[str | None] = mapped_column(
        String(3),
        nullable=True,
        doc="ISO-3166 alpha-3. NULL for transnational competitions such as EuroLeague.",
    )
    tier: Mapped[int | None] = mapped_column(
        nullable=True,
        doc="1 = top domestic/continental tier. Used to weight cross-competition features.",
    )

    seasons: Mapped[list[Season]] = relationship(back_populates="league")


class Season(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One edition of a competition, e.g. EuroLeague 2025-26."""

    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("league_id", "code", name="uq_seasons_league_code"),)

    league_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("leagues.id"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(
        String(32), nullable=False, doc='Provider code, e.g. "E2025".'
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    start_year: Mapped[int] = mapped_column(nullable=False)
    start_date: Mapped[date | None] = mapped_column(nullable=True)
    end_date: Mapped[date | None] = mapped_column(nullable=True)

    league: Mapped[League] = relationship(back_populates="seasons")
    games: Mapped[list[Game]] = relationship(back_populates="season")


class Team(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A club, independent of competition.

    Real Madrid is one row that plays in both EuroLeague and ACB. Season-specific
    facts -- above all the sponsored name, which changes yearly -- live on
    :class:`TeamSeason` so that a rebrand never orphans historical statistics.
    """

    __tablename__ = "teams"

    code: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    country_code: Mapped[str | None] = mapped_column(String(3), nullable=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    venue_name: Mapped[str | None] = mapped_column(String(160), nullable=True)

    # Venue coordinates back the travel-distance and timezone-shift features.
    # Nullable because the provider does not supply them; they are enriched
    # separately and their absence must degrade gracefully rather than crash.
    venue_latitude: Mapped[float | None] = mapped_column(Numeric(9, 6), nullable=True)
    venue_longitude: Mapped[float | None] = mapped_column(Numeric(9, 6), nullable=True)

    team_seasons: Mapped[list[TeamSeason]] = relationship(back_populates="team")
    player_stints: Mapped[list[PlayerTeamStint]] = relationship(back_populates="team")


class TeamSeason(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A team's registration in a season, carrying season-specific naming."""

    __tablename__ = "team_seasons"
    __table_args__ = (
        UniqueConstraint("team_id", "season_id", name="uq_team_seasons_team_season"),
        Index("ix_team_seasons_season", "season_id"),
    )

    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("seasons.id"), nullable=False
    )
    display_name: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        doc='Sponsored name for this season, e.g. "Kosner Baskonia Vitoria-Gasteiz".',
    )

    team: Mapped[Team] = relationship(back_populates="team_seasons")
    season: Mapped[Season] = relationship()

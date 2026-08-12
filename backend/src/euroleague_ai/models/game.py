"""Games, the two sides of a game, and box scores."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import GameStatus, TeamSide
from euroleague_ai.models.mixins import ObservedMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from euroleague_ai.models.competition import Season, Team
    from euroleague_ai.models.people import Coach, Player


class Game(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """A scheduled or completed fixture."""

    __tablename__ = "games"
    __table_args__ = (
        UniqueConstraint("season_id", "game_code", name="uq_games_season_game_code"),
        # These three indexes carry the schedule-density and rest features, which
        # are the hottest queries in the feature layer.
        Index("ix_games_season_tipoff", "season_id", "tip_off_utc"),
        Index("ix_games_home_tipoff", "home_team_id", "tip_off_utc"),
        Index("ix_games_away_tipoff", "away_team_id", "tip_off_utc"),
    )

    season_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("seasons.id"), nullable=False
    )
    game_code: Mapped[int] = mapped_column(nullable=False, doc="Provider game code within season.")

    round_number: Mapped[int | None] = mapped_column(nullable=True, index=True)
    phase_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    phase_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    group_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # tip_off_utc is authoritative; every time window in the feature layer uses
    # it. Local time and zone are kept separately and only for the travel and
    # circadian features that genuinely need them.
    tip_off_utc: Mapped[datetime] = mapped_column(nullable=False, index=True)
    tip_off_local: Mapped[datetime | None] = mapped_column(nullable=True)
    local_timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[GameStatus] = mapped_column(default=GameStatus.SCHEDULED, nullable=False)

    home_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    away_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    home_score: Mapped[int | None] = mapped_column(nullable=True)
    away_score: Mapped[int | None] = mapped_column(nullable=True)

    is_neutral_venue: Mapped[bool] = mapped_column(default=False, nullable=False)
    venue_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    attendance: Mapped[int | None] = mapped_column(nullable=True)

    season: Mapped[Season] = relationship(back_populates="games")
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    game_teams: Mapped[list[GameTeam]] = relationship(
        back_populates="game", cascade="all, delete-orphan"
    )

    @property
    def is_finished(self) -> bool:
        return self.status is GameStatus.FINISHED


class GameTeam(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One side of a game.

    Exists so that box score rows attach to a single foreign key rather than
    carrying both a game id and a team id, which can disagree.
    """

    __tablename__ = "game_teams"
    __table_args__ = (UniqueConstraint("game_id", "side", name="uq_game_teams_game_side"),)

    game_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False, index=True
    )
    side: Mapped[TeamSide] = mapped_column(nullable=False)
    score: Mapped[int | None] = mapped_column(nullable=True)
    quarter_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    coach_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("coaches.id"), nullable=True
    )

    game: Mapped[Game] = relationship(back_populates="game_teams")
    team: Mapped[Team] = relationship()
    coach: Mapped[Coach | None] = relationship()
    player_stats: Mapped[list[PlayerGameStats]] = relationship(
        back_populates="game_team", cascade="all, delete-orphan"
    )
    team_stats: Mapped[TeamGameStats | None] = relationship(
        back_populates="game_team", uselist=False, cascade="all, delete-orphan"
    )


class PlayerGameStats(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """A player's box score line for one game."""

    __tablename__ = "player_game_stats"
    __table_args__ = (
        UniqueConstraint("game_team_id", "player_id", name="uq_player_game_stats_team_player"),
        CheckConstraint("seconds_played >= 0", name="seconds_played_non_negative"),
        Index("ix_player_game_stats_player", "player_id"),
    )

    game_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("game_teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )

    # Seconds, not a "MM:SS" string and not fractional minutes. The provider
    # gives 1850.0 seconds on the v2 endpoint and "30:50" on the legacy one;
    # normalising at ingest removes parsing ambiguity from every consumer.
    # Minutes are a presentation concern.
    seconds_played: Mapped[int] = mapped_column(default=0, nullable=False)

    is_starter: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Explicitly distinct from seconds_played == 0, which is ambiguous between
    # "did not dress" and "played eleven seconds of garbage time". Minutes models
    # need that distinction: DNPs are excluded from form windows rather than
    # averaged in as zeros.
    did_not_play: Mapped[bool] = mapped_column(default=False, nullable=False)

    points: Mapped[int] = mapped_column(default=0, nullable=False)
    fg2m: Mapped[int] = mapped_column(default=0, nullable=False)
    fg2a: Mapped[int] = mapped_column(default=0, nullable=False)
    fg3m: Mapped[int] = mapped_column(default=0, nullable=False)
    fg3a: Mapped[int] = mapped_column(default=0, nullable=False)
    ftm: Mapped[int] = mapped_column(default=0, nullable=False)
    fta: Mapped[int] = mapped_column(default=0, nullable=False)
    offensive_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    defensive_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    total_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    assists: Mapped[int] = mapped_column(default=0, nullable=False)
    steals: Mapped[int] = mapped_column(default=0, nullable=False)
    turnovers: Mapped[int] = mapped_column(default=0, nullable=False)
    blocks_favour: Mapped[int] = mapped_column(default=0, nullable=False)
    blocks_against: Mapped[int] = mapped_column(default=0, nullable=False)
    fouls_committed: Mapped[int] = mapped_column(default=0, nullable=False)
    fouls_received: Mapped[int] = mapped_column(default=0, nullable=False)
    valuation: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        doc="Standard European efficiency rating (PIR). A useful baseline target.",
    )
    plus_minus: Mapped[int | None] = mapped_column(nullable=True)

    game_team: Mapped[GameTeam] = relationship(back_populates="player_stats")
    player: Mapped[Player] = relationship()

    @property
    def minutes_played(self) -> float:
        """Seconds rendered as minutes, for presentation only."""
        return round(self.seconds_played / 60.0, 2)


class TeamGameStats(Base, UUIDPrimaryKeyMixin, ObservedMixin, TimestampMixin):
    """Team totals for one side of one game.

    Stored rather than aggregated on demand: opponent-strength and pace features
    query these for every prior game of both teams, and re-aggregating box scores
    each time would dominate feature build cost for no benefit.
    """

    __tablename__ = "team_game_stats"
    __table_args__ = (UniqueConstraint("game_team_id", name="uq_team_game_stats_game_team"),)

    game_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("game_teams.id", ondelete="CASCADE"),
        nullable=False,
    )

    points: Mapped[int] = mapped_column(default=0, nullable=False)
    fg2m: Mapped[int] = mapped_column(default=0, nullable=False)
    fg2a: Mapped[int] = mapped_column(default=0, nullable=False)
    fg3m: Mapped[int] = mapped_column(default=0, nullable=False)
    fg3a: Mapped[int] = mapped_column(default=0, nullable=False)
    ftm: Mapped[int] = mapped_column(default=0, nullable=False)
    fta: Mapped[int] = mapped_column(default=0, nullable=False)
    offensive_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    defensive_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    total_rebounds: Mapped[int] = mapped_column(default=0, nullable=False)
    assists: Mapped[int] = mapped_column(default=0, nullable=False)
    steals: Mapped[int] = mapped_column(default=0, nullable=False)
    turnovers: Mapped[int] = mapped_column(default=0, nullable=False)
    blocks_favour: Mapped[int] = mapped_column(default=0, nullable=False)
    blocks_against: Mapped[int] = mapped_column(default=0, nullable=False)
    fouls_committed: Mapped[int] = mapped_column(default=0, nullable=False)
    fouls_received: Mapped[int] = mapped_column(default=0, nullable=False)
    valuation: Mapped[int] = mapped_column(default=0, nullable=False)

    # Derived at ingest from the standard estimator:
    #   FGA - OREB + TO + 0.44 * FTA
    possessions: Mapped[float | None] = mapped_column(Numeric(8, 3), nullable=True)
    pace: Mapped[float | None] = mapped_column(
        Numeric(8, 3), nullable=True, doc="Possessions per 40 minutes."
    )

    game_team: Mapped[GameTeam] = relationship(back_populates="team_stats")

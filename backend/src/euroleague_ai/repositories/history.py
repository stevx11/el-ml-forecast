"""Time-filtered reads. **The leakage choke point.**

Every query a feature builder needs lives here, and every one of them takes
``as_of`` as a mandatory keyword argument with no default. Feature builders never
write SQL, which means the leakage-critical predicate lives in a handful of
reviewable functions instead of being scattered across dozens of feature
definitions.

The rule these functions implement::

    WHERE observed_at <= :as_of     -- correctness: what we had actually recorded
      AND <event time> <  :as_of    -- sanity: the event had also happened

``observed_at`` is the one that must never be omitted. A report published at
19:00 describing a 10:00 injury was not knowable at 17:00, and filtering only on
the event time would silently leak it into a 17:00 prediction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, GameStatus, TeamSide
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.people import PlayerTeamStint


@dataclass(frozen=True, slots=True)
class PlayerGameLine:
    """A player's line from one prior game, with the context features need.

    A flat, framework-free record rather than an ORM object so that feature code
    cannot accidentally lazy-load a relationship and pull in post-cutoff data
    through the side door.
    """

    game_id: uuid.UUID
    season_id: uuid.UUID
    tip_off_utc: datetime
    team_id: uuid.UUID
    opponent_team_id: uuid.UUID
    is_home: bool
    seconds_played: int
    did_not_play: bool
    is_starter: bool
    points: int
    total_rebounds: int
    assists: int
    fg3m: int
    fg3a: int
    fg2a: int
    fta: int
    steals: int
    blocks_favour: int
    turnovers: int
    valuation: int

    @property
    def minutes(self) -> float:
        return self.seconds_played / 60.0

    @property
    def field_goal_attempts(self) -> int:
        return self.fg2a + self.fg3a


def _as_of_guard(as_of: datetime) -> datetime:
    """Normalise the cutoff, rejecting naive datetimes.

    A naive cutoff compared against timezone-aware ``observed_at`` values either
    raises or -- worse, under some drivers -- compares wrongly. Rejecting it here
    turns a subtle data bug into an immediate, obvious failure.
    """
    return ensure_utc(as_of, field="as_of")


def _observed_before(as_of: datetime) -> Select[tuple[PlayerGameStats]]:
    """Base select over box score lines visible at ``as_of``."""
    return (
        select(PlayerGameStats)
        .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
        .join(Game, Game.id == GameTeam.game_id)
        .where(
            PlayerGameStats.observed_at <= as_of,
            Game.observed_at <= as_of,
            # The game must also have finished before the cutoff. Without this a
            # box score ingested early -- a live feed, a clock skew -- would leak
            # the result of a game that had not yet tipped off.
            Game.tip_off_utc < as_of,
            Game.status == GameStatus.FINISHED,
        )
    )


class PlayerHistoryRepository:
    """Time-filtered access to a player's past."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def recent_games(
        self,
        player_id: uuid.UUID,
        *,
        as_of: datetime,
        limit: int = 20,
        include_dnp: bool = False,
        season_id: uuid.UUID | None = None,
    ) -> list[PlayerGameLine]:
        """Return the player's most recent games before ``as_of``, newest first.

        Args:
            as_of: Cutoff. **Mandatory.** Nothing recorded after this instant
                may influence the result.
            include_dnp: Whether to include games the player did not play.
                Defaults to ``False`` because form windows should average over
                games actually played -- counting a DNP as a zero-point game
                teaches the model that an injured star is a low-usage player.
        """
        as_of = _as_of_guard(as_of)

        opponent = GameTeam.__table__.alias("opponent")

        stmt = (
            _observed_before(as_of)
            .add_columns(
                Game.id,
                Game.season_id,
                Game.tip_off_utc,
                GameTeam.team_id,
                GameTeam.side,
                opponent.c.team_id.label("opponent_team_id"),
            )
            .join(
                opponent,
                and_(opponent.c.game_id == Game.id, opponent.c.side != GameTeam.side),
            )
            .where(PlayerGameStats.player_id == player_id)
            .order_by(Game.tip_off_utc.desc())
            .limit(limit)
        )
        if not include_dnp:
            stmt = stmt.where(PlayerGameStats.did_not_play.is_(False))
        if season_id is not None:
            stmt = stmt.where(Game.season_id == season_id)

        rows = (await self._session.execute(stmt)).all()
        return [
            PlayerGameLine(
                game_id=game_id,
                season_id=season,
                tip_off_utc=tip_off,
                team_id=team_id,
                opponent_team_id=opponent_team_id,
                is_home=side is TeamSide.HOME,
                seconds_played=stats.seconds_played,
                did_not_play=stats.did_not_play,
                is_starter=stats.is_starter,
                points=stats.points,
                total_rebounds=stats.total_rebounds,
                assists=stats.assists,
                fg3m=stats.fg3m,
                fg3a=stats.fg3a,
                fg2a=stats.fg2a,
                fta=stats.fta,
                steals=stats.steals,
                blocks_favour=stats.blocks_favour,
                turnovers=stats.turnovers,
                valuation=stats.valuation,
            )
            for stats, game_id, season, tip_off, team_id, side, opponent_team_id in rows
        ]

    async def games_played_between(
        self,
        player_id: uuid.UUID,
        *,
        as_of: datetime,
        since: datetime,
    ) -> int:
        """Count games played in ``[since, as_of)``. Backs schedule-density features."""
        as_of = _as_of_guard(as_of)
        stmt = (
            _observed_before(as_of)
            .with_only_columns(func.count())
            .where(
                PlayerGameStats.player_id == player_id,
                PlayerGameStats.did_not_play.is_(False),
                Game.tip_off_utc >= ensure_utc(since, field="since"),
            )
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def status_as_of(
        self, player_id: uuid.UUID, *, as_of: datetime
    ) -> AvailabilityStatus:
        """Return the player's most recent known availability at ``as_of``.

        This is the query the worked example in ``docs/data-model.md`` describes.
        Given ``10:00 OUT``, ``14:00 QUESTIONABLE``, ``18:30 AVAILABLE``, asking
        at 17:00 returns ``QUESTIONABLE`` -- the 18:30 row exists in the table but
        is invisible at this cutoff.

        Returns ``UNKNOWN`` when nothing has been observed. That is deliberately
        distinct from ``AVAILABLE``: absence of evidence is not evidence of
        availability, and conflating them would treat a silent roster as a
        healthy one.
        """
        as_of = _as_of_guard(as_of)
        stmt = (
            select(PlayerStatusEvent.status)
            .where(
                PlayerStatusEvent.player_id == player_id,
                PlayerStatusEvent.observed_at <= as_of,
                PlayerStatusEvent.event_timestamp <= as_of,
            )
            .order_by(
                PlayerStatusEvent.event_timestamp.desc(),
                PlayerStatusEvent.observed_at.desc(),
            )
            .limit(1)
        )
        status = (await self._session.execute(stmt)).scalar_one_or_none()
        return status if status is not None else AvailabilityStatus.UNKNOWN

    async def team_as_of(
        self, player_id: uuid.UUID, *, as_of: datetime
    ) -> uuid.UUID | None:
        """Return the team the player was registered with at ``as_of``.

        Queries the stint intervals rather than a mutable column, so a February
        transfer does not retroactively change which club the player belonged to
        in November.
        """
        as_of = _as_of_guard(as_of)
        stmt = (
            select(PlayerTeamStint.team_id)
            .where(
                PlayerTeamStint.player_id == player_id,
                PlayerTeamStint.observed_at <= as_of,
                PlayerTeamStint.effective_from <= as_of,
                (PlayerTeamStint.effective_until.is_(None))
                | (PlayerTeamStint.effective_until > as_of),
            )
            .order_by(PlayerTeamStint.effective_from.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


@dataclass(frozen=True, slots=True)
class SquadAppearance:
    """One player's appearance (or non-appearance) in one prior team game."""

    game_id: uuid.UUID
    tip_off_utc: datetime
    player_id: uuid.UUID
    seconds_played: int
    did_not_play: bool
    is_starter: bool

    @property
    def minutes(self) -> float:
        return self.seconds_played / 60.0

    @property
    def played(self) -> bool:
        return not self.did_not_play and self.seconds_played > 0


@dataclass(frozen=True, slots=True)
class TeamGameLine:
    """A team's totals from one prior game."""

    game_id: uuid.UUID
    tip_off_utc: datetime
    opponent_team_id: uuid.UUID
    is_home: bool
    points: int
    points_allowed: int
    possessions: float | None
    pace: float | None
    offensive_rebounds: int
    defensive_rebounds: int
    turnovers: int
    fg3a: int
    fta: int


class TeamHistoryRepository:
    """Time-filtered access to a team's past. Backs team and opponent features."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def recent_games(
        self,
        team_id: uuid.UUID,
        *,
        as_of: datetime,
        limit: int = 20,
        season_id: uuid.UUID | None = None,
    ) -> list[TeamGameLine]:
        """Return the team's most recent finished games before ``as_of``.

        Opponent strength must be computed from these rows rather than from a
        precomputed season-long rating. Reusing one season aggregate for every
        game is the classic leak in this domain: March's defensive rating would
        inform an October prediction.
        """
        as_of = _as_of_guard(as_of)

        opponent = GameTeam.__table__.alias("opponent")
        opponent_stats = TeamGameStats.__table__.alias("opponent_stats")

        stmt = (
            select(
                Game.id,
                Game.tip_off_utc,
                GameTeam.side,
                opponent.c.team_id.label("opponent_team_id"),
                TeamGameStats.points,
                opponent_stats.c.points.label("points_allowed"),
                TeamGameStats.possessions,
                TeamGameStats.pace,
                TeamGameStats.offensive_rebounds,
                TeamGameStats.defensive_rebounds,
                TeamGameStats.turnovers,
                TeamGameStats.fg3a,
                TeamGameStats.fta,
            )
            .join(GameTeam, GameTeam.id == TeamGameStats.game_team_id)
            .join(Game, Game.id == GameTeam.game_id)
            .join(
                opponent,
                and_(opponent.c.game_id == Game.id, opponent.c.side != GameTeam.side),
            )
            .join(opponent_stats, opponent_stats.c.game_team_id == opponent.c.id)
            .where(
                GameTeam.team_id == team_id,
                TeamGameStats.observed_at <= as_of,
                Game.observed_at <= as_of,
                Game.tip_off_utc < as_of,
                Game.status == GameStatus.FINISHED,
            )
            .order_by(Game.tip_off_utc.desc())
            .limit(limit)
        )
        if season_id is not None:
            stmt = stmt.where(Game.season_id == season_id)

        return [
            TeamGameLine(
                game_id=row.id,
                tip_off_utc=row.tip_off_utc,
                opponent_team_id=row.opponent_team_id,
                is_home=row.side is TeamSide.HOME,
                points=row.points,
                points_allowed=row.points_allowed,
                possessions=float(row.possessions) if row.possessions is not None else None,
                pace=float(row.pace) if row.pace is not None else None,
                offensive_rebounds=row.offensive_rebounds,
                defensive_rebounds=row.defensive_rebounds,
                turnovers=row.turnovers,
                fg3a=row.fg3a,
                fta=row.fta,
            )
            for row in (await self._session.execute(stmt)).all()
        ]

    async def squad_appearances(
        self,
        team_id: uuid.UUID,
        *,
        as_of: datetime,
        games: int = 5,
        season_id: uuid.UUID | None = None,
    ) -> list[SquadAppearance]:
        """Who appeared for this team in its last ``games`` fixtures before ``as_of``.

        This backs the availability-proxy features. Historical pre-game
        availability was never observed -- nobody recorded in 2018 that a player
        was ruled out at 10:00 -- so ``player_status_events`` is empty for every
        past season and always will be. Who actually took the floor in the
        previous game is the best leak-free substitute, because it was known
        before the next game tipped off.

        Returns one row per (player, game), newest game first.
        """
        as_of = _as_of_guard(as_of)

        recent_games = (
            select(Game.id, Game.tip_off_utc)
            .join(GameTeam, GameTeam.game_id == Game.id)
            .where(
                GameTeam.team_id == team_id,
                Game.observed_at <= as_of,
                Game.tip_off_utc < as_of,
                Game.status == GameStatus.FINISHED,
            )
        )
        if season_id is not None:
            recent_games = recent_games.where(Game.season_id == season_id)

        recent = recent_games.order_by(Game.tip_off_utc.desc()).limit(games).subquery()

        stmt = (
            select(
                recent.c.id,
                recent.c.tip_off_utc,
                PlayerGameStats.player_id,
                PlayerGameStats.seconds_played,
                PlayerGameStats.did_not_play,
                PlayerGameStats.is_starter,
            )
            .select_from(recent)
            .join(GameTeam, and_(GameTeam.game_id == recent.c.id, GameTeam.team_id == team_id))
            .join(PlayerGameStats, PlayerGameStats.game_team_id == GameTeam.id)
            .where(PlayerGameStats.observed_at <= as_of)
            .order_by(recent.c.tip_off_utc.desc())
        )

        return [
            SquadAppearance(
                game_id=row[0],
                tip_off_utc=row[1],
                player_id=row[2],
                seconds_played=row[3],
                did_not_play=row[4],
                is_starter=row[5],
            )
            for row in (await self._session.execute(stmt)).all()
        ]

    async def last_game_before(
        self, team_id: uuid.UUID, *, as_of: datetime
    ) -> datetime | None:
        """Tip-off of the team's most recent game before ``as_of``. Backs rest days."""
        as_of = _as_of_guard(as_of)
        stmt = (
            select(func.max(Game.tip_off_utc))
            .join(GameTeam, GameTeam.game_id == Game.id)
            .where(
                GameTeam.team_id == team_id,
                Game.observed_at <= as_of,
                Game.tip_off_utc < as_of,
                Game.status == GameStatus.FINISHED,
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


    async def squad_positions(
        self,
        team_id: uuid.UUID,
        *,
        as_of: datetime,
        season_id: uuid.UUID | None = None,
    ) -> dict[uuid.UUID, str]:
        """Listed position for every player registered with the team at ``as_of``.

        Read from ``PlayerTeamStint`` rather than from ``Player.position``. The
        stint carries ``observed_at`` and ``effective_from``, so a position we
        only learned later cannot leak backwards -- whereas ``Player.position``
        is a mutable column with no history, and reading it would quietly answer
        with today's listing for a game played in 2018.

        Where a player has several stints in effect, the latest ``effective_from``
        wins, matching how club membership is resolved elsewhere.
        """
        as_of = _as_of_guard(as_of)

        stmt = (
            select(
                PlayerTeamStint.player_id,
                PlayerTeamStint.position,
                PlayerTeamStint.effective_from,
            )
            .where(
                PlayerTeamStint.team_id == team_id,
                PlayerTeamStint.observed_at <= as_of,
                PlayerTeamStint.effective_from <= as_of,
            )
            .order_by(PlayerTeamStint.player_id, PlayerTeamStint.effective_from)
        )
        if season_id is not None:
            stmt = stmt.where(PlayerTeamStint.season_id == season_id)

        # Ordered ascending by effective_from, so the last write per player is
        # the most recent registration -- the same tie-break used for clubs.
        positions: dict[uuid.UUID, str] = {}
        for player_id, position, _ in (await self._session.execute(stmt)).all():
            positions[player_id] = str(position)
        return positions

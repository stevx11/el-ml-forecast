"""Current-state reads for the API.

These deliberately have **no** ``as_of`` parameter. They answer "what should the
site show right now?", which is a different question from "what did the model
know before tip-off?". That second question is
:mod:`euroleague_ai.repositories.history`, and keeping the two in separate
modules prevents an API convenience query from being reused inside a feature
builder, where it would leak.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from euroleague_ai.models.competition import League, Season, Team, TeamSeason
from euroleague_ai.models.enums import GameStatus
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.people import Player

MAX_PAGE_SIZE = 200


def _paginate[S: Select[Any]](stmt: S, *, limit: int, offset: int) -> S:
    """Clamp pagination so a client cannot request the entire table."""
    return stmt.limit(min(max(limit, 1), MAX_PAGE_SIZE)).offset(max(offset, 0))


class LeagueRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_leagues(self) -> Sequence[League]:
        stmt = select(League).order_by(League.name)
        return (await self._session.execute(stmt)).scalars().all()

    async def list_seasons(self, league_id: uuid.UUID | None = None) -> Sequence[Season]:
        stmt = select(Season).order_by(Season.start_year.desc())
        if league_id is not None:
            stmt = stmt.where(Season.league_id == league_id)
        return (await self._session.execute(stmt)).scalars().all()

    async def get_season_by_code(self, code: str) -> Season | None:
        stmt = select(Season).where(Season.code == code)
        return (await self._session.execute(stmt)).scalar_one_or_none()


class TeamRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_teams(
        self, *, season_id: uuid.UUID | None = None, limit: int = 50, offset: int = 0
    ) -> Sequence[Team]:
        stmt = select(Team).order_by(Team.name)
        if season_id is not None:
            stmt = stmt.join(TeamSeason, TeamSeason.team_id == Team.id).where(
                TeamSeason.season_id == season_id
            )
        result = await self._session.execute(_paginate(stmt, limit=limit, offset=offset))
        return result.scalars().all()

    async def get(self, team_id: uuid.UUID) -> Team | None:
        return await self._session.get(Team, team_id)

    async def count(self) -> int:
        return int((await self._session.execute(select(func.count(Team.id)))).scalar_one())


class PlayerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_players(
        self, *, search: str | None = None, limit: int = 50, offset: int = 0
    ) -> Sequence[Player]:
        stmt = select(Player).order_by(Player.full_name)
        if search:
            stmt = stmt.where(Player.full_name.ilike(f"%{search}%"))
        return (
            (await self._session.execute(_paginate(stmt, limit=limit, offset=offset)))
            .scalars()
            .all()
        )

    async def get(self, player_id: uuid.UUID) -> Player | None:
        return await self._session.get(Player, player_id)

    async def count(self) -> int:
        return int((await self._session.execute(select(func.count(Player.id)))).scalar_one())

    async def game_log(
        self, player_id: uuid.UUID, *, limit: int = 20
    ) -> Sequence[tuple[PlayerGameStats, Game]]:
        """Most recent box score lines, newest first.

        A presentation query. It has no cutoff and must not be used to build
        features -- :meth:`PlayerHistoryRepository.recent_games` is the
        equivalent with the leakage filter applied.
        """
        stmt = (
            select(PlayerGameStats, Game)
            .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            .join(Game, Game.id == GameTeam.game_id)
            .where(PlayerGameStats.player_id == player_id)
            .order_by(Game.tip_off_utc.desc())
            .limit(min(max(limit, 1), MAX_PAGE_SIZE))
        )
        return [(stats, game) for stats, game in (await self._session.execute(stmt)).all()]


class GameRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_games(
        self,
        *,
        season_id: uuid.UUID | None = None,
        team_id: uuid.UUID | None = None,
        status: GameStatus | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Game]:
        stmt = (
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .order_by(Game.tip_off_utc.desc())
        )
        if season_id is not None:
            stmt = stmt.where(Game.season_id == season_id)
        if status is not None:
            stmt = stmt.where(Game.status == status)
        if date_from is not None:
            stmt = stmt.where(Game.tip_off_utc >= date_from)
        if date_to is not None:
            stmt = stmt.where(Game.tip_off_utc <= date_to)
        if team_id is not None:
            stmt = stmt.where((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
        return (
            (await self._session.execute(_paginate(stmt, limit=limit, offset=offset)))
            .scalars()
            .all()
        )

    async def get(self, game_id: uuid.UUID) -> Game | None:
        stmt = (
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id == game_id)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def box_score(
        self, game_id: uuid.UUID
    ) -> Sequence[tuple[PlayerGameStats, GameTeam, Player]]:
        stmt = (
            select(PlayerGameStats, GameTeam, Player)
            .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            .join(Player, Player.id == PlayerGameStats.player_id)
            .where(GameTeam.game_id == game_id)
            .order_by(GameTeam.side, PlayerGameStats.seconds_played.desc())
        )
        return [tuple(row) for row in (await self._session.execute(stmt)).all()]  # type: ignore[misc]

    async def count(self) -> int:
        return int((await self._session.execute(select(func.count(Game.id)))).scalar_one())

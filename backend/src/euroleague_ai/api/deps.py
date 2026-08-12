"""FastAPI dependencies.

Thin wiring only. Anything with real logic belongs in a service or repository so
it stays testable without an HTTP client.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.db.session import get_session
from euroleague_ai.repositories.catalog import (
    GameRepository,
    LeagueRepository,
    PlayerRepository,
    TeamRepository,
)

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_league_repository(session: SessionDep) -> LeagueRepository:
    return LeagueRepository(session)


def get_team_repository(session: SessionDep) -> TeamRepository:
    return TeamRepository(session)


def get_player_repository(session: SessionDep) -> PlayerRepository:
    return PlayerRepository(session)


def get_game_repository(session: SessionDep) -> GameRepository:
    return GameRepository(session)


LeagueRepoDep = Annotated[LeagueRepository, Depends(get_league_repository)]
TeamRepoDep = Annotated[TeamRepository, Depends(get_team_repository)]
PlayerRepoDep = Annotated[PlayerRepository, Depends(get_player_repository)]
GameRepoDep = Annotated[GameRepository, Depends(get_game_repository)]

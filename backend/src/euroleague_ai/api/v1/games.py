"""Game endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from euroleague_ai.api.deps import GameRepoDep
from euroleague_ai.models.enums import GameStatus
from euroleague_ai.schemas.common import Page, PageMeta
from euroleague_ai.schemas.game import BoxScoreLineOut, GameDetailOut, GameOut

router = APIRouter(prefix="/games", tags=["games"])


@router.get("", response_model=Page[GameOut])
async def list_games(
    repo: GameRepoDep,
    season_id: uuid.UUID | None = Query(default=None),
    team_id: uuid.UUID | None = Query(default=None),
    game_status: GameStatus | None = Query(default=None, alias="status"),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[GameOut]:
    """Fixtures, newest first."""
    rows = await repo.list_games(
        season_id=season_id,
        team_id=team_id,
        status=game_status,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[GameOut.model_validate(row) for row in rows],
        meta=PageMeta(limit=limit, offset=offset, returned=len(rows)),
    )


@router.get("/{game_id}", response_model=GameDetailOut)
async def get_game(game_id: uuid.UUID, repo: GameRepoDep) -> GameDetailOut:
    """One fixture, with its box score when the game has been played."""
    game = await repo.get(game_id)
    if game is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Game not found")

    detail = GameDetailOut.model_validate(game)
    detail.box_score = [
        BoxScoreLineOut(
            player_id=player.id,
            player_name=player.full_name,
            side=game_team.side,
            seconds_played=stats.seconds_played,
            is_starter=stats.is_starter,
            did_not_play=stats.did_not_play,
            points=stats.points,
            fg2m=stats.fg2m,
            fg2a=stats.fg2a,
            fg3m=stats.fg3m,
            fg3a=stats.fg3a,
            ftm=stats.ftm,
            fta=stats.fta,
            offensive_rebounds=stats.offensive_rebounds,
            defensive_rebounds=stats.defensive_rebounds,
            total_rebounds=stats.total_rebounds,
            assists=stats.assists,
            steals=stats.steals,
            turnovers=stats.turnovers,
            blocks_favour=stats.blocks_favour,
            fouls_committed=stats.fouls_committed,
            valuation=stats.valuation,
            plus_minus=stats.plus_minus,
        )
        for stats, game_team, player in await repo.box_score(game_id)
    ]
    return detail

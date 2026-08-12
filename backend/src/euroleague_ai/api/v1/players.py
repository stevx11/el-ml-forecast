"""Player endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from euroleague_ai.api.deps import PlayerRepoDep
from euroleague_ai.schemas.common import Page, PageMeta
from euroleague_ai.schemas.player import PlayerGameLogOut, PlayerOut

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=Page[PlayerOut])
async def list_players(
    repo: PlayerRepoDep,
    search: str | None = Query(default=None, min_length=2, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[PlayerOut]:
    """Players, optionally filtered by a case-insensitive name substring."""
    rows = await repo.list_players(search=search, limit=limit, offset=offset)
    return Page(
        items=[PlayerOut.model_validate(row) for row in rows],
        meta=PageMeta(limit=limit, offset=offset, returned=len(rows)),
    )


@router.get("/{player_id}", response_model=PlayerOut)
async def get_player(player_id: uuid.UUID, repo: PlayerRepoDep) -> PlayerOut:
    player = await repo.get(player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Player not found")
    return PlayerOut.model_validate(player)


@router.get("/{player_id}/stats", response_model=list[PlayerGameLogOut])
async def player_game_log(
    player_id: uuid.UUID,
    repo: PlayerRepoDep,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[PlayerGameLogOut]:
    """Recent box score lines, newest first.

    A presentation view with no cutoff applied. It is not the feature-layer
    history query, which requires an explicit ``as_of``.
    """
    player = await repo.get(player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Player not found")

    return [
        PlayerGameLogOut(
            game_id=game.id,
            tip_off_utc=game.tip_off_utc,
            seconds_played=stats.seconds_played,
            did_not_play=stats.did_not_play,
            is_starter=stats.is_starter,
            points=stats.points,
            total_rebounds=stats.total_rebounds,
            assists=stats.assists,
            fg3m=stats.fg3m,
            steals=stats.steals,
            blocks_favour=stats.blocks_favour,
            turnovers=stats.turnovers,
            valuation=stats.valuation,
        )
        for stats, game in await repo.game_log(player_id, limit=limit)
    ]

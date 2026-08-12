"""League and season endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from euroleague_ai.api.deps import LeagueRepoDep
from euroleague_ai.schemas.competition import LeagueOut, SeasonOut

router = APIRouter(prefix="/leagues", tags=["leagues"])


@router.get("", response_model=list[LeagueOut])
async def list_leagues(repo: LeagueRepoDep) -> list[LeagueOut]:
    """Every competition known to the system."""
    return [LeagueOut.model_validate(row) for row in await repo.list_leagues()]


@router.get("/seasons", response_model=list[SeasonOut])
async def list_seasons(
    repo: LeagueRepoDep,
    league_id: uuid.UUID | None = Query(default=None),
) -> list[SeasonOut]:
    """Seasons, newest first, optionally filtered by league."""
    return [SeasonOut.model_validate(row) for row in await repo.list_seasons(league_id)]

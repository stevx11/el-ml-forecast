"""Team endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from euroleague_ai.api.deps import TeamRepoDep
from euroleague_ai.schemas.common import Page, PageMeta
from euroleague_ai.schemas.competition import TeamOut

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("", response_model=Page[TeamOut])
async def list_teams(
    repo: TeamRepoDep,
    season_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[TeamOut]:
    """Clubs, optionally restricted to those registered in a season."""
    rows = await repo.list_teams(season_id=season_id, limit=limit, offset=offset)
    return Page(
        items=[TeamOut.model_validate(row) for row in rows],
        meta=PageMeta(limit=limit, offset=offset, returned=len(rows)),
    )


@router.get("/{team_id}", response_model=TeamOut)
async def get_team(team_id: uuid.UUID, repo: TeamRepoDep) -> TeamOut:
    team = await repo.get(team_id)
    if team is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Team not found")
    return TeamOut.model_validate(team)

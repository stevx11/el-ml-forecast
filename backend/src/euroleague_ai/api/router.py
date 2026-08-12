"""Top-level API router.

Only endpoints backed by functionality that actually exists are mounted. There
are no stubs for predictions or news: an endpoint that returns an empty list
because the feature was never built is worse than no endpoint, because a client
cannot tell the difference between "no data" and "not implemented".
"""

from fastapi import APIRouter

from euroleague_ai.api.v1 import games, leagues, players, teams

api_router = APIRouter()
api_router.include_router(leagues.router)
api_router.include_router(teams.router)
api_router.include_router(players.router)
api_router.include_router(games.router)

__all__ = ["api_router"]

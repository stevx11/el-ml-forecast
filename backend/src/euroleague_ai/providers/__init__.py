"""Adapters to external basketball data sources.

Everything downstream is written against :class:`BasketballDataProvider` and the
DTOs in :mod:`euroleague_ai.providers.base`, never against provider JSON.
"""

from euroleague_ai.providers.base import (
    BasketballDataProvider,
    GameDTO,
    GameStatsDTO,
    PlayerLineDTO,
    PlayerStintDTO,
    SeasonRef,
    TeamDTO,
    TeamLineDTO,
    TeamStatsDTO,
)
from euroleague_ai.providers.euroleague import EuroLeagueProvider, build_response_cache
from euroleague_ai.providers.factory import build_provider
from euroleague_ai.providers.fixture import FixtureProvider

__all__ = [
    "BasketballDataProvider",
    "EuroLeagueProvider",
    "FixtureProvider",
    "GameDTO",
    "GameStatsDTO",
    "PlayerLineDTO",
    "PlayerStintDTO",
    "SeasonRef",
    "TeamDTO",
    "TeamLineDTO",
    "TeamStatsDTO",
    "build_provider",
    "build_response_cache",
]

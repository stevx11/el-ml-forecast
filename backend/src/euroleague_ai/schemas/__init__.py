"""Pydantic request and response models.

Separate from the ORM on purpose. Serialising ORM objects directly couples the
public API to the storage schema and makes it easy to leak a column that was
never meant to be exposed.
"""

from euroleague_ai.schemas.common import Page, PageMeta
from euroleague_ai.schemas.competition import LeagueOut, SeasonOut, TeamOut
from euroleague_ai.schemas.game import BoxScoreLineOut, GameDetailOut, GameOut
from euroleague_ai.schemas.health import HealthOut, ReadinessOut
from euroleague_ai.schemas.player import PlayerGameLogOut, PlayerOut
from euroleague_ai.schemas.prediction import (
    PlayerProjectionOut,
    PredictionOut,
    ProjectionsOut,
)

__all__ = [
    "BoxScoreLineOut",
    "GameDetailOut",
    "GameOut",
    "HealthOut",
    "LeagueOut",
    "Page",
    "PageMeta",
    "PlayerGameLogOut",
    "PlayerOut",
    "PlayerProjectionOut",
    "PredictionOut",
    "ProjectionsOut",
    "ReadinessOut",
    "SeasonOut",
    "TeamOut",
]

"""SQLAlchemy ORM models.

Importing this package registers every table on ``Base.metadata``. Alembic's
``env.py`` imports it for exactly that reason -- without it, autogenerate sees an
empty metadata and cheerfully proposes dropping the entire schema.
"""

from euroleague_ai.models.availability import PlayerStatusEvent, StartingLineupEntry
from euroleague_ai.models.competition import League, Season, Team, TeamSeason
from euroleague_ai.models.enums import (
    AvailabilityStatus,
    EntityType,
    GameStatus,
    IngestionStatus,
    NewsEventType,
    PlayerPosition,
    PredictionTarget,
    StatusSourceType,
    TeamSide,
)
from euroleague_ai.models.features import PlayerFeatureSnapshot, TeamFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.models.ingestion import IngestionRun
from euroleague_ai.models.news import NewsArticle, NewsEvent, NewsSource
from euroleague_ai.models.people import Coach, Player, PlayerTeamStint, TeamCoachStint
from euroleague_ai.models.prediction import ModelVersion, Prediction, PredictionOutcome

__all__ = [
    "AvailabilityStatus",
    "Coach",
    "EntityType",
    "ExternalRef",
    "Game",
    "GameStatus",
    "GameTeam",
    "IngestionRun",
    "IngestionStatus",
    "League",
    "ModelVersion",
    "NewsArticle",
    "NewsEvent",
    "NewsEventType",
    "NewsSource",
    "Player",
    "PlayerFeatureSnapshot",
    "PlayerGameStats",
    "PlayerPosition",
    "PlayerStatusEvent",
    "PlayerTeamStint",
    "Prediction",
    "PredictionOutcome",
    "PredictionTarget",
    "Season",
    "StartingLineupEntry",
    "StatusSourceType",
    "Team",
    "TeamCoachStint",
    "TeamFeatureSnapshot",
    "TeamGameStats",
    "TeamSeason",
    "TeamSide",
]

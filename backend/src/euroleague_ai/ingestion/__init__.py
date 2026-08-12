"""Provider output into the time-aware store."""

from euroleague_ai.ingestion.identity import IdentityResolver
from euroleague_ai.ingestion.jobs import (
    ingest_all,
    ingest_game_stats,
    ingest_games,
    ingest_players,
    ingest_teams,
)
from euroleague_ai.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "IdentityResolver",
    "IngestionPipeline",
    "IngestionResult",
    "ingest_all",
    "ingest_game_stats",
    "ingest_games",
    "ingest_players",
    "ingest_teams",
]

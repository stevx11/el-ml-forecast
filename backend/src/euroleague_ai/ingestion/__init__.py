"""Provider output into the time-aware store."""

from euroleague_ai.ingestion.identity import IdentityResolver
from euroleague_ai.ingestion.jobs import (
    ingest_all,
    ingest_game_stats,
    ingest_games,
    ingest_players,
    ingest_teams,
)
from euroleague_ai.ingestion.knowability import (
    box_score_observed_at,
    fixture_observed_at,
    stint_observed_at,
)
from euroleague_ai.ingestion.pipeline import IngestionPipeline, IngestionResult
from euroleague_ai.ingestion.repair import (
    RepairResult,
    purge_feature_snapshots,
    repair_observed_at,
)

__all__ = [
    "IdentityResolver",
    "IngestionPipeline",
    "IngestionResult",
    "RepairResult",
    "box_score_observed_at",
    "fixture_observed_at",
    "ingest_all",
    "ingest_game_stats",
    "ingest_games",
    "ingest_players",
    "ingest_teams",
    "purge_feature_snapshots",
    "repair_observed_at",
    "stint_observed_at",
]

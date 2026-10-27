"""Feature computation. The leakage boundary.

Every builder takes ``as_of`` as a mandatory keyword argument and reaches the
database only through :mod:`euroleague_ai.repositories.history`, which applies
the ``observed_at <= as_of`` predicate. Feature code never writes SQL.

:mod:`~euroleague_ai.features.registry` composes the individual families into the
versioned feature set that snapshots and models refer to.
"""

from euroleague_ai.features.availability import build_availability_features
from euroleague_ai.features.backfill import BackfillResult, backfill_snapshots
from euroleague_ai.features.player_form import build_player_form_features
from euroleague_ai.features.registry import (
    FEATURE_FAMILIES,
    FEATURE_SET_VERSION,
    build_player_features,
    feature_names,
)
from euroleague_ai.features.snapshot import (
    build_and_store_player_snapshot,
    default_prediction_timestamp,
)
from euroleague_ai.features.team_context import build_team_context_features

__all__ = [
    "FEATURE_FAMILIES",
    "FEATURE_SET_VERSION",
    "BackfillResult",
    "backfill_snapshots",
    "build_and_store_player_snapshot",
    "build_availability_features",
    "build_player_features",
    "build_player_form_features",
    "build_team_context_features",
    "default_prediction_timestamp",
    "feature_names",
]

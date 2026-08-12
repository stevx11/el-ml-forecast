"""Feature computation. The leakage boundary.

Every builder here takes ``as_of`` as a mandatory keyword argument and reaches
the database only through :mod:`euroleague_ai.repositories.history`, which
applies the ``observed_at <= as_of`` predicate. Feature code never writes SQL.
"""

from euroleague_ai.features.player_form import (
    FEATURE_SET_VERSION,
    build_player_form_features,
)
from euroleague_ai.features.snapshot import (
    build_and_store_player_snapshot,
    default_prediction_timestamp,
)

__all__ = [
    "FEATURE_SET_VERSION",
    "build_and_store_player_snapshot",
    "build_player_form_features",
    "default_prediction_timestamp",
]

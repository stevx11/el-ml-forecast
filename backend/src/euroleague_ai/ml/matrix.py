"""Feature matrix assembly.

Turns :class:`~euroleague_ai.ml.dataset.TrainingRow` lists into numeric arrays
for LightGBM.

**Column order is pinned and persisted with the model.** Feature vectors are
stored as JSONB, so their key order is not guaranteed and new families get added
over time. If training and inference derived column order independently, a model
trained with ``opp_def_rating`` at index 40 could be served a vector with
``days_rest`` there -- silently, producing plausible-looking nonsense. The order
is therefore computed once at training time, written into the model registry, and
replayed at inference.

Missing values stay ``NaN``. LightGBM learns a default split direction for
missing data natively, which is strictly better than any imputation this layer
could invent. A rookie with no ten-game average is genuinely unknown, and telling
the model "zero" would be a lie it would faithfully learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TrainingRow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = get_logger(__name__)

#: Keys that must never reach a model.
#:
#: These are row metadata, not features. ``round_number`` is the dangerous one:
#: it correlates with everything seasonal and lets a tree memorise the calendar
#: instead of learning basketball. The others are identifiers with no meaning as
#: numbers.
EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "round_number",
        "season_code",
        "game_id",
        "player_id",
        "snapshot_id",
        "tip_off_utc",
        "prediction_timestamp",
        "actual_minutes",
    }
)


def derive_feature_names(rows: Sequence[TrainingRow]) -> list[str]:
    """Return the sorted union of feature keys across ``rows``.

    Union rather than intersection, and sorted for determinism. Using the first
    row's keys alone would silently drop any feature that happens to be absent
    from it; sorting means two runs over the same data always produce the same
    column order.
    """
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    return sorted(names - EXCLUDED_KEYS)


def to_matrix(
    rows: Sequence[TrainingRow], feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` arrays in the given column order.

    Args:
        rows: Training rows.
        feature_names: Column order. Must be the order stored with the model.

    Returns:
        ``X`` of shape ``(len(rows), len(feature_names))`` with ``NaN`` for
        missing values, and ``y`` of shape ``(len(rows),)``.
    """
    import numpy as np

    x = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float64)
    y = np.empty(len(rows), dtype=np.float64)

    for row_index, row in enumerate(rows):
        features = row.features
        for column_index, name in enumerate(feature_names):
            value = features.get(name)
            if value is not None:
                try:
                    x[row_index, column_index] = float(value)
                except (TypeError, ValueError):
                    # Leave NaN. A non-numeric feature value is a bug upstream,
                    # but poisoning the matrix with 0.0 would hide it.
                    logger.warning(
                        "non-numeric feature value",
                        extra={"feature": name, "value": repr(value)},
                    )
        y[row_index] = row.target

    return x, y


def features_to_matrix(
    feature_dicts: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    """Build ``X`` from raw feature dicts, without needing labelled rows.

    Inference has features but no target, so it cannot go through
    :func:`to_matrix`. Both paths share this ordering logic so training and
    serving cannot drift apart in how a vector is laid out.
    """
    import numpy as np

    x = np.full((len(feature_dicts), len(feature_names)), np.nan, dtype=np.float64)
    for row_index, features in enumerate(feature_dicts):
        for column_index, name in enumerate(feature_names):
            value = features.get(name)
            if value is not None:
                try:
                    x[row_index, column_index] = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "non-numeric feature value",
                        extra={"feature": name, "value": repr(value)},
                    )
    return x


def matrix_summary(x: np.ndarray, feature_names: Sequence[str]) -> dict[str, Any]:
    """Diagnostics for a built matrix.

    ``all_missing`` is the one to watch: a feature that is null for every row
    contributes nothing and usually means a family failed to build -- exactly
    the failure that produced 66,000 empty snapshots earlier.
    """
    import numpy as np

    missing_fraction = np.isnan(x).mean(axis=0)
    all_missing = [
        name for name, fraction in zip(feature_names, missing_fraction, strict=True)
        if fraction == 1.0
    ]
    mostly_missing = [
        (name, round(float(fraction), 3))
        for name, fraction in zip(feature_names, missing_fraction, strict=True)
        if 0.5 <= fraction < 1.0
    ]

    return {
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "mean_missing_fraction": round(float(missing_fraction.mean()), 4),
        "all_missing": all_missing,
        "mostly_missing": sorted(mostly_missing, key=lambda item: -item[1])[:10],
    }

"""Machine learning: datasets, baselines, evaluation.

Phase 2 of the roadmap. Everything here is pure Python and runs in the standard
backend image -- the heavy wheels (LightGBM, numpy, pandas) only arrive with
Phase 3, and only for training.

The governing rule: a model that does not beat
:class:`~euroleague_ai.ml.baselines.MinutesScaledRate` on walk-forward
validation is not an improvement, whatever its offline metrics say.
"""

from euroleague_ai.ml.baselines import (
    Baseline,
    LastNMean,
    MinutesScaledRate,
    SeasonMean,
    WeightedBlend,
    baseline_coverage,
    default_baselines,
)
from euroleague_ai.ml.dataset import (
    TARGET_COLUMNS,
    TARGET_FEATURE_STEMS,
    TrainingRow,
    describe_rows,
    load_training_rows,
)
from euroleague_ai.ml.evaluation import (
    Fold,
    Metrics,
    backtest,
    compare,
    compute_metrics,
    stratified_metrics,
    walk_forward_folds,
)
from euroleague_ai.ml.inference import (
    LoadedModel,
    load_active_models,
    predict_upcoming,
)
from euroleague_ai.ml.matrix import derive_feature_names, features_to_matrix, to_matrix
from euroleague_ai.repositories.rosters import candidate_players

__all__ = [
    "TARGET_COLUMNS",
    "TARGET_FEATURE_STEMS",
    "Baseline",
    "Fold",
    "LastNMean",
    "LoadedModel",
    "Metrics",
    "MinutesScaledRate",
    "SeasonMean",
    "TrainingRow",
    "WeightedBlend",
    "backtest",
    "baseline_coverage",
    "candidate_players",
    "compare",
    "compute_metrics",
    "default_baselines",
    "derive_feature_names",
    "describe_rows",
    "features_to_matrix",
    "load_active_models",
    "load_training_rows",
    "predict_upcoming",
    "stratified_metrics",
    "to_matrix",
    "walk_forward_folds",
]

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

__all__ = [
    "TARGET_COLUMNS",
    "TARGET_FEATURE_STEMS",
    "Baseline",
    "Fold",
    "LastNMean",
    "Metrics",
    "MinutesScaledRate",
    "SeasonMean",
    "TrainingRow",
    "WeightedBlend",
    "backtest",
    "baseline_coverage",
    "compare",
    "compute_metrics",
    "default_baselines",
    "describe_rows",
    "load_training_rows",
    "stratified_metrics",
    "walk_forward_folds",
]

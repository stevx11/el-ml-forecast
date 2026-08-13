"""LightGBM training with walk-forward validation.

Design decisions worth stating, because each is a place this could go wrong:

**L1 objective, not L2.** Box score distributions are right-skewed. Squared error
is dominated by the occasional 40-point night, so an L2 model sacrifices accuracy
on the 90% of predictions people actually use in order to chase outliers. MAE is
the reported metric, so MAE is what we optimise.

**Chronological holdout, never a random one.** The final fold is held out by
time. A random validation split would put a player's adjacent games on both
sides, so early stopping would trigger on interpolation performance and the model
would be tuned to a task it will never face.

**Minutes first.** Every other box score target is conditional on playing time,
and the Phase 2 baselines confirmed minutes are by far the most predictable
target (R2 0.48 vs 0.31 for points). Downstream targets can consume predicted
minutes as an input once the minutes model exists.

**The gate is enforced in code.** A model that fails to beat its baseline is
saved but never activated. Publishing a model on the strength of a metric nobody
compared to anything is the failure this whole phase exists to prevent.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.ml.baselines import default_baselines
from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.evaluation import Metrics, compute_metrics, stratified_metrics
from euroleague_ai.ml.matrix import derive_feature_names, matrix_summary, to_matrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightgbm as lgb

logger = get_logger(__name__)

#: Deliberately conservative. ~60k rows and ~200 features is a small dataset by
#: gradient boosting standards, and EuroLeague rotations shift often enough that
#: an over-fitted model degrades fast. Shallow trees with strong regularisation
#: generalise better here than a deep, finely tuned one.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}

DEFAULT_NUM_ROUNDS = 2000
DEFAULT_EARLY_STOPPING = 100

#: Fraction of the (chronologically ordered) data held out for validation.
VALIDATION_FRACTION = 0.15

#: Fraction held out for the final test report, after validation. Also
#: chronological, so it is the most recent data -- the closest available proxy
#: for "next season".
TEST_FRACTION = 0.15


@dataclass(slots=True)
class TrainingResult:
    """Everything needed to register, judge and reproduce a trained model."""

    target: PredictionTarget
    feature_names: list[str]
    params: dict[str, Any]
    best_iteration: int
    train_rows: int
    validation_rows: int
    test_rows: int
    metrics: Metrics
    #: Additive offset that removes systematic bias. See :func:`calibration_offset`.
    calibration_offset: float
    #: Metrics before calibration, so the trade-off stays visible.
    raw_metrics: Metrics
    baseline_name: str
    baseline_metrics: Metrics
    by_minutes: dict[str, Any]
    feature_importance: list[tuple[str, float]]
    training_data_hash: str
    booster: Any = None
    seasons: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Fractional MAE improvement over the baseline. Positive is better."""
        if not math.isfinite(self.baseline_metrics.mae) or self.baseline_metrics.mae == 0:
            return float("nan")
        return (self.baseline_metrics.mae - self.metrics.mae) / self.baseline_metrics.mae

    @property
    def beats_baseline(self) -> bool:
        """**The Phase 3 gate.**

        Strictly better, on the same held-out rows, measured identically.
        """
        return math.isfinite(self.metrics.mae) and self.metrics.mae < self.baseline_metrics.mae

    def summary(self) -> dict[str, Any]:
        return {
            "target": str(self.target),
            "model_mae": round(self.metrics.mae, 4),
            "baseline": self.baseline_name,
            "baseline_mae": round(self.baseline_metrics.mae, 4),
            "improvement_pct": round(self.improvement * 100, 2),
            "beats_baseline": self.beats_baseline,
            "rmse": round(self.metrics.rmse, 4),
            "r2": round(self.metrics.r2, 4),
            "bias": round(self.metrics.bias, 4),
            "calibration_offset": round(self.calibration_offset, 4),
            "uncalibrated_mae": round(self.raw_metrics.mae, 4),
            "uncalibrated_bias": round(self.raw_metrics.bias, 4),
            "best_iteration": self.best_iteration,
            "rows": {
                "train": self.train_rows,
                "validation": self.validation_rows,
                "test": self.test_rows,
            },
        }


def _hash_rows(rows: list[TrainingRow]) -> str:
    """Stable hash of the exact rows used, so a run can be reproduced.

    Hashes identity and cutoff rather than feature values: it answers "was this
    trained on the same rows?", and is unaffected by harmless float formatting
    changes in the snapshot.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.snapshot_id).encode())
        digest.update(row.prediction_timestamp.isoformat().encode())
    return digest.hexdigest()


def chronological_split(
    rows: list[TrainingRow],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Split into train / validation / test by time, oldest first.

    The test set is the most recent data, which is the closest thing available
    to "the season we have not played yet".
    """
    ordered = sorted(rows, key=lambda row: row.tip_off_utc)
    total = len(ordered)
    test_size = int(total * test_fraction)
    validation_size = int(total * validation_fraction)
    train_size = total - validation_size - test_size

    if train_size <= 0:
        raise ValueError(
            f"not enough rows to split: {total} rows with "
            f"{validation_fraction:.0%} validation and {test_fraction:.0%} test"
        )

    return (
        ordered[:train_size],
        ordered[train_size : train_size + validation_size],
        ordered[train_size + validation_size :],
    )


def calibration_offset(predictions: Any, actuals: Any) -> float:
    """Additive correction that removes systematic bias.

    **Why this is needed.** The L1 objective minimises MAE, and MAE is minimised
    by the *median*. Box score distributions are right-skewed, so the median sits
    below the mean and the raw model under-predicts systematically -- measured at
    -0.577 points against a mean of 7.55, a 7.6% shortfall. That is invisible in
    MAE and very visible in a product that claims to project an expected line.

    The offset is the mean residual on the validation set. Adding it makes the
    predictions unbiased on average at a small MAE cost, which is the right trade
    for a number a human reads as an expectation.

    Computed on validation, never on test: fitting a correction on the data used
    to report accuracy would flatter the result.
    """
    import numpy as np

    return float(np.mean(np.asarray(actuals) - np.asarray(predictions)))


def _best_baseline(
    rows: list[TrainingRow], target: PredictionTarget
) -> tuple[str, Metrics]:
    """Best baseline on exactly the rows the model is judged on.

    Recomputed here rather than reusing the Phase 2 report, because the model is
    scored on a chronological test split and the comparison is only meaningful on
    identical rows.
    """
    best_name, best = "", Metrics(0, math.inf, math.inf, 0.0, 0.0, 0.0)
    for baseline in default_baselines():
        pairs = [
            (prediction, row.target)
            for row in rows
            if (prediction := baseline.predict(row, target)) is not None
        ]
        metrics = compute_metrics(pairs, total_rows=len(rows))
        if math.isfinite(metrics.mae) and metrics.mae < best.mae:
            best_name, best = baseline.name, metrics
    return best_name, best


def train_target(
    rows: list[TrainingRow],
    target: PredictionTarget,
    *,
    params: dict[str, Any] | None = None,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    early_stopping: int = DEFAULT_EARLY_STOPPING,
) -> TrainingResult:
    """Train one LightGBM model and judge it against the best baseline.

    Raises:
        ValueError: if there are too few rows to form a chronological split.
    """
    import lightgbm as lgb

    resolved_params = {**DEFAULT_PARAMS, **(params or {})}
    train_rows, validation_rows, test_rows = chronological_split(rows)

    feature_names = derive_feature_names(rows)
    x_train, y_train = to_matrix(train_rows, feature_names)
    x_validation, y_validation = to_matrix(validation_rows, feature_names)
    x_test, y_test = to_matrix(test_rows, feature_names)

    logger.info(
        "training",
        extra={
            "target": str(target),
            "features": len(feature_names),
            **matrix_summary(x_train, feature_names),
        },
    )

    booster = lgb.train(
        resolved_params,
        lgb.Dataset(x_train, label=y_train, feature_name=list(feature_names)),
        num_boost_round=num_rounds,
        valid_sets=[lgb.Dataset(x_validation, label=y_validation)],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    validation_predictions = booster.predict(
        x_validation, num_iteration=booster.best_iteration
    )
    offset = calibration_offset(validation_predictions, y_validation)

    raw_predictions = booster.predict(x_test, num_iteration=booster.best_iteration)
    raw_metrics = compute_metrics(
        [(float(p), float(a)) for p, a in zip(raw_predictions, y_test, strict=True)],
        total_rows=len(test_rows),
    )

    # Counting stats cannot be negative; clip after correcting.
    predictions = [max(0.0, float(p) + offset) for p in raw_predictions]
    pairs = [(p, float(a)) for p, a in zip(predictions, y_test, strict=True)]
    metrics = compute_metrics(pairs, total_rows=len(test_rows))

    baseline_name, baseline_metrics = _best_baseline(test_rows, target)

    importance = sorted(
        zip(
            feature_names,
            booster.feature_importance(importance_type="gain"),
            strict=True,
        ),
        key=lambda item: -item[1],
    )

    result = TrainingResult(
        target=target,
        feature_names=list(feature_names),
        params=resolved_params,
        best_iteration=int(booster.best_iteration or num_rounds),
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        test_rows=len(test_rows),
        metrics=metrics,
        calibration_offset=offset,
        raw_metrics=raw_metrics,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        by_minutes=stratified_metrics(test_rows, predictions),
        feature_importance=[(name, float(gain)) for name, gain in importance[:30]],
        training_data_hash=_hash_rows(train_rows),
        booster=booster,
        seasons=sorted({row.season_code for row in rows}),
    )

    logger.info("trained", extra=result.summary())

    if not result.beats_baseline:
        # Loud, because the temptation is to ship it anyway. A model worse than
        # arithmetic is a liability: slower, harder to explain, and wrong.
        logger.warning(
            "MODEL FAILED THE GATE - does not beat its baseline",
            extra={
                "target": str(target),
                "model_mae": round(metrics.mae, 4),
                "baseline": baseline_name,
                "baseline_mae": round(baseline_metrics.mae, 4),
            },
        )

    return result


def suspiciously_good(result: TrainingResult, *, threshold: float = 0.25) -> bool:
    """Flag an improvement large enough to suggest leakage rather than skill.

    Published projection systems beat naive baselines by roughly 5-15% MAE. A
    25%+ jump on this dataset is far more likely to be a leak than a
    breakthrough, and the right response is to go looking for it before
    celebrating.
    """
    return math.isfinite(result.improvement) and result.improvement > threshold

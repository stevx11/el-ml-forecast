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
from typing import Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.baselines import default_baselines
from euroleague_ai.ml.dataset import TARGET_FEATURE_STEMS, TrainingRow
from euroleague_ai.ml.distribution import (
    ResidualDistribution,
    choose_volatility_scaling,
    coverage_report,
    fit_residual_distribution,
    volatility_features,
)
from euroleague_ai.ml.evaluation import Metrics, compute_metrics, stratified_metrics
from euroleague_ai.ml.expectation import ExpectationReport, fit_expectation
from euroleague_ai.ml.matrix import derive_feature_names, matrix_summary, to_matrix
from euroleague_ai.models.enums import PredictionTarget

logger = get_logger(__name__)

#: Seed for every stochastic part of training.
#:
#: ``bagging_fraction`` and ``feature_fraction`` sample rows and columns at every
#: iteration. Left unseeded, two runs on byte-identical data produce different
#: models -- which was not a theoretical concern here: model 0.1.0 and model
#: 0.1.1 were trained on the same v2 snapshots against the same baselines, and
#: the three-pointer MAE moved from 0.6823 to 0.7621. A 12% swing from nothing
#: but the random number generator.
#:
#: That makes an unseeded gate meaningless. "Beats the baseline by 2.4%" is not a
#: result if re-running the identical command can turn it into -4.6%, and every
#: comparison between feature set versions is measuring noise. Pin it.
TRAINING_SEED = 20_251_101

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
    # LightGBM derives the individual seeds from ``seed``, but setting them
    # explicitly means a future change to one cannot silently reshuffle another.
    "seed": TRAINING_SEED,
    "bagging_seed": TRAINING_SEED + 1,
    "feature_fraction_seed": TRAINING_SEED + 2,
    "data_random_seed": TRAINING_SEED + 3,
    # Seeds alone are not enough: histogram construction is order-sensitive, so
    # floating-point summation varies with thread scheduling. These two make the
    # result reproducible on a fixed machine. ``deterministic`` requires one of
    # the force_*_wise options to be set, or LightGBM warns and ignores it.
    "deterministic": True,
    "force_row_wise": True,
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
    #: Additive offset, the MAE-minimising median residual. See
    #: :func:`calibration_offset`.
    calibration_offset: float
    #: Mean residual on validation. Diagnostic only -- never corrected away,
    #: because doing so costs MAE in proportion to the target's skew.
    validation_mean_residual: float
    #: Held-out error distribution, fitted on validation. Turns the point
    #: estimate into quantiles and threshold probabilities at serving time.
    distribution: ResidualDistribution
    #: Interval coverage on the test split. The gate for the distribution, in
    #: the way MAE is the gate for the point estimate. Also per volatility third,
    #: the threshold Brier score, and how the spread scaling was chosen.
    coverage: dict[str, Any]
    #: Metrics before calibration, so the trade-off stays visible.
    raw_metrics: Metrics
    baseline_name: str
    baseline_metrics: Metrics
    by_minutes: dict[str, Any]
    feature_importance: list[tuple[str, float]]
    training_data_hash: str
    booster: Any = None
    seasons: list[str] = field(default_factory=list)
    #: How the expected value is produced, and how well. ``None`` when training
    #: was asked to skip it (the tuner does, to halve its cost).
    expectation: ExpectationReport | None = None
    #: The expectation booster, when it won on validation and passed its gate.
    expected_booster: Any = None

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
            "validation_mean_residual": round(self.validation_mean_residual, 4),
            "coverage": self.coverage,
            "distribution_rows": self.distribution.occupancy(),
            "uncalibrated_mae": round(self.raw_metrics.mae, 4),
            "uncalibrated_bias": round(self.raw_metrics.bias, 4),
            "best_iteration": self.best_iteration,
            "rows": {
                "train": self.train_rows,
                "validation": self.validation_rows,
                "test": self.test_rows,
            },
            "expectation": self.expectation.summary() if self.expectation else None,
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


def _fit_spread(
    target: PredictionTarget,
    medians: list[float],
    actuals: list[float],
    histories: list[float | None],
    rows: list[TrainingRow],
) -> tuple[ResidualDistribution, ResidualDistribution | None, dict[str, Any]]:
    """The served error distribution, the unscaled one to compare, and the choice.

    The volatility scale (source and power) is chosen on validation alone; see
    :func:`~euroleague_ai.ml.distribution.choose_volatility_scaling`. When no
    scale beats the unscaled spread, both are the same and the second is None.
    """
    stem = TARGET_FEATURE_STEMS.get(target)
    candidates = (
        {name: [row.features.get(name) for row in rows] for name in volatility_features(stem)}
        if stem
        else {}
    )
    choice = choose_volatility_scaling(medians, actuals, histories, candidates)
    unscaled = fit_residual_distribution(medians, actuals, histories)
    if choice.volatility_feature is None:
        return unscaled, None, choice.as_dict()
    scaled = fit_residual_distribution(
        medians,
        actuals,
        histories,
        candidates[choice.volatility_feature],
        volatility_feature=choice.volatility_feature,
        scale_power=choice.scale_power,
    )
    logger.info(
        "per-player spread chosen",
        extra={"target": str(target), **choice.as_dict()},
    )
    return scaled, unscaled, choice.as_dict()


def _history_depths(rows: list[TrainingRow]) -> list[float | None]:
    """Prior-game counts, used to condition the error distribution.

    Read from the feature vector rather than recomputed, so it is exactly the
    quantity the model saw -- and so it stays subject to the same ``as_of``
    filtering as everything else.
    """
    return [row.features.get("games_played_career_window") for row in rows]


def calibration_offset(predictions: Any, actuals: Any) -> float:
    """Additive correction, chosen to minimise MAE rather than to zero the bias.

    **This used to be the mean residual, and that was a serious mistake.**

    The reasoning was: an L1 objective predicts the conditional *median*, box
    score distributions are right-skewed, so the median sits below the mean and
    the model under-predicts what a reader interprets as an expectation. Shifting
    by the mean residual fixes that. The docstring called the MAE cost "small".

    It was not small, and the numbers were in the training logs the whole time::

        target      uncalibrated   calibrated   baseline   verdict
        MINUTES         4.4381       4.4361      4.5452    unchanged
        POINTS          3.9900       4.0463      4.1123    +2.97% -> +1.61%
        REBOUNDS        1.5199       1.5402      1.5756    +3.53% -> +2.25%
        ASSISTS         1.0764       1.1263      1.1329    +4.99% -> +0.58%
        THREE_POINTERS  0.6813       0.7631      0.7335    +7.12% -> -4.04%

    The correction cost the three-point model eleven percentage points and turned
    a comfortable pass into a failure. Minutes was untouched because minutes are
    roughly symmetric; the damage scaled with skew, which is exactly what theory
    predicts. And the whole arrangement was incoherent: training deliberately
    degraded MAE and the release gate then rejected the model for poor MAE.

    So the offset is now the *median* residual, which is by construction the
    additive constant that minimises MAE. For a well-fitted L1 model it is close
    to zero -- the model is already median-unbiased -- so this is nearly a no-op,
    which is the correct behaviour.

    The underlying observation stands: a median is not an expectation, and a user
    reading "5.2 points" wants one. The answer to that is a predicted
    distribution, not a constant nudge that damages every metric we report.

    Computed on validation, never on test: fitting a correction on the data used
    to report accuracy would flatter the result.
    """
    import numpy as np

    return float(np.median(np.asarray(actuals) - np.asarray(predictions)))


def mean_residual(predictions: Any, actuals: Any) -> float:
    """Mean residual, kept purely as a diagnostic.

    This is what the old calibration corrected away. Reporting it keeps the
    median-versus-mean gap visible without letting it distort the model.
    """
    import numpy as np

    return float(np.mean(np.asarray(actuals) - np.asarray(predictions)))


def _best_baseline(rows: list[TrainingRow], target: PredictionTarget) -> tuple[str, Metrics]:
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
    fit_expected: bool = True,
    split: tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]] | None = None,
) -> TrainingResult:
    """Train one LightGBM model and judge it against the best baseline.

    With ``fit_expected`` (the default) a second, mean-targeting head is fitted
    and gated too. See :mod:`euroleague_ai.ml.expectation`.

    ``split`` replaces the default 70/15/15 chronological split with explicit
    ``(train, validation, test)`` rows; walk-forward evaluation uses it to test
    on one whole season at a time. Everything else is identical, so a fold is
    judged exactly as a production training run is.

    Raises:
        ValueError: if there are too few rows to form a chronological split.
    """
    import lightgbm as lgb

    resolved_params = {**DEFAULT_PARAMS, **(params or {})}
    train_rows, validation_rows, test_rows = split or chronological_split(rows)

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

    validation_predictions = booster.predict(x_validation, num_iteration=booster.best_iteration)
    offset = calibration_offset(validation_predictions, y_validation)
    validation_mean_residual = mean_residual(validation_predictions, y_validation)

    # The median residual minimises MAE over additive constants, but predictions
    # are clipped at zero afterwards, which breaks that guarantee at the low end.
    # Check on validation and drop the offset if it does not actually help.
    def _validation_mae(shift: float) -> float:
        return sum(
            abs(max(0.0, float(p) + shift) - float(a))
            for p, a in zip(validation_predictions, y_validation, strict=True)
        ) / max(1, len(y_validation))

    if _validation_mae(offset) > _validation_mae(0.0):
        logger.info(
            "calibration offset rejected; it did not improve validation MAE",
            extra={"target": str(target), "rejected_offset": round(offset, 4)},
        )
        offset = 0.0

    raw_predictions = booster.predict(x_test, num_iteration=booster.best_iteration)
    raw_metrics = compute_metrics(
        [(float(p), float(a)) for p, a in zip(raw_predictions, y_test, strict=True)],
        total_rows=len(test_rows),
    )

    # Counting stats cannot be negative; clip after correcting.
    predictions = [max(0.0, float(p) + offset) for p in raw_predictions]
    pairs = [(p, float(a)) for p, a in zip(predictions, y_test, strict=True)]
    metrics = compute_metrics(pairs, total_rows=len(test_rows))

    # The error distribution is fitted on the numbers we actually serve -- after
    # the offset and the clip -- because those are the predictions whose error we
    # need to characterise. Fitting on the raw booster output would describe a
    # model nobody sees.
    calibrated_validation = [max(0.0, float(p) + offset) for p in validation_predictions]
    validation_history = _history_depths(validation_rows)
    validation_actuals = [float(a) for a in y_validation]
    distribution, unscaled, scaling = _fit_spread(
        target, calibrated_validation, validation_actuals, validation_history, validation_rows
    )
    stem = TARGET_FEATURE_STEMS.get(target)
    coverage = coverage_report(
        distribution,
        predictions,
        [float(a) for a in y_test],
        _history_depths(test_rows),
        volatilities=[distribution.volatility_of(row.features) for row in test_rows],
        group_volatilities=(
            [row.features.get(f"std10_{stem}") for row in test_rows] if stem else None
        ),
        target=str(target),
        reference=unscaled,
    )
    coverage["spread_scaling"] = scaling

    baseline_name, baseline_metrics = _best_baseline(test_rows, target)

    expectation = None
    expected_booster = None
    if fit_expected:
        fitted = fit_expectation(
            target=target,
            base_params=resolved_params,
            feature_names=feature_names,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            x_test=x_test,
            validation_rows=validation_rows,
            test_rows=test_rows,
            validation_medians=calibrated_validation,
            test_medians=predictions,
            distribution=distribution,
            num_rounds=num_rounds,
            early_stopping=early_stopping,
        )
        expectation, expected_booster = fitted.report, fitted.booster

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
        validation_mean_residual=validation_mean_residual,
        distribution=distribution,
        coverage=coverage,
        raw_metrics=raw_metrics,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        by_minutes=stratified_metrics(test_rows, predictions),
        feature_importance=[(name, float(gain)) for name, gain in importance[:30]],
        training_data_hash=_hash_rows(train_rows),
        booster=booster,
        seasons=sorted({row.season_code for row in rows}),
        expectation=expectation,
        expected_booster=expected_booster,
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

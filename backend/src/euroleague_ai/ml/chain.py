"""Minutes first, then per-minute rates.

**The idea.** Points are minutes multiplied by scoring rate. Asking one model to
predict points makes it learn both at once -- how long the coach will use him,
and how productive he is while out there -- from a single number. Those are
different questions with different answers, and one of them is far easier: a
rotation is a decision made by a person with a pattern, while shooting is close
to a coin.

So: predict minutes, predict points per forty minutes, multiply.

**Why the rate is the more stable quantity.** A player's points swing with
minutes even when nothing about him changed -- foul trouble, a blowout, a coach
shortening the rotation. Dividing that out leaves a number that moves only when
his actual production moves, which is a cleaner thing to learn.

**The trap this design has to avoid.** A player with two minutes and one basket
has a rate of forty per forty. Left alone those rows are pure noise and would
dominate the fit, because there are a lot of them. Every rate model here is
therefore fitted with per-row weights equal to minutes played -- a row is worth
exactly as much evidence as the playing time it represents. It is the standard
treatment for a ratio estimated from unequal samples, and without it the chain
is worse than what it replaces.

**The honest catch: errors multiply.** If minutes come in 10% high and the rate
comes in 10% high, points land 21% high, where a direct model would have made
one error rather than compounding two. Whether the stability gained outweighs
the compounding is an empirical question, not a design opinion -- so nothing
here ships on the strength of the argument above. :func:`compare_chain` trains
both approaches on identical rows and reports which actually won.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.evaluation import Metrics, compute_metrics
from euroleague_ai.ml.matrix import derive_feature_names, to_matrix
from euroleague_ai.ml.train import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_PARAMS,
    _best_baseline,
    calibration_offset,
    chronological_split,
)
from euroleague_ai.models.enums import PredictionTarget

logger = get_logger(__name__)

#: Rates are expressed per forty minutes -- a full game -- because that is the
#: unit basketball already thinks in, which makes the intermediate number
#: readable rather than an internal artefact.
RATE_SCALE = 40.0

#: Rows below this many minutes are dropped from *rate* training entirely.
#:
#: Weighting already discounts them, but a one-minute appearance carries a rate
#: with essentially infinite variance, and a handful of those still distort the
#: tree splits at the extremes. Five minutes is the point where a rate starts
#: describing a player rather than a rounding artefact. They stay in the
#: minutes model, where a short appearance is a perfectly real outcome.
MIN_RATE_MINUTES = 5.0

#: Targets worth chaining. Minutes is the multiplier, so it is never itself
#: chained, and a rate is only meaningful for a counting stat accumulated while
#: on court.
CHAINABLE: tuple[PredictionTarget, ...] = (
    PredictionTarget.POINTS,
    PredictionTarget.REBOUNDS,
    PredictionTarget.ASSISTS,
    PredictionTarget.THREE_POINTERS_MADE,
)


@dataclass(slots=True)
class ChainComparison:
    """Chain versus direct, measured on identical test rows."""

    target: PredictionTarget
    rows: int
    direct: Metrics
    chained: Metrics
    baseline_name: str
    baseline: Metrics
    minutes_mae: float
    rate_mae: float
    #: Chained MAE minus direct MAE, as a fraction. Negative means the chain won.
    delta_pct: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def chain_wins(self) -> bool:
        return self.chained.mae < self.direct.mae

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": str(self.target),
            "rows": self.rows,
            "direct_mae": round(self.direct.mae, 4),
            "chained_mae": round(self.chained.mae, 4),
            "baseline": self.baseline_name,
            "baseline_mae": round(self.baseline.mae, 4),
            "minutes_mae": round(self.minutes_mae, 4),
            "rate_mae": round(self.rate_mae, 4),
            "delta_pct": round(self.delta_pct, 2),
            "chain_wins": self.chain_wins,
            "notes": self.notes,
        }


def rate_label(row: TrainingRow) -> float | None:
    """The row's outcome expressed per forty minutes, or ``None`` if unusable."""
    if row.actual_minutes is None or row.actual_minutes < MIN_RATE_MINUTES:
        return None
    return float(row.target) / float(row.actual_minutes) * RATE_SCALE


def _fit(
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    feature_names: list[str],
    *,
    weights: Any = None,
    params: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """Train one booster. Returns ``(booster, best_iteration)``.

    Deliberately thin: the chain needs several models fitted the same way, and
    the full :func:`train_target` machinery -- baselines, calibration,
    distributions -- belongs to a model that is going to be *served*, not to an
    intermediate one.
    """
    import lightgbm as lgb

    resolved = {**DEFAULT_PARAMS, **(params or {})}
    booster = lgb.train(
        resolved,
        lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=feature_names),
        num_boost_round=DEFAULT_NUM_ROUNDS,
        valid_sets=[lgb.Dataset(x_validation, label=y_validation)],
        callbacks=[
            lgb.early_stopping(DEFAULT_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster, booster.best_iteration


def compare_chain(
    rows: list[TrainingRow],
    target: PredictionTarget,
    *,
    params: dict[str, Any] | None = None,
) -> ChainComparison:
    """Train chained and direct models on the same split and report the winner.

    One row set produces both labels: ``row.target`` is the counting stat and
    ``row.actual_minutes`` the playing time, so the two approaches are compared
    on genuinely identical data rather than on two loads that might disagree
    about which rows exist.

    Returns:
        The comparison. Nothing is registered or activated here -- deciding to
        ship is a separate step, taken after reading the numbers.
    """
    import numpy as np

    if target not in CHAINABLE:
        raise ValueError(f"{target} is not a chainable target")

    train_rows, validation_rows, test_rows = chronological_split(rows)
    feature_names = derive_feature_names(rows)
    notes: list[str] = []

    x_train, y_direct_train = to_matrix(train_rows, feature_names)
    x_validation, y_direct_validation = to_matrix(validation_rows, feature_names)
    x_test, y_direct_test = to_matrix(test_rows, feature_names)

    # -- direct: one model straight to the counting stat, as today -----------
    direct_booster, direct_best = _fit(
        x_train,
        y_direct_train,
        x_validation,
        y_direct_validation,
        feature_names,
        params=params,
    )
    direct_validation = direct_booster.predict(x_validation, num_iteration=direct_best)
    direct_offset = calibration_offset(direct_validation, y_direct_validation)
    direct_raw = direct_booster.predict(x_test, num_iteration=direct_best)
    direct_predictions = [max(0.0, float(v) + direct_offset) for v in direct_raw]
    direct_metrics = compute_metrics(
        list(zip(direct_predictions, [float(v) for v in y_direct_test], strict=True)),
        total_rows=len(test_rows),
    )

    # -- minutes: the multiplier ---------------------------------------------
    y_minutes_train = np.array([r.actual_minutes for r in train_rows], dtype=float)
    y_minutes_validation = np.array([r.actual_minutes for r in validation_rows], dtype=float)
    y_minutes_test = np.array([r.actual_minutes for r in test_rows], dtype=float)

    minutes_booster, minutes_best = _fit(
        x_train,
        y_minutes_train,
        x_validation,
        y_minutes_validation,
        feature_names,
        params=params,
    )
    minutes_validation = minutes_booster.predict(x_validation, num_iteration=minutes_best)
    minutes_offset = calibration_offset(minutes_validation, y_minutes_validation)
    minutes_test_pred = np.clip(
        minutes_booster.predict(x_test, num_iteration=minutes_best) + minutes_offset, 0.0, None
    )
    minutes_metrics = compute_metrics(
        list(
            zip(
                [float(v) for v in minutes_test_pred],
                [float(v) for v in y_minutes_test],
                strict=True,
            )
        ),
        total_rows=len(test_rows),
    )

    # -- rate: per forty minutes, weighted by minutes played -----------------
    #
    # Weighting is what makes this work. A row is worth exactly as much evidence
    # as the playing time behind it, so a 34-minute line counts seven times a
    # 5-minute cameo instead of equally.
    def rate_arrays(subset: list[TrainingRow]) -> tuple[Any, Any, Any]:
        keep, labels, weights = [], [], []
        for index, row in enumerate(subset):
            label = rate_label(row)
            if label is None:
                continue
            keep.append(index)
            labels.append(label)
            weights.append(float(row.actual_minutes))
        return (
            np.array(keep, dtype=int),
            np.array(labels, dtype=float),
            np.array(weights, dtype=float),
        )

    train_index, y_rate_train, w_rate_train = rate_arrays(train_rows)
    validation_index, y_rate_validation, _ = rate_arrays(validation_rows)
    dropped = len(train_rows) - len(train_index)
    if dropped:
        notes.append(
            f"{dropped:,} of {len(train_rows):,} training rows below "
            f"{MIN_RATE_MINUTES:.0f} minutes excluded from the rate model"
        )
    if len(train_index) < 500:
        raise ValueError(f"only {len(train_index)} usable rate rows for {target}; too few to fit")

    rate_booster, rate_best = _fit(
        x_train[train_index],
        y_rate_train,
        x_validation[validation_index],
        y_rate_validation,
        feature_names,
        weights=w_rate_train,
        params=params,
    )
    rate_test_pred = np.clip(rate_booster.predict(x_test, num_iteration=rate_best), 0.0, None)

    # Rate accuracy on its own terms, on the test rows where a rate is defined.
    test_index, y_rate_test, _ = rate_arrays(test_rows)
    rate_metrics = compute_metrics(
        list(
            zip(
                [float(v) for v in rate_test_pred[test_index]],
                [float(v) for v in y_rate_test],
                strict=True,
            )
        ),
        total_rows=max(1, len(test_index)),
    )

    # -- recombine and calibrate the product ---------------------------------
    #
    # The product needs its own offset: two calibrated factors do not multiply
    # into a calibrated product, because E[XY] != E[X]E[Y] whenever the two are
    # correlated -- and minutes and scoring rate certainly are.
    minutes_validation_pred = np.clip(minutes_validation + minutes_offset, 0.0, None)
    rate_validation_pred = np.clip(
        rate_booster.predict(x_validation, num_iteration=rate_best), 0.0, None
    )
    chained_validation = minutes_validation_pred * rate_validation_pred / RATE_SCALE
    chained_offset = calibration_offset(chained_validation, y_direct_validation)

    chained_predictions = np.clip(
        minutes_test_pred * rate_test_pred / RATE_SCALE + chained_offset, 0.0, None
    )
    chained_metrics = compute_metrics(
        list(
            zip(
                [float(v) for v in chained_predictions],
                [float(v) for v in y_direct_test],
                strict=True,
            )
        ),
        total_rows=len(test_rows),
    )

    baseline_name, baseline_metrics = _best_baseline(test_rows, target)
    delta = (
        (chained_metrics.mae - direct_metrics.mae) / direct_metrics.mae * 100.0
        if direct_metrics.mae
        else 0.0
    )

    comparison = ChainComparison(
        target=target,
        rows=len(test_rows),
        direct=direct_metrics,
        chained=chained_metrics,
        baseline_name=baseline_name,
        baseline=baseline_metrics,
        minutes_mae=minutes_metrics.mae,
        rate_mae=rate_metrics.mae,
        delta_pct=delta,
        notes=notes,
    )
    logger.info("chain comparison", extra=comparison.as_dict())
    return comparison

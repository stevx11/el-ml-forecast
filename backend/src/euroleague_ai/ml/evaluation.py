"""Evaluation metrics and the walk-forward backtest.

Pure Python, no numpy. Phase 2 has to be runnable in the standard backend image,
and these are a dozen lines of arithmetic each -- pulling in the ML extra just to
compute a mean would put the bar between "look at the baselines" and "install
heavy wheels", which is exactly the wrong friction to add to the phase whose
whole purpose is establishing an honest starting point.

**Why MAE is the headline.** Box score distributions are right-skewed, so RMSE
is dominated by the handful of 40-point nights. Optimising or judging by RMSE
sacrifices the 90% of predictions people actually use in order to chase
outliers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TrainingRow

logger = get_logger(__name__)

#: Minutes buckets for stratified reporting. A model can look strong overall
#: while being useless for the 10-25 minute rotation players, which is precisely
#: where the interesting decisions are. Aggregate metrics hide that completely.
MINUTES_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-10", 0.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30+", 30.0, math.inf),
)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Accuracy of one predictor on one set of rows."""

    n: int
    mae: float
    rmse: float
    r2: float
    bias: float
    coverage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "bias": round(self.bias, 4),
            "coverage": round(self.coverage, 4),
        }


def compute_metrics(
    pairs: Sequence[tuple[float, float]], *, total_rows: int | None = None
) -> Metrics:
    """Metrics over ``(predicted, actual)`` pairs.

    Args:
        pairs: Predictions and their realised values.
        total_rows: Rows the predictor was *offered*, used for coverage. Defaults
            to ``len(pairs)``, i.e. 100% coverage.
    """
    n = len(pairs)
    offered = total_rows if total_rows is not None else n
    if n == 0:
        return Metrics(0, math.nan, math.nan, math.nan, math.nan, 0.0)

    errors = [predicted - actual for predicted, actual in pairs]
    actuals = [actual for _, actual in pairs]

    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    # Signed, so systematic over- or under-prediction is visible. A model with
    # good MAE and large bias is miscalibrated in a way MAE alone hides.
    bias = sum(errors) / n

    mean_actual = sum(actuals) / n
    ss_total = sum((a - mean_actual) ** 2 for a in actuals)
    ss_residual = sum(e * e for e in errors)
    # R2 is undefined when every actual is identical; report 0 rather than
    # dividing by zero or emitting NaN into a comparison table.
    r2 = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    return Metrics(
        n=n,
        mae=mae,
        rmse=rmse,
        r2=r2,
        bias=bias,
        coverage=n / offered if offered else 0.0,
    )


def stratified_metrics(
    rows: Sequence[TrainingRow],
    predictions: Sequence[float | None],
) -> dict[str, dict[str, Any]]:
    """Metrics split by the player's realised minutes.

    Uses realised rather than projected minutes deliberately: the question being
    answered is "how good is this on rotation players?", and realised minutes is
    the honest way to define that group after the fact. It is an evaluation
    input, never a feature.
    """
    buckets: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in MINUTES_BUCKETS}
    offered: dict[str, int] = {name: 0 for name, _, _ in MINUTES_BUCKETS}

    for row, prediction in zip(rows, predictions, strict=True):
        for name, low, high in MINUTES_BUCKETS:
            if low <= row.actual_minutes < high:
                offered[name] += 1
                if prediction is not None:
                    buckets[name].append((prediction, row.target))
                break

    return {
        name: compute_metrics(pairs, total_rows=offered[name]).as_dict()
        for name, pairs in buckets.items()
    }


@dataclass(slots=True)
class Fold:
    """One walk-forward split."""

    season_code: str
    round_number: int
    train: list[TrainingRow] = field(default_factory=list)
    test: list[TrainingRow] = field(default_factory=list)


def walk_forward_folds(rows: Sequence[TrainingRow], *, min_train_rounds: int = 5) -> list[Fold]:
    """Split chronologically: train on rounds ``1..k``, test on ``k+1``.

    **Never a random split.** Randomly partitioning player-games puts the same
    player's adjacent games on both sides, so the model interpolates between
    known values rather than forecasting an unknown one. Accuracy measured that
    way is roughly meaningless for this task and always flatteringly high.

    Folds are built within each season: form windows reset across the summer, and
    training on next season to predict this one is not a mistake anyone should be
    able to make by accident.

    Args:
        min_train_rounds: Rounds that must precede a fold before it is evaluated.
            Early rounds have near-empty form windows, so including them measures
            missingness handling rather than basketball.
    """
    by_season: dict[str, list[TrainingRow]] = {}
    for row in rows:
        by_season.setdefault(row.season_code, []).append(row)

    folds: list[Fold] = []
    for season_code, season_rows in sorted(by_season.items()):
        by_round: dict[int, list[TrainingRow]] = {}
        for row in season_rows:
            if row.round_number is None:
                continue
            by_round.setdefault(row.round_number, []).append(row)

        rounds = sorted(by_round)
        for index, round_number in enumerate(rounds):
            if index < min_train_rounds:
                continue
            train = [row for earlier in rounds[:index] for row in by_round[earlier]]
            folds.append(
                Fold(
                    season_code=season_code,
                    round_number=round_number,
                    train=train,
                    test=by_round[round_number],
                )
            )
    return folds


def backtest(
    rows: Sequence[TrainingRow],
    predictor: Callable[[TrainingRow], float | None],
    *,
    min_train_rounds: int = 5,
) -> dict[str, Any]:
    """Walk-forward evaluation of a predictor.

    ``predictor`` is a plain callable, so baselines and trained models are
    measured by identical machinery. Baselines ignore the training half of each
    fold; a model would refit on it.
    """
    folds = walk_forward_folds(rows, min_train_rounds=min_train_rounds)
    if not folds:
        return {
            "folds": 0,
            "overall": Metrics(0, math.nan, math.nan, math.nan, math.nan, 0.0).as_dict(),
        }

    test_rows: list[TrainingRow] = []
    predictions: list[float | None] = []
    for fold in folds:
        for row in fold.test:
            test_rows.append(row)
            predictions.append(predictor(row))

    pairs = [
        (prediction, row.target)
        for row, prediction in zip(test_rows, predictions, strict=True)
        if prediction is not None
    ]

    return {
        "folds": len(folds),
        "test_rows": len(test_rows),
        "overall": compute_metrics(pairs, total_rows=len(test_rows)).as_dict(),
        "by_minutes": stratified_metrics(test_rows, predictions),
        "seasons": sorted({fold.season_code for fold in folds}),
    }


def compare(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank predictors by MAE, best first.

    Returns a flat table ready to print. NaN MAE sorts last so a predictor that
    covered nothing cannot top the table.
    """
    table = [
        {
            "predictor": name,
            "mae": result["overall"]["mae"],
            "rmse": result["overall"]["rmse"],
            "r2": result["overall"]["r2"],
            "bias": result["overall"]["bias"],
            "coverage": result["overall"]["coverage"],
            "n": result["overall"]["n"],
        }
        for name, result in results.items()
    ]
    return sorted(table, key=lambda entry: (math.isnan(entry["mae"]), entry["mae"]))

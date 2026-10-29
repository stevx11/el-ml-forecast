"""Expected values: what a player averages, not what he usually does.

**Why the median is not enough.** The stat models are trained with an L1
objective, so they predict the conditional *median*. That is the right choice
for MAE and the release gate, but box scores are right-skewed and the median sits
below the mean. The held-out residuals of model 0.1.8 show how far below:

    target            mean(actual - predicted)   x ~12 players per side
    POINTS                    +0.69                   ~8 points per team
    REBOUNDS                  +0.31                   ~4
    ASSISTS                   +0.28                   ~3
    THREE_POINTERS_MADE       +0.26                   ~3, a third of a team's makes

So a squad's projected lines add up to a team that scores eight points less
than real teams do, and a 38% shooter is shown making "1" three every night.
Nobody reads "5.2 points" as a median. They read it as an expectation.

**Two ways to get an expectation, and both are tried.**

1. **An expectation booster.** A second LightGBM model on the same features,
   trained for the mean: Tweedie loss for the counting stats (non-negative,
   right-skewed, lots of zeros for threes) and squared error for minutes
   (roughly symmetric).
2. **The shifted median.** The median plus the *mean* residual of its
   held-out error cell. Costs nothing, because the cells already exist for the
   intervals.

The winner is chosen on **validation** RMSE and reported on **test**, the same
"select on validation, report on test" rule the tuner follows. The shifted
median's cell means are themselves fitted on validation, which gives it a small
in-sample edge in that comparison. That is accepted knowingly: twenty cell means
cannot overfit much, and the edge only ever favours the simpler option.

**The gate.** The expected value is a second reading of a model that already
passed its own gate, and it is published *instead of the median* as the
headline number. So it is judged against the median it replaces, on test, on
the two things an expectation is for:

* **lower RMSE.** Squared error is the loss whose minimiser is the mean. Judging
  an expectation by MAE would reward the median it is meant to replace.
* **smaller squad-total bias.** The reason this module exists.

If either fails, nothing is published for that target. The best *baseline*'s
RMSE is reported alongside as a warning rather than a gate. The baselines are
season means, which are themselves estimates of the mean. On data where a
season mean is nearly the truth, a booster can tie it on RMSE while still
fixing a ten-point team-total bias, and refusing that fix would be the wrong
call.

**The headline realism check** is team totals: sum each squad's predictions per
game and compare with what the squad actually produced. That is the number that
was eight points short.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.baselines import default_baselines
from euroleague_ai.ml.evaluation import Metrics, compute_metrics
from euroleague_ai.models.enums import PredictionTarget

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from euroleague_ai.ml.dataset import TrainingRow
    from euroleague_ai.ml.distribution import ResidualDistribution, ScaleInput

logger = get_logger(__name__)

#: Where each target's expectation comes from. ``NONE`` means it failed the gate
#: and nothing is published.
SOURCE_BOOSTER = "booster"
SOURCE_SHIFTED_MEDIAN = "shifted_median"
SOURCE_NONE = "none"

#: Tweedie power between Poisson (1.0) and gamma (2.0). 1.3 suits box-score
#: counts: overdispersed relative to Poisson, with a point mass at zero that a
#: gamma cannot represent.
TWEEDIE_VARIANCE_POWER = 1.3


def expectation_params(target: PredictionTarget) -> dict[str, Any]:
    """Objective overrides for a target's expectation booster.

    Early stopping uses ``l2`` everywhere, so stopping is driven by the same
    quantity the gate measures.
    """
    if target is PredictionTarget.MINUTES:
        return {"objective": "regression", "metric": "l2"}
    return {
        "objective": "tweedie",
        "tweedie_variance_power": TWEEDIE_VARIANCE_POWER,
        "metric": "l2",
    }


def team_key(row: TrainingRow) -> tuple[Any, Any]:
    """Which side of which game a row belongs to.

    ``is_home`` is in every snapshot and splits a game into its two squads
    without needing the team id, which training rows do not carry.
    """
    return (row.game_id, row.features.get("is_home"))


def team_total_error(predictions: Sequence[float], rows: Sequence[TrainingRow]) -> dict[str, float]:
    """Error in summed squad totals, per team-game.

    ``bias`` is predicted minus actual, matching
    :func:`~euroleague_ai.ml.evaluation.compute_metrics`: negative means the
    projected squad produces less than the real one did.

    Training rows contain only players who played, so this measures whether the
    lines *of the players who played* add up. Absences are the play-probability
    model's job and are folded in at serving time.
    """
    predicted: dict[tuple[Any, Any], float] = defaultdict(float)
    actual: dict[tuple[Any, Any], float] = defaultdict(float)
    for value, row in zip(predictions, rows, strict=True):
        key = team_key(row)
        predicted[key] += float(value)
        actual[key] += float(row.target)
    if not predicted:
        return {"team_games": 0, "mae": float("nan"), "bias": float("nan")}
    errors = [predicted[key] - actual[key] for key in predicted]
    return {
        "team_games": len(errors),
        "mae": round(sum(abs(e) for e in errors) / len(errors), 4),
        "bias": round(sum(errors) / len(errors), 4),
    }


def shifted_median(
    medians: Sequence[float],
    rows: Sequence[TrainingRow] | None,
    distribution: ResidualDistribution,
    *,
    histories: Sequence[float | None] | None = None,
    volatilities: Sequence[ScaleInput] | None = None,
) -> list[float]:
    """Median plus its error cell's mean residual, clipped at zero.

    The mean residual is scaled by the player's own spread when the
    distribution has one, exactly as his quantiles are.
    """
    if histories is None:
        if rows is None:
            raise ValueError("need rows or histories")
        histories = [row.features.get("games_played_career_window") for row in rows]
    if volatilities is None:
        volatilities = (
            [distribution.scale_input(row.features) for row in rows]
            if rows is not None
            else [None] * len(medians)
        )
    return [
        max(0.0, float(m) + distribution.mean_residual(float(m), h, v))
        for m, h, v in zip(medians, histories, volatilities, strict=True)
    ]


def best_baseline_rmse(
    rows: Sequence[TrainingRow], target: PredictionTarget
) -> tuple[str, Metrics]:
    """The baseline with the lowest RMSE on exactly these rows.

    Separate from the MAE baseline choice in ``train``: the baseline that is
    best at the median is not necessarily best at the mean.
    """
    best_name, best = "", Metrics(0, math.inf, math.inf, 0.0, 0.0, 0.0)
    for baseline in default_baselines():
        pairs = [
            (prediction, row.target)
            for row in rows
            if (prediction := baseline.predict(row, target)) is not None
        ]
        metrics = compute_metrics(pairs, total_rows=len(rows))
        if math.isfinite(metrics.rmse) and metrics.rmse < best.rmse:
            best_name, best = baseline.name, metrics
    return best_name, best


@dataclass(slots=True)
class ExpectationReport:
    """How the expected value was chosen, and how good it is."""

    source: str
    #: Validation RMSE of each candidate. The selection was made on these.
    validation_rmse: dict[str, float]
    #: Test metrics of each candidate and of the median it replaces.
    test: dict[str, Metrics]
    #: Squad-total error on test, per candidate and for the median.
    team_totals: dict[str, dict[str, float]]
    baseline_name: str
    baseline_metrics: Metrics
    booster_best_iteration: int | None = None
    objective: dict[str, Any] = field(default_factory=dict)

    @property
    def chosen(self) -> str:
        """The candidate that won on validation, whether or not it passed."""
        return min(self.validation_rmse, key=lambda k: self.validation_rmse[k])

    @property
    def passes_gate(self) -> bool:
        """Beats the served median on test RMSE *and* on squad-total bias."""
        chosen, median = self.test.get(self.chosen), self.test.get("median")
        if chosen is None or median is None or not math.isfinite(chosen.rmse):
            return False
        chosen_bias = abs(self.team_totals.get(self.chosen, {}).get("bias", math.inf))
        median_bias = abs(self.team_totals.get("median", {}).get("bias", math.inf))
        return chosen.rmse < median.rmse and chosen_bias < median_bias

    @property
    def beats_baseline(self) -> bool:
        """Diagnostic only: lower test RMSE than the best baseline."""
        metrics = self.test.get(self.chosen)
        return (
            metrics is not None
            and math.isfinite(metrics.rmse)
            and metrics.rmse < self.baseline_metrics.rmse
        )

    def summary(self) -> dict[str, Any]:
        def m(metrics: Metrics) -> dict[str, float]:
            return {
                "rmse": round(metrics.rmse, 4),
                "mae": round(metrics.mae, 4),
                "bias": round(metrics.bias, 4),
            }

        return {
            "source": self.source,
            "chosen": self.chosen,
            "passes_gate": self.passes_gate,
            "beats_baseline": self.beats_baseline,
            "validation_rmse": {k: round(v, 4) for k, v in self.validation_rmse.items()},
            "test": {name: m(metrics) for name, metrics in self.test.items()},
            "team_totals": self.team_totals,
            "baseline": self.baseline_name,
            "baseline_rmse": round(self.baseline_metrics.rmse, 4),
            "booster_best_iteration": self.booster_best_iteration,
            "objective": self.objective,
        }


@dataclass(slots=True)
class ExpectationFit:
    """The fitted head: the report plus the booster, if it won."""

    report: ExpectationReport
    booster: Any = None


def fit_expectation(
    *,
    target: PredictionTarget,
    base_params: dict[str, Any],
    feature_names: Sequence[str],
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    x_test: Any,
    validation_rows: Sequence[TrainingRow],
    test_rows: Sequence[TrainingRow],
    validation_medians: Sequence[float],
    test_medians: Sequence[float],
    distribution: ResidualDistribution,
    num_rounds: int,
    early_stopping: int,
) -> ExpectationFit:
    """Fit the booster, compare with the shifted median, select, gate.

    ``*_medians`` are the served median predictions (offset and clip applied),
    the same numbers the residual distribution was fitted on.
    """
    import lightgbm as lgb
    import numpy as np

    objective = expectation_params(target)
    params = {**base_params, **objective}
    y_validation = np.asarray([row.target for row in validation_rows], dtype=np.float64)

    booster = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train, feature_name=list(feature_names)),
        num_boost_round=num_rounds,
        valid_sets=[lgb.Dataset(x_validation, label=y_validation)],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(booster.best_iteration or num_rounds)

    def boost(x: Any) -> list[float]:
        return [max(0.0, float(v)) for v in booster.predict(x, num_iteration=best_iteration)]

    candidates_validation = {
        SOURCE_BOOSTER: boost(x_validation),
        SOURCE_SHIFTED_MEDIAN: shifted_median(validation_medians, validation_rows, distribution),
    }
    candidates_test = {
        SOURCE_BOOSTER: boost(x_test),
        SOURCE_SHIFTED_MEDIAN: shifted_median(test_medians, test_rows, distribution),
        "median": [float(v) for v in test_medians],
    }

    def metrics_for(values: Sequence[float], rows: Sequence[TrainingRow]) -> Metrics:
        return compute_metrics(
            [(v, float(r.target)) for v, r in zip(values, rows, strict=True)],
            total_rows=len(rows),
        )

    validation_rmse = {
        name: metrics_for(values, validation_rows).rmse
        for name, values in candidates_validation.items()
    }
    test = {name: metrics_for(values, test_rows) for name, values in candidates_test.items()}
    team_totals = {
        name: team_total_error(values, test_rows) for name, values in candidates_test.items()
    }
    baseline_name, baseline_metrics = best_baseline_rmse(test_rows, target)

    report = ExpectationReport(
        source=SOURCE_NONE,
        validation_rmse=validation_rmse,
        test=test,
        team_totals=team_totals,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        booster_best_iteration=best_iteration,
        objective=objective,
    )
    if report.passes_gate:
        report.source = report.chosen
    else:
        logger.warning(
            "EXPECTED VALUE FAILED THE GATE - none will be published",
            extra={
                "target": str(target),
                "chosen": report.chosen,
                "rmse": round(test[report.chosen].rmse, 4),
                "median_rmse": round(test["median"].rmse, 4),
            },
        )
    if not report.beats_baseline:
        logger.warning(
            "expected value does not beat the best baseline on RMSE",
            extra={
                "target": str(target),
                "rmse": round(test[report.chosen].rmse, 4),
                "baseline": baseline_name,
                "baseline_rmse": round(baseline_metrics.rmse, 4),
            },
        )

    logger.info("fitted expectation", extra={"target": str(target), **report.summary()})
    return ExpectationFit(
        report=report, booster=booster if report.source == SOURCE_BOOSTER else None
    )

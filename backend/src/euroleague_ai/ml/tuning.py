"""Hyperparameter search.

The current settings were chosen conservatively when the feature set had about
a hundred and twenty columns. It now has a hundred and sixty-six, and nothing
was ever re-tuned -- which is the likeliest reason adding features has recently
made some targets slightly *worse* rather than better. ``feature_fraction`` of
0.7 samples a different fraction of a bigger space, and ``min_data_in_leaf`` of
100 is a stronger constraint when the trees have more to choose from.

**The rule this module exists to obey: select on validation, report on test.**

A search that picks the configuration with the best test score and then reports
that score is not measuring anything. With enough trials something will look
good on any fixed set of rows, and the number you publish is then the maximum of
a hundred noisy draws rather than an estimate of accuracy. So every trial is
scored on the *validation* split, the winner is chosen there, and only then is
it retrained and measured once on test -- against the current defaults, on the
same rows.

This project has already shipped one model whose offline gate looked fine while
a defect degraded every target. Selection bias would be the same failure with a
more respectable name.

**Random search rather than a grid.** With eight parameters a grid either
resolves each one coarsely or takes days. Random sampling explores far more
distinct values of the parameters that actually matter, and stops whenever you
run out of patience rather than at a predetermined corner of the space.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.matrix import derive_feature_names, to_matrix
from euroleague_ai.ml.train import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_PARAMS,
    TRAINING_SEED,
    chronological_split,
)
from euroleague_ai.models.enums import PredictionTarget

logger = get_logger(__name__)

#: Where the search looks. Ranges are deliberately wide around the current
#: values rather than centred on them -- a search that only confirms what you
#: already had is not worth the electricity.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "num_leaves": (15, 255),
    "max_depth": (4, 12),
    "min_data_in_leaf": (20, 300),
    "learning_rate": (0.02, 0.12),
    "feature_fraction": (0.4, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "lambda_l1": (0.0, 5.0),
    "lambda_l2": (0.0, 10.0),
}

#: Parameters that must stay integers.
INTEGER_PARAMS = frozenset({"num_leaves", "max_depth", "min_data_in_leaf"})

#: Below this the search is not worth running: forty random draws over eight
#: dimensions is already a coarse look, and a handful tells you nothing.
MIN_TRIALS = 10


@dataclass(slots=True)
class Trial:
    """One sampled configuration and how it did on validation."""

    params: dict[str, Any]
    validation_mae: float
    best_iteration: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "validation_mae": round(self.validation_mae, 4),
            "best_iteration": self.best_iteration,
        }


@dataclass(slots=True)
class SearchResult:
    """The outcome of a search for one target."""

    target: PredictionTarget
    trials: int
    baseline_validation_mae: float
    best: Trial
    #: Sorted best-first, for inspection.
    top: list[Trial] = field(default_factory=list)

    @property
    def validation_gain_pct(self) -> float:
        if not self.baseline_validation_mae:
            return 0.0
        return (
            (self.baseline_validation_mae - self.best.validation_mae)
            / self.baseline_validation_mae
            * 100.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": str(self.target),
            "trials": self.trials,
            "baseline_validation_mae": round(self.baseline_validation_mae, 4),
            "best_validation_mae": round(self.best.validation_mae, 4),
            "validation_gain_pct": round(self.validation_gain_pct, 2),
            "best_params": self.best.params,
            "top": [t.as_dict() for t in self.top],
        }


def sample_params(rng: random.Random) -> dict[str, Any]:
    """Draw one configuration from the search space."""
    params: dict[str, Any] = {}
    for name, (low, high) in SEARCH_SPACE.items():
        if name in INTEGER_PARAMS:
            params[name] = rng.randint(int(low), int(high))
        else:
            params[name] = round(rng.uniform(low, high), 4)

    # num_leaves must stay under 2**max_depth or the depth limit silently
    # governs and the two parameters stop being independent -- which would make
    # half the search space duplicates of the other half.
    params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"] - 1)
    return params


def _validation_mae(
    rows_train: list[TrainingRow],
    rows_validation: list[TrainingRow],
    feature_names: list[str],
    params: dict[str, Any],
) -> tuple[float, int]:
    """Fit on train, score on validation. Returns ``(mae, best_iteration)``.

    No calibration and no distribution: those belong to a model being shipped,
    and adding them here would only slow the search without changing which
    configuration wins.
    """
    import lightgbm as lgb
    import numpy as np

    x_train, y_train = to_matrix(rows_train, feature_names)
    x_validation, y_validation = to_matrix(rows_validation, feature_names)

    resolved = {**DEFAULT_PARAMS, **params}
    booster = lgb.train(
        resolved,
        lgb.Dataset(x_train, label=y_train, feature_name=list(feature_names)),
        num_boost_round=DEFAULT_NUM_ROUNDS,
        valid_sets=[lgb.Dataset(x_validation, label=y_validation)],
        callbacks=[
            lgb.early_stopping(DEFAULT_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    predicted = booster.predict(x_validation, num_iteration=booster.best_iteration)
    mae = float(np.mean(np.abs(np.asarray(predicted) - np.asarray(y_validation))))
    return mae, int(booster.best_iteration)


def search(
    rows: list[TrainingRow],
    target: PredictionTarget,
    *,
    trials: int = 40,
    seed: int = TRAINING_SEED,
    keep_top: int = 5,
) -> SearchResult:
    """Random search, selected on validation.

    The test split is not touched here at all. That is the whole point: it stays
    unused until the winner is retrained once, so the number finally reported is
    an estimate rather than the maximum of a hundred draws.

    Args:
        trials: Configurations to sample. Forty is a coarse but useful look;
            a hundred is better if you have the time.
        seed: Makes the search itself reproducible, so a surprising result can
            be re-examined rather than merely re-rolled.
    """
    if trials < MIN_TRIALS:
        raise ValueError(f"at least {MIN_TRIALS} trials, got {trials}")

    train_rows, validation_rows, _test_rows = chronological_split(rows)
    feature_names = derive_feature_names(rows)
    rng = random.Random(seed)

    baseline_mae, _ = _validation_mae(train_rows, validation_rows, feature_names, {})
    logger.info(
        "tuning baseline",
        extra={"target": str(target), "validation_mae": round(baseline_mae, 4)},
    )

    results: list[Trial] = []
    for index in range(trials):
        params = sample_params(rng)
        try:
            mae, best_iteration = _validation_mae(
                train_rows, validation_rows, feature_names, params
            )
        except Exception as exc:  # noqa: BLE001 - a bad draw must not end the search
            logger.warning(
                "trial failed", extra={"target": str(target), "error": str(exc)}
            )
            continue

        results.append(Trial(params=params, validation_mae=mae, best_iteration=best_iteration))
        if (index + 1) % 10 == 0:
            best_so_far = min(r.validation_mae for r in results)
            logger.info(
                "tuning progress",
                extra={
                    "target": str(target),
                    "trial": index + 1,
                    "of": trials,
                    "best_validation_mae": round(best_so_far, 4),
                },
            )

    if not results:
        raise RuntimeError(f"every trial failed for {target}")

    results.sort(key=lambda t: t.validation_mae)
    return SearchResult(
        target=target,
        trials=len(results),
        baseline_validation_mae=baseline_mae,
        best=results[0],
        top=results[:keep_top],
    )


#: Where tuned parameters live, beside the model artefacts. One file per target,
#: because the best settings for minutes and for three-pointers have no reason
#: to agree -- one is a rotation decision, the other is closer to a coin.
def params_path(target: PredictionTarget, root: Path | None = None) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    base = root or DEFAULT_ARTIFACT_ROOT
    return base / "params" / f"{str(target).lower()}.json"


def save_params(
    target: PredictionTarget, params: dict[str, Any], *, root: Path | None = None
) -> Path:
    """Persist tuned parameters for a target."""
    path = params_path(target, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_params(
    target: PredictionTarget, *, root: Path | None = None
) -> dict[str, Any] | None:
    """Tuned parameters for a target, or ``None`` if the search never ran.

    Returning ``None`` rather than the defaults keeps the caller able to say
    *which* it used, so a training run can report that it was tuned instead of
    quietly behaving differently from the last one.
    """
    path = params_path(target, root)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning("ignoring unreadable tuned params", extra={"path": str(path)})
        return None
    return loaded if isinstance(loaded, dict) else None

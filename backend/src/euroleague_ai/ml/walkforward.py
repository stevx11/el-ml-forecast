"""Walk-forward evaluation: train on every season before N, test on season N.

One 70/15/15 split gives one number, from one stretch of games. Whether a
change really helps, or happened to suit that stretch, only shows when the
same comparison is repeated season after season. This module repeats
production training once per recent season:

* **train**: every row from earlier seasons except the latest 15% of them;
* **validation**: that latest 15%, for early stopping, the calibration
  offset, the error distribution and the expectation head, exactly as in
  ``ml train``;
* **test**: the whole of season N, which no part of the fit has seen.

Each fold is judged like a production model, against the best baseline on the
same test rows. The summary is the spread across seasons (median, worst,
best, and how many seasons the model beats the baseline in), not one average.

The current season is a fold too once it has enough rows, and is marked
partial: a few rounds in, it is the noisiest fold and the most relevant one.

Nothing here touches the model registry or what is served. Reports are saved
as JSON so a later change (a new feature set, say) can be compared fold by fold
against today's.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.train import VALIDATION_FRACTION, train_target

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datetime import datetime

    from euroleague_ai.ml.dataset import TrainingRow
    from euroleague_ai.models.enums import PredictionTarget

logger = get_logger(__name__)

#: Seasons tested by default: the most recent ones, where the game looks most
#: like the one being predicted.
DEFAULT_SEASONS = 4

#: A season needs this many test rows to be a fold. Below it (the first round
#: or two of a new season), one hot shooting night moves the result.
MIN_TEST_ROWS = 1_000

#: And at least this many training seasons before it.
MIN_TRAIN_SEASONS = 2

REPORTS_DIR = Path("reports") / "walkforward"


@dataclass(frozen=True, slots=True)
class Fold:
    season: str
    train: list[TrainingRow]
    validation: list[TrainingRow]
    test: list[TrainingRow]
    partial: bool


def season_folds(
    rows: list[TrainingRow],
    *,
    seasons: int = DEFAULT_SEASONS,
    min_test_rows: int = MIN_TEST_ROWS,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
    current_season: str | None = None,
) -> list[Fold]:
    """The last ``seasons`` usable seasons as folds, oldest first.

    ``current_season`` marks that fold as partial (still being played).
    """
    by_season: dict[str, list[TrainingRow]] = {}
    for row in rows:
        by_season.setdefault(row.season_code, []).append(row)
    ordered = sorted(by_season)
    folds: list[Fold] = []
    for index, season in enumerate(ordered):
        if index < min_train_seasons or len(by_season[season]) < min_test_rows:
            continue
        pool = sorted(
            (row for earlier in ordered[:index] for row in by_season[earlier]),
            key=lambda row: row.tip_off_utc,
        )
        cut = len(pool) - max(1, int(len(pool) * VALIDATION_FRACTION))
        folds.append(
            Fold(
                season=season,
                train=pool[:cut],
                validation=pool[cut:],
                test=sorted(by_season[season], key=lambda row: row.tip_off_utc),
                partial=season == current_season,
            )
        )
    return folds[-seasons:] if seasons > 0 else folds


@dataclass(slots=True)
class FoldResult:
    season: str
    partial: bool
    train_rows: int
    test_rows: int
    model_mae: float
    baseline: str
    baseline_mae: float
    bias: float
    covered_80: float | None
    expected_rmse: float | None = None
    median_rmse: float | None = None

    @property
    def improvement(self) -> float:
        if not math.isfinite(self.baseline_mae) or self.baseline_mae == 0:
            return float("nan")
        return (self.baseline_mae - self.model_mae) / self.baseline_mae

    @property
    def beats_baseline(self) -> bool:
        return math.isfinite(self.model_mae) and self.model_mae < self.baseline_mae

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "partial": self.partial,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "model_mae": round(self.model_mae, 4),
            "baseline": self.baseline,
            "baseline_mae": round(self.baseline_mae, 4),
            "improvement_pct": round(self.improvement * 100, 2),
            "beats_baseline": self.beats_baseline,
            "bias": round(self.bias, 4),
            "covered_80": self.covered_80,
            "expected_rmse": self.expected_rmse,
            "median_rmse": self.median_rmse,
        }


@dataclass(slots=True)
class TargetWalkForward:
    target: PredictionTarget
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        gains = [f.improvement * 100 for f in self.folds if math.isfinite(f.improvement)]
        complete = [f for f in self.folds if not f.partial]
        complete_gains = [f.improvement * 100 for f in complete if math.isfinite(f.improvement)]
        return {
            "folds": len(self.folds),
            "beats_baseline": sum(f.beats_baseline for f in self.folds),
            "median_improvement_pct": round(statistics.median(gains), 2) if gains else None,
            "worst_improvement_pct": round(min(gains), 2) if gains else None,
            "best_improvement_pct": round(max(gains), 2) if gains else None,
            # Complete seasons only: the partial one is too short to weigh equally.
            "median_improvement_complete_pct": (
                round(statistics.median(complete_gains), 2) if complete_gains else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.name,
            "folds": [f.to_dict() for f in self.folds],
            "summary": self.summary(),
        }


def evaluate_target(
    rows: list[TrainingRow],
    target: PredictionTarget,
    *,
    seasons: int = DEFAULT_SEASONS,
    current_season: str | None = None,
    fit_expected: bool = False,
    min_test_rows: int = MIN_TEST_ROWS,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
    **train_kwargs: Any,
) -> TargetWalkForward:
    """Train and judge one model per fold.

    ``fit_expected`` is off by default: it doubles the cost, and the median is
    what the gate judges. Turn it on to see the expected-value RMSE per season.
    """
    result = TargetWalkForward(target)
    for fold in season_folds(
        rows,
        seasons=seasons,
        min_test_rows=min_test_rows,
        min_train_seasons=min_train_seasons,
        current_season=current_season,
    ):
        trained = train_target(
            rows,
            target,
            split=(fold.train, fold.validation, fold.test),
            fit_expected=fit_expected,
            **train_kwargs,
        )
        expected_rmse = median_rmse = None
        if trained.expectation is not None:
            chosen = trained.expectation.chosen
            test = trained.expectation.test
            if chosen in test:
                expected_rmse = round(test[chosen].rmse, 4)
            if "median" in test:
                median_rmse = round(test["median"].rmse, 4)
        result.folds.append(
            FoldResult(
                season=fold.season,
                partial=fold.partial,
                train_rows=len(fold.train),
                test_rows=len(fold.test),
                model_mae=trained.metrics.mae,
                baseline=trained.baseline_name,
                baseline_mae=trained.baseline_metrics.mae,
                bias=trained.metrics.bias,
                covered_80=trained.coverage.get("covered_80"),
                expected_rmse=expected_rmse,
                median_rmse=median_rmse,
            )
        )
        logger.info(
            "walk-forward fold",
            extra={"target": target.name, **result.folds[-1].to_dict()},
        )
    return result


def save_report(
    results: list[TargetWalkForward],
    *,
    feature_set_version: str,
    now: datetime,
    root: Path | None = None,
) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    directory = (root or DEFAULT_ARTIFACT_ROOT) / REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y%m%dT%H%M%S}-{feature_set_version}.json"
    path.write_text(
        json.dumps(
            {
                "feature_set_version": feature_set_version,
                "created_at": now.isoformat(),
                "targets": [r.to_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

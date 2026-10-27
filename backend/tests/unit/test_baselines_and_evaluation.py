"""Baselines, metrics and the walk-forward split.

No database. These cover the arithmetic that decides whether Phase 3 is allowed
to proceed, so being wrong here would mean measuring the gate incorrectly.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from euroleague_ai.ml.baselines import (
    LastNMean,
    MinutesScaledRate,
    SeasonMean,
    WeightedBlend,
    baseline_coverage,
)
from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.evaluation import (
    backtest,
    compare,
    compute_metrics,
    stratified_metrics,
    walk_forward_folds,
)
from euroleague_ai.models.enums import PredictionTarget

BASE = datetime(2025, 10, 1, 18, 0, tzinfo=UTC)


def row(
    *,
    features: dict[str, float | None],
    target: float,
    season: str = "E2025",
    round_number: int | None = 1,
    minutes: float = 25.0,
    day: int = 0,
) -> TrainingRow:
    return TrainingRow(
        snapshot_id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=uuid.uuid4(),
        season_code=season,
        round_number=round_number,
        tip_off_utc=BASE + timedelta(days=day),
        prediction_timestamp=BASE + timedelta(days=day) - timedelta(hours=1),
        features=features,
        target=target,
        actual_minutes=minutes,
    )


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        metrics = compute_metrics([(10.0, 10.0), (20.0, 20.0)])
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == pytest.approx(1.0)
        assert metrics.bias == 0.0

    def test_mae_and_rmse_differ_on_skewed_error(self) -> None:
        """RMSE punishes the outlier far harder. That is why MAE is the headline.

        Box score distributions are right-skewed; judging by RMSE sacrifices the
        bulk of predictions to chase 40-point nights.
        """
        pairs = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0), (10.0, 40.0)]
        metrics = compute_metrics(pairs)
        assert metrics.mae == pytest.approx(7.5)
        assert metrics.rmse == pytest.approx(15.0)
        assert metrics.rmse > metrics.mae

    def test_bias_is_signed(self) -> None:
        """Systematic over-prediction must be visible; MAE alone hides it."""
        assert compute_metrics([(12.0, 10.0), (13.0, 11.0)]).bias == pytest.approx(2.0)
        assert compute_metrics([(8.0, 10.0), (9.0, 11.0)]).bias == pytest.approx(-2.0)

    def test_r2_is_zero_when_actuals_are_constant(self) -> None:
        """Undefined rather than infinite -- must not emit NaN into a table."""
        assert compute_metrics([(9.0, 10.0), (11.0, 10.0)]).r2 == 0.0

    def test_coverage_reflects_rows_not_predicted(self) -> None:
        """A predictor covering only the easy half must not look better for it."""
        metrics = compute_metrics([(10.0, 10.0)], total_rows=4)
        assert metrics.coverage == pytest.approx(0.25)

    def test_empty_is_nan_not_a_crash(self) -> None:
        metrics = compute_metrics([])
        assert metrics.n == 0
        assert math.isnan(metrics.mae)


class TestStratification:
    def test_splits_by_realised_minutes(self) -> None:
        """Overall accuracy hides how bad a model is on rotation players."""
        rows = [
            row(features={}, target=2.0, minutes=5.0),
            row(features={}, target=10.0, minutes=15.0),
            row(features={}, target=18.0, minutes=25.0),
            row(features={}, target=25.0, minutes=35.0),
        ]
        buckets = stratified_metrics(rows, [2.0, 10.0, 18.0, 25.0])
        assert set(buckets) == {"0-10", "10-20", "20-30", "30+"}
        assert all(bucket["n"] == 1 for bucket in buckets.values())

    def test_missing_predictions_reduce_bucket_coverage(self) -> None:
        rows = [row(features={}, target=10.0, minutes=15.0) for _ in range(4)]
        buckets = stratified_metrics(rows, [10.0, None, None, None])
        assert buckets["10-20"]["coverage"] == pytest.approx(0.25)


class TestBaselines:
    TARGET = PredictionTarget.POINTS

    def test_season_mean(self) -> None:
        r = row(features={"season_points_mean": 14.0}, target=18.0)
        assert SeasonMean().predict(r, self.TARGET) == pytest.approx(14.0)

    def test_last_n_mean(self) -> None:
        r = row(features={"last5_points_mean": 16.0, "last10_points_mean": 12.0}, target=18.0)
        assert LastNMean(window=5).predict(r, self.TARGET) == pytest.approx(16.0)
        assert LastNMean(window=10).predict(r, self.TARGET) == pytest.approx(12.0)

    def test_weighted_blend(self) -> None:
        r = row(features={"season_points_mean": 10.0, "last5_points_mean": 20.0}, target=18.0)
        assert WeightedBlend().predict(r, self.TARGET) == pytest.approx(0.4 * 10 + 0.6 * 20)

    def test_weighted_blend_falls_back_when_one_side_is_missing(self) -> None:
        """Early in a season only one window exists; refusing to predict is worse."""
        only_season = row(features={"season_points_mean": 10.0}, target=12.0)
        only_recent = row(features={"last5_points_mean": 20.0}, target=12.0)
        assert WeightedBlend().predict(only_season, self.TARGET) == pytest.approx(10.0)
        assert WeightedBlend().predict(only_recent, self.TARGET) == pytest.approx(20.0)

    def test_minutes_scaled_rate(self) -> None:
        """0.6 points/minute over a season, projected onto 30 recent minutes."""
        r = row(
            features={
                "season_points_mean": 18.0,
                "season_minutes_mean": 30.0,
                "last5_minutes_mean": 30.0,
            },
            target=20.0,
        )
        assert MinutesScaledRate().predict(r, self.TARGET) == pytest.approx(18.0)

    def test_minutes_scaled_rate_reacts_to_a_minutes_change(self) -> None:
        """The point of the baseline: rising minutes should raise the projection.

        A player averaging 18 points in 30 minutes who is now playing 36 should
        be projected higher, and a plain season mean cannot express that.
        """
        r = row(
            features={
                "season_points_mean": 18.0,
                "season_minutes_mean": 30.0,
                "last5_minutes_mean": 36.0,
            },
            target=22.0,
        )
        assert MinutesScaledRate().predict(r, self.TARGET) == pytest.approx(21.6)
        assert SeasonMean().predict(r, self.TARGET) == pytest.approx(18.0)

    def test_minutes_scaled_rate_degenerates_correctly_for_minutes(self) -> None:
        """Applied to MINUTES it is just the recent minutes average."""
        r = row(features={"last5_minutes_mean": 27.0, "season_minutes_mean": 24.0}, target=28.0)
        assert MinutesScaledRate().predict(r, PredictionTarget.MINUTES) == pytest.approx(27.0)

    def test_returns_none_rather_than_guessing_zero(self) -> None:
        """A rookie's projection is unknown. Zero would score undeservedly well."""
        empty = row(features={}, target=10.0)
        for baseline in (SeasonMean(), LastNMean(), WeightedBlend(), MinutesScaledRate()):
            assert baseline.predict(empty, self.TARGET) is None

    def test_zero_season_minutes_does_not_divide_by_zero(self) -> None:
        r = row(
            features={
                "season_points_mean": 5.0,
                "season_minutes_mean": 0.0,
                "last5_minutes_mean": 10.0,
            },
            target=8.0,
        )
        assert MinutesScaledRate().predict(r, self.TARGET) is None

    def test_coverage_reported_per_baseline(self) -> None:
        rows = [
            row(features={"season_points_mean": 10.0}, target=12.0),
            row(features={}, target=8.0),
        ]
        coverage = baseline_coverage(rows, self.TARGET)
        assert coverage["season_mean"] == pytest.approx(0.5)
        assert coverage["minutes_scaled_rate"] == pytest.approx(0.0)


class TestWalkForward:
    def test_train_is_strictly_earlier_than_test(self) -> None:
        """The property the whole harness exists to guarantee."""
        rows = [row(features={}, target=float(r), round_number=r, day=r) for r in range(1, 11)]
        folds = walk_forward_folds(rows, min_train_rounds=5)

        assert folds
        for fold in folds:
            assert all(train.round_number < fold.round_number for train in fold.train), (
                "a training row is not earlier than the fold it trains for"
            )

    def test_early_rounds_are_not_evaluated(self) -> None:
        """Form windows are empty there; evaluating them measures missingness."""
        rows = [row(features={}, target=1.0, round_number=r, day=r) for r in range(1, 11)]
        folds = walk_forward_folds(rows, min_train_rounds=5)
        assert min(fold.round_number for fold in folds) == 6

    def test_seasons_never_mix(self) -> None:
        """Training on next season to predict this one must be impossible."""
        rows = [
            row(features={}, target=1.0, season="E2024", round_number=r, day=r)
            for r in range(1, 11)
        ] + [
            row(features={}, target=1.0, season="E2025", round_number=r, day=100 + r)
            for r in range(1, 11)
        ]
        for fold in walk_forward_folds(rows, min_train_rounds=5):
            assert all(train.season_code == fold.season_code for train in fold.train)

    def test_rows_without_a_round_are_excluded(self) -> None:
        rows = [row(features={}, target=1.0, round_number=None) for _ in range(10)]
        assert walk_forward_folds(rows) == []

    def test_backtest_runs_a_predictor_over_folds(self) -> None:
        rows = [
            row(features={"season_points_mean": 10.0}, target=10.0, round_number=r, day=r)
            for r in range(1, 11)
        ]
        result = backtest(rows, lambda r: SeasonMean().predict(r, PredictionTarget.POINTS))
        assert result["folds"] == 5
        assert result["overall"]["mae"] == pytest.approx(0.0)

    def test_backtest_with_no_folds_is_not_a_crash(self) -> None:
        assert backtest([], lambda r: 1.0)["folds"] == 0


class TestCompare:
    def test_ranks_by_mae_best_first(self) -> None:
        results = {
            "worse": {
                "overall": {
                    "mae": 5.0,
                    "rmse": 6.0,
                    "r2": 0.1,
                    "bias": 0.0,
                    "coverage": 1.0,
                    "n": 10,
                }
            },
            "better": {
                "overall": {
                    "mae": 3.0,
                    "rmse": 4.0,
                    "r2": 0.4,
                    "bias": 0.0,
                    "coverage": 1.0,
                    "n": 10,
                }
            },
        }
        assert [entry["predictor"] for entry in compare(results)] == ["better", "worse"]

    def test_uncovered_predictor_cannot_top_the_table(self) -> None:
        """NaN must sort last, not first."""
        results = {
            "real": {
                "overall": {
                    "mae": 3.0,
                    "rmse": 4.0,
                    "r2": 0.4,
                    "bias": 0.0,
                    "coverage": 1.0,
                    "n": 10,
                }
            },
            "empty": {
                "overall": {
                    "mae": math.nan,
                    "rmse": math.nan,
                    "r2": math.nan,
                    "bias": math.nan,
                    "coverage": 0.0,
                    "n": 0,
                }
            },
        }
        assert compare(results)[0]["predictor"] == "real"

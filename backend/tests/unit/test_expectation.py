"""Expected values: selection, gate, squad totals, persistence and serving."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.distribution import ResidualDistribution, fit_residual_distribution
from euroleague_ai.ml.evaluation import Metrics
from euroleague_ai.ml.expectation import (
    SOURCE_BOOSTER,
    SOURCE_NONE,
    SOURCE_SHIFTED_MEDIAN,
    ExpectationReport,
    expectation_params,
    shifted_median,
    team_total_error,
)
from euroleague_ai.ml.inference import LoadedModel, load_model
from euroleague_ai.ml.outcomes import summarise_team_totals
from euroleague_ai.ml.registry import expectation_paths, save_artifact
from euroleague_ai.ml.train import train_target
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import ModelVersion

T0 = datetime(2019, 10, 1, 18, 0, tzinfo=UTC)


def _row(target: float, *, game: uuid.UUID, home: float, features: dict[str, Any]) -> TrainingRow:
    return TrainingRow(
        snapshot_id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=game,
        season_code="E2019",
        round_number=None,
        tip_off_utc=T0,
        prediction_timestamp=T0,
        features={"is_home": home, **features},
        target=target,
        actual_minutes=20.0,
    )


def _metrics(rmse: float) -> Metrics:
    return Metrics(n=10, mae=rmse, rmse=rmse, r2=0.0, bias=0.0, coverage=1.0)


def _skewed_world(games: int = 1200, seed: int = 0) -> list[TrainingRow]:
    """Negative-binomial points: right-skewed, so median < mean, like real box scores.

    Every feature is a *noisy* view of the true mean, so no baseline is exact and
    the median/mean gap is the dominant, learnable error.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[TrainingRow] = []
    for g in range(games):
        game = uuid.uuid4()
        for home in (0.0, 1.0):
            for _ in range(11):
                mu = rng.uniform(1.0, 16.0)
                points = float(rng.negative_binomial(2, 2 / (2 + mu)))
                rows.append(
                    TrainingRow(
                        snapshot_id=uuid.uuid4(),
                        player_id=uuid.uuid4(),
                        game_id=game,
                        season_code="E2019",
                        round_number=None,
                        tip_off_utc=T0 + timedelta(hours=g),
                        prediction_timestamp=T0 + timedelta(hours=g),
                        features={
                            "is_home": home,
                            "season_points_mean": mu * rng.normal(1.0, 0.25),
                            "last5_points_mean": mu * rng.normal(1.0, 0.4),
                            "signal": mu,
                            "games_played_career_window": 30.0,
                        },
                        target=points,
                        actual_minutes=20.0,
                    )
                )
    return rows


class TestTeamTotals:
    def test_sums_per_side_of_each_game(self) -> None:
        game = uuid.uuid4()
        rows = [
            _row(10, game=game, home=1.0, features={}),
            _row(4, game=game, home=1.0, features={}),
            _row(7, game=game, home=0.0, features={}),
        ]
        result = team_total_error([9.0, 3.0, 9.0], rows)
        # home 12 vs 14 (-2), away 9 vs 7 (+2)
        assert result == {"team_games": 2, "mae": 2.0, "bias": 0.0}

    def test_summarise_team_totals_handles_missing_actuals_as_zero(self) -> None:
        key = ("g", "t")
        summary = summarise_team_totals({"median": {key: 70.0}, "expected": {}}, {key: 80.0})
        assert summary == {"median": {"team_games": 1, "mae": 10.0, "bias": -10.0}}


class TestShiftedMedian:
    def test_adds_the_cell_mean_and_clips(self) -> None:
        # Residuals skewed upwards: median residual 0, mean residual +1.
        residuals = [-1.0, 0.0, 0.0, 0.0, 6.0] * 60
        predictions = [5.0] * len(residuals)
        distribution = fit_residual_distribution(
            predictions,
            [p + r for p, r in zip(predictions, residuals, strict=True)],
            [30.0] * len(residuals),
        )
        assert distribution.mean_residual(5.0, 30.0) == pytest.approx(1.0)
        assert shifted_median([5.0], None, distribution, histories=[30.0]) == [pytest.approx(6.0)]

        negative = ResidualDistribution(level_edges=[], cells={}, by_history={}, pooled=[-9.0])
        assert shifted_median([1.0], None, negative, histories=[None]) == [0.0]

    def test_needs_rows_or_histories(self) -> None:
        distribution = ResidualDistribution(level_edges=[], cells={}, by_history={}, pooled=[0.0])
        with pytest.raises(ValueError):
            shifted_median([1.0], None, distribution)


class TestObjectives:
    def test_minutes_use_squared_error_and_counts_tweedie(self) -> None:
        assert expectation_params(PredictionTarget.MINUTES)["objective"] == "regression"
        points = expectation_params(PredictionTarget.POINTS)
        assert points["objective"] == "tweedie"
        assert 1.0 < points["tweedie_variance_power"] < 2.0
        assert points["metric"] == "l2"


class TestGate:
    def _report(
        self, *, chosen_rmse: float, median_rmse: float, chosen_bias: float, median_bias: float
    ) -> ExpectationReport:
        return ExpectationReport(
            source=SOURCE_NONE,
            validation_rmse={SOURCE_BOOSTER: 1.0, SOURCE_SHIFTED_MEDIAN: 2.0},
            test={SOURCE_BOOSTER: _metrics(chosen_rmse), "median": _metrics(median_rmse)},
            team_totals={
                SOURCE_BOOSTER: {"bias": chosen_bias},
                "median": {"bias": median_bias},
            },
            baseline_name="season_mean",
            baseline_metrics=_metrics(0.5),
        )

    def test_selection_is_made_on_validation(self) -> None:
        report = self._report(chosen_rmse=4.0, median_rmse=5.0, chosen_bias=0.1, median_bias=-8)
        assert report.chosen == SOURCE_BOOSTER

    def test_passes_when_it_beats_the_median_on_both(self) -> None:
        report = self._report(chosen_rmse=4.0, median_rmse=5.0, chosen_bias=0.3, median_bias=-8)
        assert report.passes_gate
        assert not report.beats_baseline  # reported, but not a gate

    def test_fails_on_worse_rmse(self) -> None:
        report = self._report(chosen_rmse=5.1, median_rmse=5.0, chosen_bias=0.3, median_bias=-8)
        assert not report.passes_gate

    def test_fails_when_team_totals_get_worse(self) -> None:
        report = self._report(chosen_rmse=4.0, median_rmse=5.0, chosen_bias=9.0, median_bias=-8)
        assert not report.passes_gate


@pytest.fixture(scope="module")
def result() -> Any:
    """One real training run, shared: it takes a couple of seconds."""
    return train_target(_skewed_world(), PredictionTarget.POINTS)


class TestTrainingEndToEnd:
    def test_expected_value_closes_the_squad_total_gap(self, result: Any) -> None:
        expectation = result.expectation
        assert expectation is not None
        assert expectation.source in (SOURCE_BOOSTER, SOURCE_SHIFTED_MEDIAN)
        median_bias = expectation.team_totals["median"]["bias"]
        expected_bias = expectation.team_totals[expectation.source]["bias"]
        assert median_bias < -3.0  # eleven skewed medians fall well short
        assert abs(expected_bias) < abs(median_bias) / 3

    def test_summary_carries_the_report(self, result: Any) -> None:
        summary = result.summary()["expectation"]
        assert summary["passes_gate"] is True
        assert set(summary["test"]) == {SOURCE_BOOSTER, SOURCE_SHIFTED_MEDIAN, "median"}

    def test_artifact_round_trip_serves_the_same_expectation(
        self, result: Any, tmp_path: Path
    ) -> None:
        model_path = save_artifact(result, name="points_lgbm", version="9.9.9", root=tmp_path)
        booster_path, report_path = expectation_paths(model_path)
        assert json.loads(report_path.read_text())["source"] == result.expectation.source
        assert booster_path.is_file() == (result.expectation.source == SOURCE_BOOSTER)

        loaded = load_model(
            ModelVersion(
                artifact_path=str(model_path),
                metrics={"calibration_offset": result.calibration_offset},
                target=PredictionTarget.POINTS,
            )
        )
        assert loaded.expected_source == result.expectation.source
        features = [{"signal": 12.0, "season_points_mean": 12.0, "is_home": 1.0}]
        medians = loaded.predict_batch(features)
        expected = loaded.expected_batch(features, medians)
        assert expected[0] is not None
        assert expected[0] > medians[0]  # a skewed stat averages above its median

    def test_served_spread_is_refitted_on_validation_and_test(self, result: Any) -> None:
        served = result.coverage["served_fit_rows"]
        assert served == result.validation_rows + result.test_rows
        assert result.distribution.sample_size == served

    def test_tuner_can_skip_the_head(self) -> None:
        quick = train_target(
            _skewed_world(games=300, seed=1), PredictionTarget.POINTS, fit_expected=False
        )
        assert quick.expectation is None
        assert quick.summary()["expectation"] is None


class _Booster:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: Any, **_: Any) -> list[float]:
        return [self.value] * len(x)


class TestServingSources:
    def _model(self, **kwargs: Any) -> LoadedModel:
        defaults: dict[str, Any] = {
            "model_version": None,
            "booster": _Booster(4.0),
            "feature_names": ["x"],
            "calibration_offset": 0.0,
            "target": PredictionTarget.POINTS,
        }
        return LoadedModel(**{**defaults, **kwargs})

    def test_legacy_models_get_the_shifted_median(self, tmp_path: Path) -> None:
        """A model directory with a distribution but no expectation record (every
        model trained before this change) serves the shifted median."""
        import lightgbm as lgb
        import numpy as np

        booster = lgb.train(
            {"objective": "regression_l1", "verbosity": -1, "min_data_in_leaf": 1},
            lgb.Dataset(np.arange(20, dtype=float).reshape(-1, 1), label=np.arange(20.0)),
            num_boost_round=3,
        )
        directory = tmp_path / "points_lgbm" / "0.1.8"
        directory.mkdir(parents=True)
        booster.save_model(str(directory / "model.txt"))
        (directory / "columns.json").write_text(json.dumps({"feature_names": ["Column_0"]}))
        (directory / "distribution.json").write_text(
            json.dumps(
                ResidualDistribution(
                    level_edges=[], cells={}, by_history={}, pooled=[0.0, 0.0, 3.0]
                ).to_dict()
            )
        )
        loaded = load_model(
            ModelVersion(
                artifact_path=str(directory / "model.txt"),
                metrics={},
                target=PredictionTarget.POINTS,
            )
        )
        assert loaded.expected_source == SOURCE_SHIFTED_MEDIAN
        assert loaded.expected_batch([{"Column_0": 5.0}], [5.0]) == [pytest.approx(6.0)]

    def test_booster_source(self) -> None:
        model = self._model(expected_source=SOURCE_BOOSTER, expected_booster=_Booster(7.5))
        assert model.expected_batch([{"x": 1.0}], [4.0]) == [7.5]

    def test_failed_gate_publishes_nothing(self) -> None:
        model = self._model(expected_source=SOURCE_NONE)
        assert model.expected_batch([{"x": 1.0}], [4.0]) == [None]

    def test_play_probability_has_no_expected_value(self) -> None:
        model = self._model(
            target=PredictionTarget.PLAYS,
            expected_source=SOURCE_BOOSTER,
            expected_booster=_Booster(1.0),
        )
        assert model.expected_batch([{"x": 1.0}], [0.9]) == [None]

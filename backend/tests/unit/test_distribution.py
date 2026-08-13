"""Distribution, threshold probability and confidence maths.

A predicted interval is worth nothing unless it is calibrated, and calibration
is not something you can eyeball -- an 80% interval holding 62% of outcomes
looks perfectly reasonable in a table. These tests build worlds with known
answers and check the numbers against them.
"""

from __future__ import annotations

import random

import pytest

from euroleague_ai.ml.distribution import (
    MIN_CELL_SIZE,
    ResidualDistribution,
    coverage_report,
    fit_residual_distribution,
    history_bucket,
    thresholds_for,
)

# No marker: unit tests are selected by directory (`pytest backend/tests/unit`),
# and `--strict-markers` rejects anything not registered in pyproject.toml. Only
# `integration` is registered, because that is the one that changes what the test
# needs -- a live database.


def _flat_world(
    n: int = 4000, spread: float = 5.0, seed: int = 7
) -> tuple[list[float], list[float], list[float | None]]:
    """Predictions with symmetric noise and plenty of history."""
    rng = random.Random(seed)
    predictions = [rng.uniform(2.0, 30.0) for _ in range(n)]
    actuals = [p + rng.uniform(-spread, spread) for p in predictions]
    games = [float(rng.randint(20, 60)) for _ in range(n)]
    return predictions, actuals, games


class TestHistoryBuckets:
    def test_none_and_zero_mean_the_same_thing(self) -> None:
        """A missing count and a count of zero both mean "we know nothing"."""
        assert history_bucket(None) == history_bucket(0.0) == 0

    def test_buckets_increase_with_history(self) -> None:
        assert history_bucket(3.0) == 1
        assert history_bucket(10.0) == 2
        assert history_bucket(50.0) == 3


class TestQuantiles:
    def test_quantiles_never_decrease(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        values = list(dist.quantiles(12.0, 40.0).values())
        assert values == sorted(values), values

    def test_quantiles_are_clipped_at_zero(self) -> None:
        """No player scores negative points, however wide the interval."""
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        assert all(v >= 0.0 for v in dist.quantiles(0.5, 40.0).values())

    def test_the_median_sits_near_the_point_estimate(self) -> None:
        """Symmetric noise means the 0.5 quantile should land on the prediction."""
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        assert dist.quantiles(15.0, 40.0)["0.5"] == pytest.approx(15.0, abs=0.6)


class TestCalibration:
    """The gate. An interval that does not cover what it claims is worthless."""

    def test_the_80_percent_interval_covers_about_80_percent(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        held_out = _flat_world(n=3000, seed=99)
        report = coverage_report(dist, *held_out)
        assert report["covered_80"] == pytest.approx(0.80, abs=0.03)

    def test_the_50_percent_interval_covers_about_50_percent(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        held_out = _flat_world(n=3000, seed=123)
        report = coverage_report(dist, *held_out)
        assert report["covered_50"] == pytest.approx(0.50, abs=0.04)

    def test_misses_are_reported_on_both_sides(self) -> None:
        """A single coverage number hides *which way* an interval is wrong."""
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)
        report = coverage_report(dist, *_flat_world(n=3000, seed=5))

        assert report["covered_80"] + report["below_interval"] + report[
            "above_interval"
        ] == pytest.approx(1.0, abs=1e-6)


class TestThresholdProbabilities:
    def test_probability_falls_as_the_bar_rises(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        odds = [dist.probability_at_least(15.0, 40.0, t) for t in (10, 15, 20, 25)]
        assert odds == sorted(odds, reverse=True), odds

    def test_probabilities_stay_in_the_unit_interval(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        for threshold in (-5.0, 0.0, 15.0, 200.0):
            value = dist.probability_at_least(15.0, 40.0, threshold)
            assert 0.0 <= value <= 1.0

    def test_the_median_threshold_is_a_coin_flip(self) -> None:
        """P(actual >= prediction) is about a half when errors are symmetric."""
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)

        assert dist.probability_at_least(15.0, 40.0, 15.0) == pytest.approx(0.5, abs=0.05)

    def test_an_unreachable_bar_is_impossible_not_merely_unlikely(self) -> None:
        predictions, actuals, games = _flat_world()
        dist = fit_residual_distribution(predictions, actuals, games)
        assert dist.probability_at_least(5.0, 40.0, 500.0) == 0.0


class TestConfidenceReflectsWhatWeActuallyKnow:
    """The point of conditioning on history depth.

    A player with no prior EuroLeague games gets the model's fallback constant.
    Presenting that with the same interval as a well-known starter is dishonest,
    and this is the test that keeps it from happening.
    """

    def _mixed_world(
        self, seed: int = 11
    ) -> tuple[list[float], list[float], list[float | None]]:
        rng = random.Random(seed)
        predictions: list[float] = []
        actuals: list[float] = []
        games: list[float | None] = []

        # Known players: tight errors.
        for _ in range(3 * MIN_CELL_SIZE):
            p = rng.uniform(8.0, 20.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-2.0, 2.0))
            games.append(float(rng.randint(30, 60)))

        # Unknown players: same predictions, far wider errors.
        for _ in range(3 * MIN_CELL_SIZE):
            p = rng.uniform(8.0, 20.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-14.0, 14.0))
            games.append(0.0)

        return predictions, actuals, games

    def test_an_unknown_player_gets_a_wider_interval(self) -> None:
        dist = fit_residual_distribution(*self._mixed_world())

        known = dist.quantiles(14.0, 45.0)
        unknown = dist.quantiles(14.0, 0.0)
        known_width = known["0.9"] - known["0.1"]
        unknown_width = unknown["0.9"] - unknown["0.1"]

        assert unknown_width > known_width * 2.0, (known_width, unknown_width)

    def test_an_unknown_player_gets_lower_confidence(self) -> None:
        dist = fit_residual_distribution(*self._mixed_world())
        assert dist.confidence(14.0, 0.0) < dist.confidence(14.0, 45.0)

    def test_confidence_stays_in_the_unit_interval(self) -> None:
        dist = fit_residual_distribution(*self._mixed_world())
        for games in (None, 0.0, 3.0, 12.0, 80.0):
            assert 0.0 < dist.confidence(14.0, games) <= 1.0

    def test_spread_is_larger_for_the_unknown_player(self) -> None:
        dist = fit_residual_distribution(*self._mixed_world())
        assert dist.spread(14.0, 0.0) > dist.spread(14.0, 45.0)


class TestRareBucketsSurviveTheFallback:
    """Regression: the fallback used to eat the signal it existed to preserve.

    Debut games are rare -- a player has exactly one, ever -- so in any
    validation slice the zero-history bucket is a couple of percent of rows. With
    a single size threshold for both the 2D cells and the history marginal, that
    bucket came in under the bar, fell through to the pooled sample, and every
    player in the league received an identical confidence of 0.50. The output
    looked plausible, which is what made it dangerous.
    """

    def _realistic_world(
        self,
    ) -> tuple[list[float], list[float], list[float | None]]:
        rng = random.Random(3)
        predictions: list[float] = []
        actuals: list[float] = []
        games: list[float | None] = []

        for _ in range(9600):
            p = rng.uniform(2.0, 30.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-5.0, 5.0))
            games.append(float(rng.randint(20, 60)))

        # Roughly 2.6% of rows, which is about what a real slice contains.
        for _ in range(260):
            p = rng.uniform(4.0, 18.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-13.0, 13.0))
            games.append(0.0)

        return predictions, actuals, games

    def test_a_rare_debut_bucket_is_still_used(self) -> None:
        dist = fit_residual_distribution(*self._realistic_world())
        assert len(dist.residuals_for(14.0, 0.0)) < dist.sample_size

    def test_confidence_separates_known_from_unknown(self) -> None:
        dist = fit_residual_distribution(*self._realistic_world())
        assert dist.confidence(14.0, 0.0) < dist.confidence(14.0, 45.0) - 0.05

    def test_occupancy_exposes_the_debut_count(self) -> None:
        """So a silent fallback is visible in the training report."""
        dist = fit_residual_distribution(*self._realistic_world())
        occupancy = dist.occupancy()
        assert occupancy["history_0"] == 260
        assert occupancy["total"] == 9860

    def test_a_genuinely_tiny_bucket_still_falls_back(self) -> None:
        """The threshold was lowered, not removed. Twenty rows is still noise."""
        rng = random.Random(4)
        predictions = []
        actuals = []
        games: list[float | None] = []
        for _ in range(4000):
            p = rng.uniform(2.0, 30.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-5.0, 5.0))
            games.append(50.0)
        for _ in range(20):
            p = rng.uniform(4.0, 18.0)
            predictions.append(p)
            actuals.append(p + rng.uniform(-13.0, 13.0))
            games.append(0.0)

        dist = fit_residual_distribution(predictions, actuals, games)
        assert dist.residuals_for(14.0, 0.0) is dist.pooled


class TestThinCellsDegradeSafely:
    def test_a_sparse_cell_falls_back_rather_than_inventing_precision(self) -> None:
        """One observation must not become a razor-thin interval."""
        predictions, actuals, games = _flat_world(n=2000)
        # A single very high prediction, alone in its corner.
        predictions.append(500.0)
        actuals.append(500.0)
        games.append(41.0)

        dist = fit_residual_distribution(predictions, actuals, games)
        residuals = dist.residuals_for(500.0, 41.0)
        assert len(residuals) >= MIN_CELL_SIZE

    def test_pooled_is_the_last_resort(self) -> None:
        predictions, actuals, games = _flat_world(n=1000)
        dist = fit_residual_distribution(predictions, actuals, games)
        # A history depth never seen during fitting still resolves.
        assert dist.residuals_for(10.0, 7.0)


class TestRoundTrip:
    def test_serialisation_preserves_the_answers(self) -> None:
        """The distribution travels to disk beside the booster; it must survive."""
        predictions, actuals, games = _flat_world()
        original = fit_residual_distribution(predictions, actuals, games)
        restored = ResidualDistribution.from_dict(original.to_dict())

        assert restored.quantiles(12.0, 40.0) == original.quantiles(12.0, 40.0)
        assert restored.probability_at_least(
            12.0, 40.0, 15.0
        ) == original.probability_at_least(12.0, 40.0, 15.0)
        assert restored.confidence(12.0, 40.0) == original.confidence(12.0, 40.0)


class TestThresholdsAreDefinedForEveryTarget:
    def test_every_served_target_publishes_thresholds(self) -> None:
        for target in (
            "MINUTES",
            "POINTS",
            "REBOUNDS",
            "ASSISTS",
            "THREE_POINTERS_MADE",
        ):
            assert thresholds_for(target), target

    def test_the_brief_s_points_thresholds_are_present(self) -> None:
        """The original spec asks for P(15+), P(20+) and P(25+)."""
        assert {15.0, 20.0, 25.0} <= set(thresholds_for("POINTS"))

    def test_an_unknown_target_returns_nothing_rather_than_raising(self) -> None:
        assert thresholds_for("STEALS") == ()


class TestFitRejectsMismatchedInputs:
    def test_misaligned_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="align"):
            fit_residual_distribution([1.0, 2.0], [1.0], [None, None])

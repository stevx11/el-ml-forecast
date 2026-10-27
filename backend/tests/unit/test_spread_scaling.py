"""Per-player spreads: the error distribution scaled by each player's own swing.

Two players projected for the same 12 points, one a streaky shooter and one a
steady big man, should not get the same range. These tests build leagues where
the answer is known: one where players really differ in volatility (scaling
must be chosen and must fix coverage for both kinds of player), and one where
they do not (scaling must be refused, leaving the spread exactly as before).
"""

from __future__ import annotations

import random

import pytest

from euroleague_ai.ml.distribution import (
    SCALE_BOUNDS,
    ResidualDistribution,
    choose_volatility_scaling,
    coverage_report,
    fit_residual_distribution,
    quantile_loss,
    volatility_features,
)
from euroleague_ai.ml.expectation import shifted_median


def _league(
    *, heterogeneous: bool, n: int = 9000, seed: int = 11
) -> tuple[list[float], list[float], list[float | None], list[float | None]]:
    """Projections, outcomes, history depths and each player's recent swing.

    In the heterogeneous league a third of the players swing three times as much
    as another third, and their recent standard deviation says so (with noise).
    """
    rng = random.Random(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    histories: list[float | None] = []
    swings: list[float | None] = []
    for _ in range(n):
        predicted = rng.uniform(4.0, 20.0)
        sigma = rng.choice([1.0, 2.0, 3.0]) if heterogeneous else 2.0
        predictions.append(predicted)
        actuals.append(max(0.0, predicted + rng.gauss(0.0, sigma)))
        histories.append(float(rng.randint(20, 60)))
        swings.append(sigma * rng.uniform(0.8, 1.2))
    return predictions, actuals, histories, swings


def _split(values: list, fraction: float = 0.6) -> tuple[list, list]:
    cut = int(len(values) * fraction)
    return values[:cut], values[cut:]


class TestScale:
    def test_unscaled_distribution_scales_nothing(self) -> None:
        p, a, h, _ = _league(heterogeneous=True, n=1000)
        distribution = fit_residual_distribution(p, a, h)
        assert not distribution.scaled
        assert distribution.scale(10.0, 5.0) == 1.0

    def test_unknown_volatility_gets_the_cell_spread(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=2000)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=1.0
        )
        assert distribution.scale(10.0, None) == 1.0
        assert distribution.scale(10.0, float("nan")) == 1.0

    def test_scale_is_bounded(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=2000)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=1.0
        )
        low, high = SCALE_BOUNDS
        assert distribution.scale(10.0, 0.0) == low
        assert distribution.scale(10.0, 1e6) == high

    def test_power_zero_is_exactly_the_old_distribution(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=3000)
        old = fit_residual_distribution(p, a, h)
        new = fit_residual_distribution(p, a, h, v, volatility_feature="std10_points")
        for predicted, games, vol in zip(p[:50], h[:50], v[:50], strict=True):
            assert new.quantiles(predicted, games, vol) == old.quantiles(predicted, games)

    def test_a_streaky_player_gets_a_wider_range_than_a_steady_one(self) -> None:
        p, a, h, v = _league(heterogeneous=True)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=1.0
        )
        steady = distribution.quantiles(12.0, 40.0, 1.0)
        streaky = distribution.quantiles(12.0, 40.0, 3.0)
        assert streaky["0.9"] - streaky["0.1"] > 2 * (steady["0.9"] - steady["0.1"])
        assert distribution.spread(12.0, 40.0, 3.0) > distribution.spread(12.0, 40.0, 1.0)
        assert distribution.confidence(12.0, 40.0, 3.0) < distribution.confidence(12.0, 40.0, 1.0)

    def test_probabilities_use_the_players_spread(self) -> None:
        p, a, h, v = _league(heterogeneous=True)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=1.0
        )
        # 18+ from a 12-point projection is a big night: likelier for the streaky one.
        steady = distribution.probability_at_least(12.0, 40.0, 17.5, 1.0)
        streaky = distribution.probability_at_least(12.0, 40.0, 17.5, 3.0)
        assert streaky > 3 * steady

    def test_round_trip_keeps_the_scale(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=2000)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=0.5
        )
        restored = ResidualDistribution.from_dict(distribution.to_dict())
        assert restored.quantiles(10.0, 30.0, 2.5) == distribution.quantiles(10.0, 30.0, 2.5)
        assert restored.volatility_of({"std10_points": 2.5}) == 2.5

    def test_a_distribution_saved_before_scaling_loads_unscaled(self) -> None:
        p, a, h, _ = _league(heterogeneous=True, n=1000)
        payload = fit_residual_distribution(p, a, h).to_dict()
        for key in ("volatility_feature", "scale_power", "volatility_reference"):
            payload.pop(key)
        restored = ResidualDistribution.from_dict(payload)
        assert not restored.scaled
        assert restored.volatility_of({"std10_points": 3.0}) is None


class TestChoice:
    def test_scaling_is_chosen_when_players_differ(self) -> None:
        p, a, h, v = _league(heterogeneous=True)
        choice = choose_volatility_scaling(p, a, h, {"std10_points": v})
        assert choice.volatility_feature == "std10_points"
        assert choice.scale_power > 0
        assert choice.scores[f"std10_points^{choice.scale_power}"] < choice.scores["unscaled"]

    def test_scaling_is_refused_when_they_do_not(self) -> None:
        p, a, h, v = _league(heterogeneous=False)
        choice = choose_volatility_scaling(p, a, h, {"std10_points": v})
        assert choice.volatility_feature is None
        assert choice.scale_power == 0.0

    def test_a_useless_source_loses_to_a_useful_one(self) -> None:
        p, a, h, v = _league(heterogeneous=True)
        rng = random.Random(3)
        noise: list[float | None] = [rng.uniform(1.0, 3.0) for _ in v]
        choice = choose_volatility_scaling(p, a, h, {"std10_minutes": noise, "std10_points": v})
        assert choice.volatility_feature == "std10_points"

    def test_too_few_rows_stays_unscaled(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=300)
        assert choose_volatility_scaling(p, a, h, {"std10_points": v}).volatility_feature is None

    def test_candidate_sources(self) -> None:
        assert volatility_features("points") == ("std10_points", "std10_minutes")
        assert volatility_features("minutes") == ("std10_minutes",)


class TestCoverage:
    def test_scaling_evens_out_coverage_across_volatility(self) -> None:
        p, a, h, v = _league(heterogeneous=True, n=12000)
        (fp, tp), (fa, ta), (fh, th), (fv, tv) = (_split(x) for x in (p, a, h, v))
        unscaled = fit_residual_distribution(fp, fa, fh)
        scaled = fit_residual_distribution(
            fp, fa, fh, fv, volatility_feature="std10_points", scale_power=1.0
        )
        report = coverage_report(
            scaled,
            tp,
            ta,
            th,
            volatilities=tv,
            group_volatilities=tv,
            target="POINTS",
            reference=unscaled,
        )
        before = report["reference_by_volatility"]
        after = report["by_volatility"]
        # Unscaled: far too wide for steady players, far too narrow for streaky.
        assert before["low"]["covered_80"] > 0.9
        assert before["high"]["covered_80"] < 0.72
        for group in ("low", "mid", "high"):
            assert after[group]["covered_80"] == pytest.approx(0.80, abs=0.03)
        assert report["threshold_brier"] < report["reference_threshold_brier"]
        assert quantile_loss(scaled, tp, ta, th, tv) < quantile_loss(unscaled, tp, ta, th)

    def test_report_without_extras_keeps_its_old_shape(self) -> None:
        p, a, h, _ = _league(heterogeneous=False, n=2000)
        distribution = fit_residual_distribution(p, a, h)
        report = coverage_report(distribution, p, a, h)
        assert set(report) == {
            "covered_80",
            "covered_50",
            "below_interval",
            "above_interval",
            "rows",
        }


class TestExpectedValue:
    def test_shifted_median_scales_the_mean_residual(self) -> None:
        rng = random.Random(5)
        p = [rng.uniform(4.0, 20.0) for _ in range(6000)]
        sigma = [rng.choice([1.0, 3.0]) for _ in p]
        # Right-skewed errors: the mean sits above the median, more so for big swings.
        a = [x + s * (rng.expovariate(1.0) - 0.69) for x, s in zip(p, sigma, strict=True)]
        h: list[float | None] = [40.0] * len(p)
        v: list[float | None] = list(sigma)
        distribution = fit_residual_distribution(
            p, a, h, v, volatility_feature="std10_points", scale_power=1.0
        )
        steady, streaky = shifted_median(
            [12.0, 12.0], None, distribution, histories=[40.0, 40.0], volatilities=[1.0, 3.0]
        )
        assert streaky - 12.0 > 1.5 * (steady - 12.0) > 0

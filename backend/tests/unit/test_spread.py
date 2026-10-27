"""Spread checks: slope, deciles and big nights, on worlds with a known answer."""

from __future__ import annotations

import pytest

from euroleague_ai.ml.spread import big_nights, calibration_line, deciles, spread_report


def _world(shrink: float, n: int = 20_000, seed: int = 0) -> tuple[list[float], list[float]]:
    """True means spread over 2-20; projections pulled toward the middle by ``shrink``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    mu = rng.uniform(2.0, 20.0, n)
    actual = rng.poisson(mu).astype(float)
    centre = float(mu.mean())
    projected = centre + shrink * (mu - centre)
    return [float(p) for p in projected], [float(a) for a in actual]


class TestSlope:
    @pytest.mark.parametrize(
        ("shrink", "verdict"),
        [(1.0, "about right"), (0.6, "too bunched"), (1.5, "too spread out")],
    )
    def test_verdicts(self, shrink: float, verdict: str) -> None:
        projected, actual = _world(shrink)
        report = spread_report(
            "POINTS",
            medians=projected,
            expected=projected,
            actual=actual,
            thresholds=[None] * len(actual),
        )
        assert report.slope == pytest.approx(1 / shrink, rel=0.05)
        assert report.verdict == verdict

    def test_line_is_exact_on_a_line(self) -> None:
        assert calibration_line([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == (
            pytest.approx(1.0),
            pytest.approx(2.0),
        )
        intercept, slope = calibration_line([2.0, 2.0], [1.0, 3.0])
        assert intercept != intercept and slope != slope  # NaN: no spread to regress on


class TestViews:
    def test_bunched_projections_show_in_the_top_and_bottom_deciles(self) -> None:
        projected, actual = _world(0.6)
        table = deciles(projected, actual)
        assert [row["decile"] for row in table] == list(range(1, 11))
        assert sum(row["rows"] for row in table) == len(actual)
        assert table[-1]["actual"] > table[-1]["projected"] + 1  # top scores more
        assert table[0]["actual"] < table[0]["projected"] - 1  # bottom less

    def test_big_nights_compare_implied_and_realised(self) -> None:
        result = big_nights(
            [{"15": 0.5, "20": 0.1}, {"15": 0.25, "20": 0.05}, None, {"15": 0.25, "20": 0.05}],
            [16.0, 21.0, 30.0, 3.0],
        )
        assert result == [
            {"bar": "15", "rows": 3, "implied": 1.0, "realised": 2, "ratio": 0.5},
            {"bar": "20", "rows": 3, "implied": 0.2, "realised": 1, "ratio": 0.2},
        ]

    def test_missing_expected_values_fall_back_to_the_median(self) -> None:
        report = spread_report(
            "THREE_POINTERS_MADE",
            medians=[1.0, 2.0, 3.0],
            expected=[None, None, None],
            actual=[1.0, 2.0, 3.0],
            thresholds=[None] * 3,
        )
        assert report.slope == pytest.approx(1.0)
        assert report.quantiles["expected"] == report.quantiles["median"]


class TestContinuityCorrection:
    """P(X+) for whole-number stats, fitted and judged like production."""

    def _world(self, n: int, seed: int) -> tuple[list[float], list[float]]:
        import numpy as np

        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.1, 2.5, n)  # threes
        actual = rng.negative_binomial(3, 3 / (3 + mu)).astype(float)
        projected = np.maximum(0, mu * rng.normal(1, 0.15, n) - 0.3)
        return [float(p) for p in projected], [float(a) for a in actual]

    def test_cutoff(self) -> None:
        from euroleague_ai.ml.distribution import threshold_cutoff

        assert threshold_cutoff("THREE_POINTERS_MADE", 2.0) == 1.5
        assert threshold_cutoff("POINTS", 15.0) == 14.5
        assert threshold_cutoff("MINUTES", 20.0) == 20.0

    def test_implied_big_nights_match_what_happens(self) -> None:
        from euroleague_ai.ml.distribution import fit_residual_distribution, threshold_cutoff

        fit_p, fit_a = self._world(30_000, seed=1)
        test_p, test_a = self._world(30_000, seed=2)
        distribution = fit_residual_distribution(fit_p, fit_a, [30.0] * len(fit_p))

        def implied(cutoff_for) -> list[dict]:
            thresholds = [
                {
                    str(k): distribution.probability_at_least(p, 30.0, cutoff_for(k))
                    for k in (1, 2, 3)
                }
                for p in test_p
            ]
            return big_nights(thresholds, test_a)

        corrected = implied(lambda k: threshold_cutoff("THREE_POINTERS_MADE", k))
        uncorrected = implied(float)
        assert all(abs(row["ratio"] - 1.0) < 0.04 for row in corrected), corrected
        # The bug this guards against: about 0.75x for threes.
        assert all(row["ratio"] < 0.85 for row in uncorrected), uncorrected


def test_served_threshold_probabilities_use_the_correction() -> None:
    from euroleague_ai.ml.distribution import ResidualDistribution
    from euroleague_ai.ml.inference import LoadedModel
    from euroleague_ai.models.enums import PredictionTarget

    residuals = [float(r) for r in range(-6, 7)]
    distribution = ResidualDistribution(level_edges=[], cells={}, by_history={}, pooled=residuals)

    def served(target: PredictionTarget) -> dict[str, float]:
        model = LoadedModel(
            model_version=None,
            booster=None,
            feature_names=[],
            calibration_offset=0.0,
            target=target,
            distribution=distribution,
        )
        return model.distribution_for(15.0, {})["threshold_probabilities"]

    points = served(PredictionTarget.POINTS)
    assert points["15"] == distribution.probability_at_least(15.0, None, 14.5)
    assert points["15"] > distribution.probability_at_least(15.0, None, 15.0)
    minutes = served(PredictionTarget.MINUTES)
    assert minutes["20"] == distribution.probability_at_least(15.0, None, 20.0)

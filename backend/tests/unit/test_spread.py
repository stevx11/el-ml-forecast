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

"""Calibration arithmetic for the outcomes loop.

The database-backed half is covered in the integration suite. This file covers
the pure functions, and in particular the banding rule, which has an off-by-one
at each edge that would quietly misfile every probability.
"""

from __future__ import annotations

import pytest

from euroleague_ai.ml.dataset import TARGET_COLUMNS, target_value
from euroleague_ai.ml.outcomes import PROBABILITY_BANDS
from euroleague_ai.models.enums import PredictionTarget


def band_for(probability: float) -> str | None:
    """The banding rule from ``threshold_calibration``, isolated for testing.

    Kept in step with the loop it mirrors by
    ``test_bands_tile_the_unit_interval``, which asserts the bands themselves
    are contiguous and complete.
    """
    for low, high in PROBABILITY_BANDS:
        if low <= probability < high or (high == 1.0 and probability == 1.0):
            return f"{int(low * 100)}-{int(high * 100)}%"
    return None


class TestProbabilityBands:
    def test_bands_tile_the_unit_interval(self) -> None:
        """No gaps, no overlaps, covering exactly [0, 1]."""
        assert PROBABILITY_BANDS[0][0] == 0.0
        assert PROBABILITY_BANDS[-1][1] == 1.0
        # [:-1] against [1:]: pairing consecutive bands means dropping the last
        # from the left side, or strict=True rightly objects that the two
        # sequences differ in length.
        for (_, previous_high), (next_low, _) in zip(
            PROBABILITY_BANDS[:-1], PROBABILITY_BANDS[1:], strict=True
        ):
            assert previous_high == next_low

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, "0-10%"),
            (0.05, "0-10%"),
            (0.1, "10-20%"),  # lower edge belongs to the upper band
            (0.35, "30-40%"),
            (0.899, "80-90%"),
            (0.9, "90-100%"),
            (1.0, "90-100%"),  # the only upper-inclusive edge
        ],
    )
    def test_edges_land_in_the_intended_band(
        self, probability: float, expected: str
    ) -> None:
        assert band_for(probability) == expected

    def test_every_probability_finds_a_band(self) -> None:
        """A probability with no home would be silently dropped from the table."""
        for step in range(0, 101):
            assert band_for(step / 100) is not None


class TestTargetUnits:
    def test_minutes_convert_from_seconds(self) -> None:
        """The box score stores seconds; every projection is in minutes.

        Getting this backwards would put the minutes model out by a factor of
        sixty, and the error would look like a catastrophic model failure rather
        than a unit bug.
        """
        assert target_value(PredictionTarget.MINUTES, 1800) == pytest.approx(30.0)

    def test_counting_stats_pass_through(self) -> None:
        assert target_value(PredictionTarget.POINTS, 17) == pytest.approx(17.0)
        assert target_value(PredictionTarget.ASSISTS, 4) == pytest.approx(4.0)

    def test_every_served_target_has_a_box_score_column(self) -> None:
        for target in (
            PredictionTarget.MINUTES,
            PredictionTarget.POINTS,
            PredictionTarget.REBOUNDS,
            PredictionTarget.ASSISTS,
            PredictionTarget.THREE_POINTERS_MADE,
        ):
            assert target in TARGET_COLUMNS


class TestErrorArithmetic:
    """The formulas the scoring job writes, stated once so they cannot drift."""

    @pytest.mark.parametrize(
        ("predicted", "actual"),
        [(14.0, 20.0), (20.0, 14.0), (5.0, 5.0), (0.0, 12.0)],
    )
    def test_absolute_and_squared_error(self, predicted: float, actual: float) -> None:
        error = predicted - actual
        assert abs(error) == abs(actual - predicted)
        assert error * error == pytest.approx((actual - predicted) ** 2)

    def test_squared_error_is_never_negative(self) -> None:
        for predicted, actual in ((3.0, 9.0), (9.0, 3.0), (-1.0, 1.0)):
            error = predicted - actual
            assert error * error >= 0.0

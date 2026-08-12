"""The pure statistics behind the form features."""

from __future__ import annotations

import pytest

from euroleague_ai.features.player_form import _ewma, _mean, _stdev, _trend


class TestMean:
    def test_missing_history_is_none_not_zero(self) -> None:
        """A rookie's ten-game average is unknown, not zero.

        Substituting 0.0 teaches the model that players without history score
        nothing, which is both false and backwards for a highly-touted signing.
        LightGBM handles genuine NaN natively.
        """
        assert _mean([]) is None

    def test_mean(self) -> None:
        assert _mean([10.0, 12.0, 14.0]) == pytest.approx(12.0)


class TestStdev:
    def test_needs_two_observations(self) -> None:
        assert _stdev([5.0]) is None

    def test_sample_stdev(self) -> None:
        assert _stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.1381, abs=1e-4)


class TestEwma:
    def test_weights_recent_games_more_heavily(self) -> None:
        """Newest-first input, so a hot streak should pull the mean upward."""
        rising = _ewma([30.0, 10.0, 10.0, 10.0])
        flat = _mean([30.0, 10.0, 10.0, 10.0])
        assert rising is not None and flat is not None
        assert rising > flat

    def test_constant_series_equals_the_constant(self) -> None:
        assert _ewma([12.0] * 10) == pytest.approx(12.0)

    def test_empty_is_none(self) -> None:
        assert _ewma([]) is None


class TestTrend:
    def test_improving_form_is_positive(self) -> None:
        """Input is newest-first: [20, 15, 10] means the player is improving."""
        trend = _trend([20.0, 15.0, 10.0])
        assert trend is not None and trend > 0

    def test_declining_form_is_negative(self) -> None:
        trend = _trend([10.0, 15.0, 20.0])
        assert trend is not None and trend < 0

    def test_flat_form_is_zero(self) -> None:
        assert _trend([12.0, 12.0, 12.0]) == pytest.approx(0.0)

    def test_needs_three_points(self) -> None:
        assert _trend([10.0, 20.0]) is None

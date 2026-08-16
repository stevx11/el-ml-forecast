"""Rate labelling and chain composition.

The training half needs LightGBM and a database, and is exercised by
``euroleague-ai ml chain``. What is tested here is the arithmetic that decides
whether the chain is sound at all: the rate label, the minutes weighting, and
the fact that multiplying two calibrated factors does not give a calibrated
product.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from euroleague_ai.ml.chain import (
    CHAINABLE,
    MIN_RATE_MINUTES,
    RATE_SCALE,
    rate_label,
)
from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.models.enums import PredictionTarget


def row(target: float, minutes: float) -> TrainingRow:
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return TrainingRow(
        snapshot_id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=uuid.uuid4(),
        season_code="E2025",
        round_number=5,
        tip_off_utc=moment,
        prediction_timestamp=moment,
        features={"ewma_points": 12.0},
        target=target,
        actual_minutes=minutes,
    )


class TestRateLabel:
    def test_a_full_game_rate_is_the_raw_total(self) -> None:
        """Forty minutes is the unit, so a forty-minute line converts 1:1."""
        assert rate_label(row(20.0, 40.0)) == pytest.approx(20.0)

    def test_half_a_game_doubles_the_rate(self) -> None:
        assert rate_label(row(10.0, 20.0)) == pytest.approx(20.0)

    def test_the_scale_is_per_forty(self) -> None:
        assert rate_label(row(1.0, 1.0 * RATE_SCALE)) == pytest.approx(1.0)

    def test_zero_production_is_a_zero_rate_not_a_dropped_row(self) -> None:
        """A scoreless 25 minutes is real information about the player."""
        assert rate_label(row(0.0, 25.0)) == pytest.approx(0.0)


class TestShortAppearancesAreExcluded:
    """The trap the whole design has to avoid.

    Two minutes and one basket is a rate of forty per forty -- a number that
    describes a rounding artefact, not a player. There are a lot of those rows,
    and left in they dominate the fit.
    """

    def test_below_the_floor_returns_none(self) -> None:
        assert rate_label(row(3.0, MIN_RATE_MINUTES - 0.1)) is None

    def test_the_pathological_case_is_excluded(self) -> None:
        # 2 minutes, one basket: 40 points per 40 minutes, from nothing.
        assert rate_label(row(2.0, 2.0)) is None

    def test_at_the_floor_the_row_is_kept(self) -> None:
        assert rate_label(row(2.0, MIN_RATE_MINUTES)) is not None

    def test_missing_minutes_are_unusable(self) -> None:
        assert rate_label(row(10.0, 0.0)) is None


class TestChainComposition:
    """Recombination arithmetic, stated once so it cannot drift."""

    def test_minutes_times_rate_recovers_the_total(self) -> None:
        minutes, rate = 30.0, 16.0  # 16 per 40
        assert minutes * rate / RATE_SCALE == pytest.approx(12.0)

    def test_errors_compound_multiplicatively(self) -> None:
        """The honest cost of the chain, pinned as a test.

        Ten percent high on each factor is twenty-one percent high on the
        product -- not twenty. This is exactly why the approach has to be
        measured rather than assumed.
        """
        true_minutes, true_rate = 30.0, 16.0
        truth = true_minutes * true_rate / RATE_SCALE
        inflated = (true_minutes * 1.1) * (true_rate * 1.1) / RATE_SCALE
        assert inflated / truth == pytest.approx(1.21)

    def test_a_calibrated_product_is_not_the_product_of_calibrated_factors(self) -> None:
        """Why the chain needs its own offset.

        E[XY] != E[X]E[Y] when X and Y are correlated, and minutes and scoring
        rate certainly are: players who get more minutes tend to be the ones
        producing more per minute.
        """
        minutes = [20.0, 25.0, 30.0, 35.0]
        rates = [10.0, 13.0, 16.0, 19.0]  # rises with minutes
        mean_minutes = sum(minutes) / len(minutes)
        mean_rate = sum(rates) / len(rates)

        mean_of_products = sum(m * r for m, r in zip(minutes, rates, strict=True)) / len(minutes)
        product_of_means = mean_minutes * mean_rate

        assert mean_of_products != pytest.approx(product_of_means)
        # Positively correlated, so the naive product understates.
        assert product_of_means < mean_of_products


class TestChainableTargets:
    def test_minutes_is_not_chainable(self) -> None:
        """Minutes is the multiplier; a rate of itself is meaningless."""
        assert PredictionTarget.MINUTES not in CHAINABLE

    def test_counting_stats_are_chainable(self) -> None:
        for target in (
            PredictionTarget.POINTS,
            PredictionTarget.REBOUNDS,
            PredictionTarget.ASSISTS,
            PredictionTarget.THREE_POINTERS_MADE,
        ):
            assert target in CHAINABLE


class TestMinutesWeighting:
    """A row should count for as much evidence as the minutes behind it."""

    def test_a_starter_outweighs_a_cameo(self) -> None:
        starter, cameo = row(18.0, 35.0), row(2.0, 5.0)
        assert starter.actual_minutes / cameo.actual_minutes == pytest.approx(7.0)

    def test_weighting_moves_the_estimate_toward_the_larger_sample(self) -> None:
        """Unweighted, a 6-minute fluke counts as much as a 36-minute line."""
        long_line, short_line = row(18.0, 36.0), row(6.0, 6.0)
        long_rate = rate_label(long_line)
        short_rate = rate_label(short_line)
        assert long_rate is not None and short_rate is not None

        # 20 per 40 from the substantial sample, 40 per 40 from the cameo.
        assert long_rate == pytest.approx(20.0)
        assert short_rate == pytest.approx(40.0)

        unweighted = (long_rate + short_rate) / 2
        weighted = (
            long_rate * long_line.actual_minutes + short_rate * short_line.actual_minutes
        ) / (long_line.actual_minutes + short_line.actual_minutes)

        assert unweighted == pytest.approx(30.0)  # the cameo counts equally
        assert weighted < unweighted  # weighting discounts it

    def test_minutes_weighting_reproduces_the_pooled_rate(self) -> None:
        """The property that makes minutes the *correct* weight, not just a
        plausible one.

        A minutes-weighted mean of per-40 rates is algebraically identical to
        computing one rate from the pooled totals -- 24 points in 42 minutes.
        Any other weighting would answer a different question.
        """
        long_line, short_line = row(18.0, 36.0), row(6.0, 6.0)
        rates = [rate_label(long_line), rate_label(short_line)]
        assert all(r is not None for r in rates)

        weighted = sum(
            r * line.actual_minutes
            for r, line in zip(rates, (long_line, short_line), strict=True)
        ) / (long_line.actual_minutes + short_line.actual_minutes)

        pooled = (
            (long_line.target + short_line.target)
            / (long_line.actual_minutes + short_line.actual_minutes)
            * RATE_SCALE
        )
        assert weighted == pytest.approx(pooled)
        assert weighted == pytest.approx(24.0 / 42.0 * 40.0)

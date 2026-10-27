"""When a fact became knowable.

The regression guard for the bug that made 66,000 snapshots entirely null.

Backfilling ten seasons stamped every row with the ingest wall-clock, so every
historical fact sat in the future relative to its own game. The feature queries
filtered on ``observed_at <= as_of``, found nothing, and returned null for
everything -- correct behaviour, wrong data.

The existing leakage suite could not catch this. It ingests with a clock fixed
*before* the fixture games, so ``observed_at`` was always in the past there. The
failure only appears when ingest time is later than the games, which is exactly
the real-world backfill case and exactly what these tests pin down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from euroleague_ai.ingestion.knowability import (
    BOX_SCORE_AVAILABLE_AFTER,
    FIXTURE_KNOWN_BEFORE,
    box_score_observed_at,
    fixture_observed_at,
    stint_observed_at,
)

# A 2018 game, backfilled in 2026. The shape of the real problem.
HISTORICAL_TIP_OFF = datetime(2018, 1, 10, 18, 0, tzinfo=UTC)
BACKFILL_INGEST = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)


class TestBoxScores:
    def test_historical_box_score_is_dated_from_the_game_not_the_ingest(self) -> None:
        """**The bug.** Stamping 2026 on a 2018 game hides it from 2018 features."""
        observed = box_score_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        assert observed == HISTORICAL_TIP_OFF + BOX_SCORE_AVAILABLE_AFTER
        assert observed < BACKFILL_INGEST

    def test_it_is_visible_to_a_later_cutoff_in_the_same_season(self) -> None:
        """The property that actually matters: a later game can see an earlier one."""
        observed = box_score_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        next_game_cutoff = HISTORICAL_TIP_OFF + timedelta(days=7)
        assert observed <= next_game_cutoff

    def test_it_is_invisible_to_a_cutoff_before_the_game(self) -> None:
        """Backdating must not go so far that it leaks."""
        observed = box_score_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        pre_game_cutoff = HISTORICAL_TIP_OFF - timedelta(hours=1)
        assert observed > pre_game_cutoff

    def test_a_game_still_in_progress_keeps_the_real_ingest_time(self) -> None:
        """We genuinely did not know the result yet -- do not invent knowledge."""
        mid_game = HISTORICAL_TIP_OFF + timedelta(hours=1)
        assert box_score_observed_at(HISTORICAL_TIP_OFF, mid_game) == mid_game

    def test_never_earlier_than_the_final_buzzer(self) -> None:
        observed = box_score_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        assert observed >= HISTORICAL_TIP_OFF + timedelta(hours=2)


class TestFixtures:
    def test_historical_fixture_is_dated_before_its_own_tip_off(self) -> None:
        """A schedule row must be visible to features built before the game."""
        observed = fixture_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        assert observed == HISTORICAL_TIP_OFF - FIXTURE_KNOWN_BEFORE
        assert observed < HISTORICAL_TIP_OFF

    def test_a_fixture_announced_in_advance_keeps_the_ingest_time(self) -> None:
        """Never claim to have known a fixture before it was published."""
        early = HISTORICAL_TIP_OFF - timedelta(days=120)
        assert fixture_observed_at(HISTORICAL_TIP_OFF, early) == early

    def test_the_pre_game_cutoff_can_see_the_fixture(self) -> None:
        """Otherwise the game being predicted is invisible to its own features."""
        observed = fixture_observed_at(HISTORICAL_TIP_OFF, BACKFILL_INGEST)
        cutoff = HISTORICAL_TIP_OFF - timedelta(minutes=60)
        assert observed <= cutoff


class TestStints:
    def test_registration_date_is_the_knowable_date(self) -> None:
        signing = datetime(2017, 9, 15, tzinfo=UTC)
        assert stint_observed_at(signing, BACKFILL_INGEST) == signing

    def test_a_future_registration_keeps_the_ingest_time(self) -> None:
        future = BACKFILL_INGEST + timedelta(days=30)
        assert stint_observed_at(future, BACKFILL_INGEST) == BACKFILL_INGEST


class TestTheInvariantThatWasViolated:
    """One property, stated plainly, that the whole dataset depends on."""

    @pytest.mark.parametrize(
        "tip_off",
        [
            datetime(2016, 10, 5, 18, 0, tzinfo=UTC),
            datetime(2020, 2, 20, 20, 30, tzinfo=UTC),
            datetime(2025, 10, 31, 18, 0, tzinfo=UTC),
        ],
    )
    def test_every_backfilled_game_is_knowable_before_the_next_one(self, tip_off: datetime) -> None:
        """A box score must be visible to any cutoff after the game ends.

        If this fails, form windows are empty, every feature is null, and the
        entire training set is unusable -- which is precisely what happened.
        """
        observed = box_score_observed_at(tip_off, BACKFILL_INGEST)
        # The next fixture is typically 2-7 days later; use the tightest case.
        next_cutoff = tip_off + timedelta(days=2)
        assert observed <= next_cutoff, (
            "backfilled box score is not visible to the next game's cutoff, "
            "so form features would be null"
        )

    def test_ingest_time_never_makes_history_invisible(self) -> None:
        """Whatever 'now' is, a 2018 game stays dated in 2018."""
        for years_later in (1, 5, 20):
            ingest = HISTORICAL_TIP_OFF + timedelta(days=365 * years_later)
            observed = box_score_observed_at(HISTORICAL_TIP_OFF, ingest)
            assert observed == HISTORICAL_TIP_OFF + BOX_SCORE_AVAILABLE_AFTER

"""Schedule arithmetic.

Scheduling bugs surface at midnight, at a rollover, or on the day two run times
collide -- none of which you find by waiting for them. The calculation is pure
so that every one of those cases is a test that runs in microseconds.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest

from euroleague_ai.scheduling.schedule import (
    DEFAULT_RUN_TIMES,
    next_run_at,
    parse_run_times,
    seconds_until,
)

MORNING = time(6, 0)
AFTERNOON = time(16, 0)
BOTH = (MORNING, AFTERNOON)


def at(hour: int, minute: int = 0, day: int = 15) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=UTC)


class TestParsing:
    def test_a_single_time(self) -> None:
        assert parse_run_times("06:00") == (time(6, 0),)

    def test_several_times_are_sorted(self) -> None:
        assert parse_run_times("16:00,06:00") == (time(6, 0), time(16, 0))

    def test_whitespace_and_duplicates_are_tolerated(self) -> None:
        assert parse_run_times(" 06:00 , 06:00,  16:00 ") == (time(6, 0), time(16, 0))

    @pytest.mark.parametrize(
        "spec", ["", "   ", "6", "06:00:00", "24:00", "06:60", "-1:00", "ab:cd", ","]
    )
    def test_bad_input_raises_rather_than_defaulting(self, spec: str) -> None:
        """A schedule that silently falls back is worse than one that refuses.

        A misconfigured job that defaults looks healthy and runs at the wrong
        time indefinitely; one that refuses to start gets noticed immediately.
        """
        with pytest.raises(ValueError):
            parse_run_times(spec)

    def test_midnight_is_valid(self) -> None:
        assert parse_run_times("00:00") == (time(0, 0),)

    def test_the_default_parses(self) -> None:
        assert parse_run_times("06:00,16:00") == DEFAULT_RUN_TIMES


class TestNextRun:
    def test_picks_the_next_slot_today(self) -> None:
        assert next_run_at(at(5, 0), BOTH) == at(6, 0)
        assert next_run_at(at(7, 0), BOTH) == at(16, 0)

    def test_rolls_over_to_tomorrow_after_the_last_slot(self) -> None:
        assert next_run_at(at(17, 0), BOTH) == at(6, 0, day=16)

    def test_late_evening_rolls_to_the_earliest_slot(self) -> None:
        assert next_run_at(at(23, 59), BOTH) == at(6, 0, day=16)

    def test_exactly_on_a_run_time_moves_to_the_next_one(self) -> None:
        """Strictly-after, or the loop fires, finishes in a second, and refires.

        Returning ``now`` would turn the schedule into a tight loop for the
        remainder of that minute.
        """
        assert next_run_at(at(6, 0), BOTH) == at(16, 0)
        assert next_run_at(at(16, 0), BOTH) == at(6, 0, day=16)

    def test_a_second_past_a_slot_waits_for_the_following_one(self) -> None:
        just_after = at(6, 0) + timedelta(seconds=1)
        assert next_run_at(just_after, BOTH) == at(16, 0)

    def test_a_single_daily_slot_always_advances(self) -> None:
        assert next_run_at(at(6, 0), (MORNING,)) == at(6, 0, day=16)
        assert next_run_at(at(5, 59), (MORNING,)) == at(6, 0)

    def test_month_boundary(self) -> None:
        end_of_month = datetime(2026, 9, 30, 23, 0, tzinfo=UTC)
        assert next_run_at(end_of_month, (MORNING,)) == datetime(2026, 10, 1, 6, 0, tzinfo=UTC)

    def test_year_boundary(self) -> None:
        new_years_eve = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
        assert next_run_at(new_years_eve, (MORNING,)) == datetime(2027, 1, 1, 6, 0, tzinfo=UTC)

    def test_unordered_input_still_works(self) -> None:
        """Callers should not have to sort; the function does."""
        assert next_run_at(at(5, 0), (AFTERNOON, MORNING)) == at(6, 0)

    def test_naive_datetimes_are_rejected(self) -> None:
        """Guessing that a naive timestamp is UTC is still a guess."""
        with pytest.raises(ValueError):
            next_run_at(datetime(2026, 9, 15, 5, 0), BOTH)  # noqa: DTZ001 - deliberately naive

    def test_no_run_times_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            next_run_at(at(5, 0), ())

    def test_the_result_is_always_in_the_future(self) -> None:
        for hour in range(24):
            for minute in (0, 1, 30, 59):
                now = at(hour, minute)
                assert next_run_at(now, BOTH) > now


class TestRunnerStatusConstants:
    """Regression: the runner once wrote a status member that did not exist.

    ``IngestionStatus.SUCCESS`` is not a member -- it is ``SUCCEEDED``. Because
    the reference sat on the last line of a long job, an entire pipeline run
    ingested, scored and predicted, then died with an AttributeError while
    recording that it had succeeded. Several minutes of real work to surface a
    typo.

    The members are now bound at import, so importing this module at all is the
    check. These assertions make the intent explicit and would fail loudly if
    someone rebound them to something else.
    """

    def test_the_runner_binds_real_enum_members(self) -> None:
        from euroleague_ai.models.enums import IngestionStatus
        from euroleague_ai.scheduling import runner

        assert runner.STATUS_RUNNING is IngestionStatus.RUNNING
        assert runner.STATUS_SUCCEEDED is IngestionStatus.SUCCEEDED
        assert runner.STATUS_FAILED is IngestionStatus.FAILED

    def test_there_is_no_success_member_to_reach_for(self) -> None:
        """Pin the trap itself, so the name cannot quietly come back."""
        from euroleague_ai.models.enums import IngestionStatus

        assert not hasattr(IngestionStatus, "SUCCESS")

    def test_the_pipeline_lock_differs_from_the_backfill_lock(self) -> None:
        """Sharing a key would make a snapshot rebuild block every scheduled run."""
        from euroleague_ai.features.backfill import ADVISORY_LOCK_KEY as BACKFILL_KEY
        from euroleague_ai.scheduling.runner import ADVISORY_LOCK_KEY as PIPELINE_KEY

        assert PIPELINE_KEY != BACKFILL_KEY


class TestSecondsUntil:
    def test_counts_forward(self) -> None:
        assert seconds_until(at(5, 0), at(6, 0)) == pytest.approx(3600.0)

    def test_a_backwards_clock_clamps_to_zero(self) -> None:
        """NTP corrections and resumed laptops are ordinary events.

        A negative sleep raises, which would take the scheduler down for a
        perfectly routine reason.
        """
        assert seconds_until(at(7, 0), at(6, 0)) == 0.0

    def test_zero_when_already_there(self) -> None:
        assert seconds_until(at(6, 0), at(6, 0)) == 0.0

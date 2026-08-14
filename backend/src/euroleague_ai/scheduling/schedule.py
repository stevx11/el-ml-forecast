"""When the next run is due.

Pure time arithmetic, deliberately separated from the loop that sleeps on it.
Scheduling bugs are the kind that only appear at midnight, at a daylight-saving
boundary, or on the one day a month when two run times collide -- none of which
you want to discover by waiting. Isolating the calculation makes every one of
those cases a unit test.

**Everything here is UTC.** The project stores every timestamp as UTC and the
provider publishes fixtures in UTC, so introducing a local timezone purely for
scheduling would add a daylight-saving discontinuity to the one component whose
entire job is knowing what time it is. A run time of ``06:00`` means 06:00 UTC,
and that is the same instant all year.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from euroleague_ai.core.clock import ensure_utc

#: Parsed from ``PIPELINE_RUN_AT``. Several times a day is normal in season:
#: EuroLeague plays Tuesday through Friday with late tip-offs, so a morning run
#: catches last night's box scores and an evening run refreshes projections
#: after the day's team news.
DEFAULT_RUN_TIMES: tuple[time, ...] = (time(6, 0), time(16, 0))


def parse_run_times(spec: str) -> tuple[time, ...]:
    """Parse ``"06:00,16:00"`` into times, sorted and deduplicated.

    Raises:
        ValueError: on anything unparseable. A schedule that silently falls back
            to a default when misconfigured is worse than one that refuses to
            start: the job appears healthy and runs at the wrong time forever.
    """
    values: set[time] = set()
    for chunk in spec.split(","):
        text = chunk.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"run time must look like HH:MM, got {text!r}")
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"run time must look like HH:MM, got {text!r}") from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"run time out of range: {text!r}")
        values.add(time(hour, minute))

    if not values:
        raise ValueError(f"no run times parsed from {spec!r}")
    return tuple(sorted(values))


def next_run_at(now: datetime, run_times: tuple[time, ...]) -> datetime:
    """The next scheduled instant strictly after ``now``.

    Strictly after, never equal: returning ``now`` when the clock lands exactly
    on a run time would make the loop fire, complete in under a second, and fire
    again -- a tight loop rather than a schedule.
    """
    if not run_times:
        raise ValueError("run_times must not be empty")

    now = ensure_utc(now, field="now")
    today = now.date()

    for candidate_time in sorted(run_times):
        candidate = datetime.combine(today, candidate_time, tzinfo=now.tzinfo)
        if candidate > now:
            return candidate

    # Every slot today has passed; the next one is the earliest tomorrow.
    return datetime.combine(today + timedelta(days=1), min(run_times), tzinfo=now.tzinfo)


def seconds_until(now: datetime, target: datetime) -> float:
    """Non-negative seconds from ``now`` to ``target``.

    Clamped at zero. A negative sleep would raise, and a clock stepping backwards
    -- NTP correction, a suspended laptop -- is a perfectly ordinary event that
    should not take the scheduler down.
    """
    delta = (ensure_utc(target, field="target") - ensure_utc(now, field="now")).total_seconds()
    return max(0.0, delta)

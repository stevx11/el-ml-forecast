"""Time utilities.

Two rules hold throughout this codebase:

1. Every datetime is timezone-aware and in UTC. Naive datetimes are rejected at
   the boundary rather than silently coerced, because a naive timestamp compared
   against an aware one is exactly the kind of bug that produces a subtle leak.
2. "Now" is obtained through a :class:`Clock`, never ``datetime.now()`` scattered
   through business logic, so tests can pin the ingestion timestamp and assert on
   ``observed_at`` values deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime, *, field: str = "datetime") -> datetime:
    """Return ``value`` normalised to UTC, rejecting naive datetimes.

    Raises:
        ValueError: if ``value`` has no timezone information. Assuming a naive
            timestamp is UTC is a guess, and guessing about time in a system whose
            correctness depends on time ordering is not acceptable.
    """
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware; got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


class Clock(Protocol):
    """Source of the current time.

    Injected wherever ``observed_at`` is stamped so ingestion runs are reproducible.
    """

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        ...


class SystemClock:
    """The real clock. Used everywhere outside tests."""

    __slots__ = ()

    def now(self) -> datetime:
        return utcnow()


class FixedClock:
    """A clock frozen at a fixed instant. Test support."""

    __slots__ = ("_instant",)

    def __init__(self, instant: datetime) -> None:
        self._instant = ensure_utc(instant, field="instant")

    def now(self) -> datetime:
        return self._instant

"""Domain enumerations.

These are created as native PostgreSQL enum types. A bad value therefore fails at
the database boundary rather than surfacing three weeks later as an inexplicable
model input.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    """Entity kinds that can carry an external provider reference."""

    LEAGUE = "LEAGUE"
    SEASON = "SEASON"
    TEAM = "TEAM"
    PLAYER = "PLAYER"
    COACH = "COACH"
    GAME = "GAME"


class GameStatus(StrEnum):
    SCHEDULED = "SCHEDULED"
    LIVE = "LIVE"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"


class TeamSide(StrEnum):
    HOME = "HOME"
    AWAY = "AWAY"


class PlayerPosition(StrEnum):
    GUARD = "GUARD"
    FORWARD = "FORWARD"
    CENTER = "CENTER"
    UNKNOWN = "UNKNOWN"


class AvailabilityStatus(StrEnum):
    """Ordered loosely from most to least available.

    ``UNKNOWN`` is distinct from ``AVAILABLE``: absence of evidence is not
    evidence of availability, and a feature that conflates the two would treat a
    silent roster as a healthy one.
    """

    AVAILABLE = "AVAILABLE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class StatusSourceType(StrEnum):
    """Where an availability observation came from.

    Drives trust: provider data is authoritative, news-derived data carries a
    confidence score and can be thresholded out of the feature pipeline.
    """

    PROVIDER = "PROVIDER"
    NEWS = "NEWS"
    MANUAL = "MANUAL"


class NewsEventType(StrEnum):
    """Structured basketball events extracted from news text.

    Deliberately about *what happened*, not about tone. Generic sentiment is not
    a useful signal for projecting a stat line; "minutes will be monitored" is.
    """

    INJURY = "INJURY"
    AVAILABILITY_CHANGE = "AVAILABILITY_CHANGE"
    MINUTES_LIMIT = "MINUTES_LIMIT"
    ROLE_CHANGE = "ROLE_CHANGE"
    STARTER_CHANGE = "STARTER_CHANGE"
    TRANSFER = "TRANSFER"
    COACH_COMMENT = "COACH_COMMENT"
    COACH_CHANGE = "COACH_CHANGE"
    ROTATION_CHANGE = "ROTATION_CHANGE"
    RETURN_FROM_INJURY = "RETURN_FROM_INJURY"
    OTHER = "OTHER"


class PredictionTarget(StrEnum):
    """Statistical quantities the system predicts.

    Adding a target is a new member here plus a trained model. No schema change:
    predictions are keyed by target rather than stored in per-target columns.
    """

    MINUTES = "MINUTES"
    POINTS = "POINTS"
    REBOUNDS = "REBOUNDS"
    ASSISTS = "ASSISTS"
    THREE_POINTERS_MADE = "THREE_POINTERS_MADE"
    STEALS = "STEALS"
    BLOCKS = "BLOCKS"
    TURNOVERS = "TURNOVERS"
    FIELD_GOALS_MADE = "FIELD_GOALS_MADE"
    FIELD_GOALS_ATTEMPTED = "FIELD_GOALS_ATTEMPTED"
    THREE_POINTERS_ATTEMPTED = "THREE_POINTERS_ATTEMPTED"
    FREE_THROWS_MADE = "FREE_THROWS_MADE"
    FREE_THROWS_ATTEMPTED = "FREE_THROWS_ATTEMPTED"
    VALUATION = "VALUATION"


class IngestionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

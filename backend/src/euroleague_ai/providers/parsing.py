"""Pure parsing primitives for provider payloads.

Standard library only, no I/O, no framework types. That is deliberate: these are
the functions most likely to be subtly wrong (padded identifiers, clock strings,
naive timestamps), and keeping them dependency-free makes them trivial to test in
isolation.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

_PERSON_CODE_RE = re.compile(r"^P?0*(\d+)$")
_CLOCK_RE = re.compile(r"^(\d{1,3}):([0-5]\d)$")


def normalise_person_code(raw: str | None) -> str | None:
    """Normalise a EuroLeague person/club code to a stable form.

    The upstream API is inconsistent about this. The legacy boxscore endpoint
    returns space-padded, prefixed codes (``"P007200   "``) while the v2
    endpoints return bare zero-padded ones (``"006590"``). Both refer to the same
    person, so both must resolve to one identifier or the same player is ingested
    twice.

    Normal form is six digits, zero-padded, with no prefix.

    >>> normalise_person_code("P007200   ")
    '007200'
    >>> normalise_person_code("006590")
    '006590'
    >>> normalise_person_code("  ")

    Returns ``None`` for blank input. Codes that are not purely numeric are
    returned uppercased and stripped rather than discarded -- some feeds use
    alphanumeric coach codes such as ``"TFT"``, which are still usable keys.
    """
    if raw is None:
        return None
    candidate = raw.strip().upper()
    if not candidate:
        return None
    if match := _PERSON_CODE_RE.match(candidate):
        return match.group(1).zfill(6)
    return candidate


def parse_clock_to_seconds(raw: str | float | int | None) -> int:
    """Convert a played-time value to whole seconds.

    Handles all three shapes the upstream API produces:

    * ``"33:21"`` -- legacy boxscore clock string
    * ``1850.0``  -- v2 ``timePlayed``, already in seconds
    * ``"DNP"`` / ``None`` / ``""`` -- did not play

    >>> parse_clock_to_seconds("33:21")
    2001
    >>> parse_clock_to_seconds(1850.0)
    1850
    >>> parse_clock_to_seconds("DNP")
    0
    """
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return max(0, round(float(raw)))

    candidate = raw.strip()
    if not candidate or candidate.upper() in {"DNP", "-", "--"}:
        return 0
    if match := _CLOCK_RE.match(candidate):
        return int(match.group(1)) * 60 + int(match.group(2))
    # Some feeds emit a bare number of seconds as a string.
    try:
        return max(0, round(float(candidate)))
    except ValueError:
        return 0


def is_did_not_play(raw_minutes: str | float | int | None, seconds_played: int) -> bool:
    """Decide whether a box score line represents a DNP.

    ``seconds_played == 0`` alone is ambiguous -- it covers both "did not dress"
    and "played eleven seconds of garbage time that rounded to zero". An explicit
    ``"DNP"`` marker is authoritative when present; otherwise zero seconds is
    treated as a DNP, which is correct in practice because EuroLeague box scores
    do not list players who appeared for less than a second.

    The distinction matters: minutes models exclude DNPs from form windows rather
    than averaging them in as zeros, and conflating the two teaches the model that
    an injured star is a low-usage player.
    """
    if isinstance(raw_minutes, str) and raw_minutes.strip().upper() == "DNP":
        return True
    return seconds_played == 0


def assume_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one.

    The EuroLeague API returns naive ISO strings for fields it documents as UTC
    (``utcDate``, roster ``startDate``). Interpreting them requires an
    assumption, so the assumption gets a loud name and lives in exactly one
    place, rather than being made implicitly at a dozen call sites.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime(raw: str | None, *, as_utc: bool = True) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating the variants this API emits.

    Accepts trailing ``Z``, fractional seconds of any length, and bare naive
    forms. Returns ``None`` rather than raising for unparseable input, because
    one malformed timestamp in a 340-game season should not abort a backfill --
    the row is skipped and counted instead.

    >>> parse_datetime("2025-10-01T18:30:00Z").isoformat()
    '2025-10-01T18:30:00+00:00'
    >>> parse_datetime("2026-02-25T15:07:09.067").isoformat()
    '2026-02-25T15:07:09.067000+00:00'
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return assume_utc(parsed) if as_utc else parsed


def parse_date(raw: str | None) -> date | None:
    """Parse a date, tolerating a full timestamp in a date-shaped field.

    Birth dates arrive as ``"1992-02-23T00:00:00"``.

    >>> parse_date("1992-02-23T00:00:00")
    datetime.date(1992, 2, 23)
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return datetime.fromisoformat(candidate).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(candidate[:10])
    except ValueError:
        return None


def parse_int(raw: object) -> int:
    """Coerce a provider numeric to ``int``, defaulting to 0.

    The v2 stats endpoint types every counting stat as a float (``12.0``), and a
    handful of fields arrive as strings (``"2110"``). Both must land as integers.

    >>> parse_int(12.0), parse_int("2110"), parse_int(None), parse_int("")
    (12, 2110, 0, 0)
    """
    if raw is None or raw == "":
        return 0
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return round(float(raw))
    try:
        return round(float(str(raw).strip()))
    except (ValueError, TypeError):
        return 0


def parse_optional_int(raw: object) -> int | None:
    """Like :func:`parse_int`, but preserves the difference between 0 and absent.

    Used for plus/minus and attendance, where zero is a real value and missing
    data must not be silently recorded as zero.
    """
    if raw is None or raw == "":
        return None
    return parse_int(raw)


def estimate_possessions(*, fga: int, offensive_rebounds: int, turnovers: int, fta: int) -> float:
    """Standard possession estimate: ``FGA - OREB + TO + 0.44 * FTA``.

    The 0.44 coefficient approximates the share of free throw attempts that end a
    possession, accounting for and-ones and technical free throws. It is the
    conventional value in basketball analytics and is used here so that pace and
    efficiency ratings are comparable with published figures.

    >>> estimate_possessions(fga=66, offensive_rebounds=14, turnovers=14, fta=15)
    72.6
    """
    return round(fga - offensive_rebounds + turnovers + 0.44 * fta, 3)


def estimate_pace(possessions: float, *, minutes_played: float = 40.0) -> float:
    """Possessions per 40 minutes, normalising for overtime.

    A game that went to double overtime has more possessions but not a faster
    pace, and failing to normalise would make overtime teams look uptempo.

    >>> estimate_pace(80.0, minutes_played=50.0)
    64.0
    """
    if minutes_played <= 0:
        return 0.0
    return round(possessions * 40.0 / minutes_played, 3)

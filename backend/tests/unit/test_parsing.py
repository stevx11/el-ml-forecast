"""Parsing primitives, tested against the quirks the live API actually emits."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from euroleague_ai.providers.parsing import (
    assume_utc,
    estimate_pace,
    estimate_possessions,
    is_did_not_play,
    normalise_person_code,
    parse_clock_to_seconds,
    parse_date,
    parse_datetime,
    parse_int,
    parse_optional_int,
)


class TestNormalisePersonCode:
    def test_padded_and_bare_codes_converge(self) -> None:
        """The two endpoints disagree; both must resolve to one identity.

        The legacy boxscore returns "P007200   " and v2 returns "007200". If
        these did not converge, the same player would be ingested twice and every
        form average would be computed over half their games.
        """
        assert normalise_person_code("P007200   ") == normalise_person_code("007200")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("P007200   ", "007200"),
            ("006590", "006590"),
            ("6590", "006590"),
            ("  011201 ", "011201"),
            ("TFT", "TFT"),  # alphanumeric coach codes are kept as-is
            ("", None),
            ("   ", None),
            (None, None),
        ],
    )
    def test_cases(self, raw: str | None, expected: str | None) -> None:
        assert normalise_person_code(raw) == expected


class TestParseClock:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("33:21", 2001),
            ("07:41", 461),
            ("200:00", 12000),  # team totals for a full game
            (1850.0, 1850),
            (461, 461),
            ("DNP", 0),
            ("", 0),
            (None, 0),
            ("nonsense", 0),
        ],
    )
    def test_cases(self, raw: object, expected: int) -> None:
        assert parse_clock_to_seconds(raw) == expected  # type: ignore[arg-type]

    def test_v2_and_legacy_agree(self) -> None:
        """1850 seconds and "30:50" are the same duration in two encodings."""
        assert parse_clock_to_seconds(1850.0) == parse_clock_to_seconds("30:50")


class TestDidNotPlay:
    def test_explicit_marker_wins(self) -> None:
        assert is_did_not_play("DNP", 0) is True

    def test_zero_seconds_is_a_dnp(self) -> None:
        assert is_did_not_play(0.0, 0) is True

    def test_played_time_is_not_a_dnp(self) -> None:
        assert is_did_not_play("01:47", 107) is False


class TestDatetimes:
    def test_naive_is_treated_as_utc(self) -> None:
        """The API documents these fields as UTC but omits the offset."""
        parsed = parse_datetime("2025-09-16T06:35:18.346")
        assert parsed == datetime(2025, 9, 16, 6, 35, 18, 346000, tzinfo=UTC)

    def test_trailing_z(self) -> None:
        assert parse_datetime("2025-10-01T18:00:00Z") == datetime(2025, 10, 1, 18, 0, tzinfo=UTC)

    def test_unparseable_returns_none(self) -> None:
        """One bad timestamp must cost one row, not the whole backfill."""
        assert parse_datetime("not a date") is None
        assert parse_datetime(None) is None

    def test_assume_utc_is_idempotent(self) -> None:
        aware = datetime(2025, 1, 1, tzinfo=UTC)
        assert assume_utc(assume_utc(aware)) == aware

    def test_date_from_timestamp(self) -> None:
        assert parse_date("1992-02-23T00:00:00") == date(1992, 2, 23)


class TestNumbers:
    @pytest.mark.parametrize(
        ("raw", "expected"), [(12.0, 12), ("2110", 2110), (None, 0), ("", 0), (True, 1)]
    )
    def test_parse_int(self, raw: object, expected: int) -> None:
        assert parse_int(raw) == expected

    def test_optional_int_distinguishes_zero_from_absent(self) -> None:
        """Plus/minus of 0 is a real value; a missing plus/minus is not 0."""
        assert parse_optional_int(0) == 0
        assert parse_optional_int(None) is None
        assert parse_optional_int("") is None


class TestDerivedTeamStats:
    def test_possessions_matches_the_standard_estimator(self) -> None:
        """FGA - OREB + TO + 0.44 * FTA, on real Anadolu Efes totals."""
        assert estimate_possessions(
            fga=66, offensive_rebounds=14, turnovers=14, fta=15
        ) == pytest.approx(72.6)

    def test_pace_normalises_for_overtime(self) -> None:
        """More possessions over more minutes is not a faster pace."""
        regulation = estimate_pace(72.0, minutes_played=40.0)
        overtime = estimate_pace(90.0, minutes_played=50.0)
        assert regulation == pytest.approx(72.0)
        assert overtime == pytest.approx(72.0)

    def test_pace_handles_zero_minutes(self) -> None:
        assert estimate_pace(50.0, minutes_played=0.0) == 0.0

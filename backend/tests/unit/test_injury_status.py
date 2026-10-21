"""Injury report resolution, status words and name matching. No database."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from euroleague_ai.features.injury_status import resolve_status
from euroleague_ai.ingestion.injuries import (
    InjuryInputError,
    RosterPlayer,
    match_player,
    normalise,
    parse_status,
)
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.repositories.availability import StatusReport

GAME = uuid.uuid4()
OTHER_GAME = uuid.uuid4()
TIP = datetime(2026, 10, 30, 19, 0, tzinfo=UTC)


def _report(
    status: AvailabilityStatus,
    *,
    days_before: float,
    game: uuid.UUID | None = None,
    until: datetime | None = None,
) -> StatusReport:
    at = TIP - timedelta(days=days_before)
    return StatusReport(
        id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=game,
        status=status,
        reason="knee",
        expected_return=until,
        source=StatusSourceType.MANUAL,
        event_timestamp=at,
        observed_at=at,
    )


class TestResolve:
    def test_a_report_for_this_game_applies(self) -> None:
        resolved = resolve_status([_report(AvailabilityStatus.OUT, days_before=1, game=GAME)],
                                  game_id=GAME, tip_off=TIP, last_played=None)
        assert resolved is not None and resolved.rules_out and resolved.scope == "game"

    def test_a_report_for_another_game_does_not(self) -> None:
        assert resolve_status([_report(AvailabilityStatus.OUT, days_before=1, game=OTHER_GAME)],
                              game_id=GAME, tip_off=TIP, last_played=None) is None

    def test_until_applies_before_the_date_only(self) -> None:
        later = _report(AvailabilityStatus.OUT, days_before=5, until=TIP + timedelta(days=7))
        earlier = _report(AvailabilityStatus.OUT, days_before=5, until=TIP - timedelta(days=1))
        assert resolve_status([later], game_id=GAME, tip_off=TIP, last_played=None) is not None
        assert resolve_status([earlier], game_id=GAME, tip_off=TIP, last_played=None) is None

    def test_open_ended_is_retired_once_he_plays_again(self) -> None:
        out = _report(AvailabilityStatus.OUT, days_before=10)
        still_out = resolve_status([out], game_id=GAME, tip_off=TIP, last_played=TIP - timedelta(days=12))
        back = resolve_status([out], game_id=GAME, tip_off=TIP, last_played=TIP - timedelta(days=3))
        assert still_out is not None and still_out.scope == "open"
        assert back is None

    def test_a_game_report_is_not_retired_by_earlier_appearances(self) -> None:
        out = _report(AvailabilityStatus.OUT, days_before=1, game=GAME)
        assert resolve_status([out], game_id=GAME, tip_off=TIP,
                              last_played=TIP - timedelta(hours=2)) is not None

    def test_newest_wins_and_available_clears(self) -> None:
        out = _report(AvailabilityStatus.OUT, days_before=4)
        gtd = _report(AvailabilityStatus.QUESTIONABLE, days_before=2)
        fit = _report(AvailabilityStatus.AVAILABLE, days_before=1)
        upgraded = resolve_status([out, gtd], game_id=GAME, tip_off=TIP, last_played=None)
        assert upgraded is not None and upgraded.status is AvailabilityStatus.QUESTIONABLE
        assert not upgraded.rules_out
        assert resolve_status([out, gtd, fit], game_id=GAME, tip_off=TIP, last_played=None) is None

    def test_suspension_rules_out(self) -> None:
        resolved = resolve_status([_report(AvailabilityStatus.SUSPENDED, days_before=1, game=GAME)],
                                  game_id=GAME, tip_off=TIP, last_played=None)
        assert resolved is not None and resolved.rules_out

    def test_as_dict_is_json_ready(self) -> None:
        resolved = resolve_status(
            [_report(AvailabilityStatus.OUT, days_before=3, until=TIP + timedelta(days=10))],
            game_id=GAME, tip_off=TIP, last_played=None,
        )
        assert resolved is not None
        info = resolved.as_dict()
        assert info["status"] == "OUT" and info["scope"] == "until"
        assert info["until"] == (TIP + timedelta(days=10)).date().isoformat()


class TestStatusWords:
    @pytest.mark.parametrize(
        ("word", "status"),
        [("out", AvailabilityStatus.OUT), ("GTD", AvailabilityStatus.QUESTIONABLE),
         ("game time", AvailabilityStatus.QUESTIONABLE), ("back", AvailabilityStatus.AVAILABLE),
         ("Suspended", AvailabilityStatus.SUSPENDED), ("day_to_day", AvailabilityStatus.QUESTIONABLE)],
    )
    def test_aliases(self, word: str, status: AvailabilityStatus) -> None:
        assert parse_status(word) is status

    def test_unknown_word_lists_the_options(self) -> None:
        with pytest.raises(InjuryInputError, match="questionable"):
            parse_status("dunno")


def _roster() -> list[RosterPlayer]:
    def p(name: str, team: str, code: str) -> RosterPlayer:
        return RosterPlayer(uuid.uuid4(), name, uuid.uuid4(), team, code)

    return [
        p("PETROVIC, MARKO", "PAR", "011001"),
        p("PETROVIC, NIKOLA", "RED", "011099"),
        p("ŠARIĆ, DARIO", "ANA", "P004471"),
        p("JOVANOVIC, LUKA", "PAR", "011002"),
    ]


class TestMatching:
    def test_surname_only_when_unique(self) -> None:
        assert match_player(_roster(), "jovanovic").full_name == "JOVANOVIC, LUKA"

    def test_full_name_in_any_order(self) -> None:
        assert match_player(_roster(), "Marko Petrovic").team_code == "PAR"

    def test_accents_are_ignored(self) -> None:
        assert match_player(_roster(), "Saric").full_name == "ŠARIĆ, DARIO"
        assert normalise("ŠARIĆ, Dario") == "saric dario"

    def test_ambiguous_names_list_the_candidates(self) -> None:
        with pytest.raises(InjuryInputError, match="matches 2 players"):
            match_player(_roster(), "Petrovic")

    def test_team_narrows_it(self) -> None:
        assert match_player(_roster(), "Petrovic", team_code="red").full_name == "PETROVIC, NIKOLA"

    def test_code_and_prefix(self) -> None:
        assert match_player(_roster(), "011002").full_name == "JOVANOVIC, LUKA"
        assert match_player(_roster(), "P4471").full_name == "ŠARIĆ, DARIO"
        assert match_player(_roster(), "Jovan").full_name == "JOVANOVIC, LUKA"

    def test_nobody(self) -> None:
        with pytest.raises(InjuryInputError, match="no rostered player"):
            match_player(_roster(), "Doncic")


def _news(days_before: float) -> StatusReport:
    report = _report(AvailabilityStatus.OUT, days_before=days_before)
    return replace(report, source=StatusSourceType.NEWS, confidence=0.9)


class TestNewsExpiry:
    def test_an_open_news_report_lapses_after_eight_days(self) -> None:
        assert resolve_status([_news(7)], game_id=GAME, tip_off=TIP, last_played=None) is not None
        assert resolve_status([_news(9)], game_id=GAME, tip_off=TIP, last_played=None) is None

    def test_a_manual_open_report_does_not_lapse(self) -> None:
        old = _report(AvailabilityStatus.OUT, days_before=30)
        assert resolve_status([old], game_id=GAME, tip_off=TIP, last_played=None) is not None

    def test_a_dated_news_report_lasts_to_its_date(self) -> None:
        report = _news(20)
        dated = replace(report, expected_return=TIP + timedelta(days=5))
        assert resolve_status([dated], game_id=GAME, tip_off=TIP, last_played=None) is not None

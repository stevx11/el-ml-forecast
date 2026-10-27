"""Provider payload mapping, tested against the committed fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from euroleague_ai.models.enums import GameStatus, PlayerPosition
from euroleague_ai.providers.mapping import (
    map_game,
    map_game_stats,
    map_game_status,
    map_player_stint,
    map_position,
    map_team,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "euroleague"


def _load(*parts: str) -> dict:
    return json.loads(FIXTURES.joinpath(*parts).read_text(encoding="utf-8"))


class TestTeams:
    def test_uses_permanent_name_not_sponsored_name(self) -> None:
        """Sponsorship changes yearly; club identity does not.

        "Kosner Baskonia Vitoria-Gasteiz" is this season's sponsored name.
        Storing it on the club row would mean next season's rebrand orphans this
        season's statistics.
        """
        rows = _load("clubs.json")["data"]
        baskonia = next(row for row in rows if row["code"] == "BAS")
        team = map_team(baskonia)
        assert team is not None
        assert team.name == "Saski Baskonia"
        assert team.short_name == "Baskonia"

    def test_rejects_rows_without_a_code(self) -> None:
        assert map_team({"name": "Nameless"}) is None


class TestPlayers:
    def test_maps_biography_and_interval(self) -> None:
        rows = _load("people.json")["data"]
        stint = map_player_stint(rows[0])
        assert stint is not None
        assert stint.external_id == "011001"
        assert stint.last_name == "PETROVIC"
        assert stint.first_name == "MARKO"
        assert stint.position is PlayerPosition.GUARD
        assert stint.team_external_id == "PAR"
        assert stint.effective_from.tzinfo is not None

    def test_transfer_produces_two_disjoint_stints(self) -> None:
        """A mid-season move must not overwrite the earlier registration."""
        rows = _load("people.json")["data"]
        stints = [
            stint
            for row in rows
            if (stint := map_player_stint(row)) is not None and stint.external_id == "011003"
        ]
        assert len(stints) == 2

        first, second = sorted(stints, key=lambda item: item.effective_from)
        assert first.team_external_id == "PAR"
        assert second.team_external_id == "BAS"
        assert first.effective_until is not None
        assert first.effective_until <= second.effective_from

    def test_rejects_rows_without_a_person(self) -> None:
        assert map_player_stint({"club": {"code": "PAR"}}) is None


class TestPositions:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"position": 1}, PlayerPosition.GUARD),
            ({"position": 2}, PlayerPosition.FORWARD),
            ({"position": 3}, PlayerPosition.CENTER),
            ({"positionName": "Center"}, PlayerPosition.CENTER),
            ({"positionName": "Centre"}, PlayerPosition.CENTER),
            ({}, PlayerPosition.UNKNOWN),
        ],
    )
    def test_cases(self, payload: dict, expected: PlayerPosition) -> None:
        assert map_position(payload) is expected


class TestGames:
    def test_maps_a_finished_game(self) -> None:
        rows = _load("games.json")["data"]
        game = map_game(rows[0])
        assert game is not None
        assert game.game_code == 1
        assert game.status is GameStatus.FINISHED
        assert game.home_team_external_id == "PAR"
        assert game.away_team_external_id == "MAD"
        assert game.tip_off_utc == datetime(2025, 10, 3, 18, 0, tzinfo=UTC)
        assert game.home_score == 84

    def test_scheduled_game_has_no_scores(self) -> None:
        rows = _load("games.json")["data"]
        game = map_game(next(row for row in rows if row["gameCode"] == 5))
        assert game is not None
        assert game.status is GameStatus.SCHEDULED
        assert game.home_score is None

    def test_quarter_scores_are_normalised(self) -> None:
        """`partials1` .. `partials4` become "1" .. "4"."""
        rows = _load("games.json")["data"]
        game = map_game(rows[0])
        assert game is not None
        assert game.home_quarter_scores is not None
        assert sorted(game.home_quarter_scores) == ["1", "2", "3", "4"]
        assert sum(game.home_quarter_scores.values()) == 84

    def test_status_falls_back_to_the_played_flag(self) -> None:
        """`gameStatus` is undocumented upstream, so `played` is the backstop."""
        assert map_game_status({"played": True}) is GameStatus.FINISHED
        assert map_game_status({"played": False}) is GameStatus.SCHEDULED
        assert map_game_status({"gameStatus": "postponed"}) is GameStatus.POSTPONED

    def test_rejects_games_without_both_sides(self) -> None:
        assert map_game({"gameCode": 1, "local": {}}) is None


class TestGameStats:
    def test_maps_both_sides(self) -> None:
        stats = map_game_stats(1, _load("game_stats", "1.json"))
        assert stats is not None
        assert stats.home.team_external_id == "PAR"
        assert stats.away.team_external_id == "MAD"
        assert len(stats.home.players) == 3

    def test_seconds_not_minutes(self) -> None:
        stats = map_game_stats(1, _load("game_stats", "1.json"))
        assert stats is not None
        line = next(p for p in stats.home.players if p.player_external_id == "011001")
        assert line.seconds_played == 1800
        assert line.is_starter is True

    def test_corrects_the_upstream_misspelling(self) -> None:
        """Upstream sends `foulsCommited`; our schema spells it correctly."""
        stats = map_game_stats(1, _load("game_stats", "1.json"))
        assert stats is not None
        assert stats.home.players[0].fouls_committed == 2

    def test_float_counting_stats_become_integers(self) -> None:
        stats = map_game_stats(1, _load("game_stats", "1.json"))
        assert stats is not None
        line = stats.home.players[0]
        assert isinstance(line.points, int)
        assert line.points == 10

    def test_empty_payload_returns_none(self) -> None:
        """An unplayed game presents as a payload with no player rows."""
        assert map_game_stats(99, {"local": {}, "road": {}}) is None

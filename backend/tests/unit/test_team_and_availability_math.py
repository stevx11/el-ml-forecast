"""Pure maths behind the team, opponent and availability families.

No database. These test the parts most likely to be quietly wrong: possession
normalisation, and the rotation logic that decides whose absence matters.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from euroleague_ai.features.availability import (
    ROTATION_MINUTES_THRESHOLD,
    summarise_squad,
)
from euroleague_ai.features.team_context import summarise_team
from euroleague_ai.repositories.history import SquadAppearance, TeamGameLine

BASE = datetime(2025, 10, 1, 18, 0, tzinfo=UTC)


def _team_line(
    *,
    points: int,
    allowed: int,
    possessions: float = 72.0,
    pace: float = 72.0,
    is_home: bool = True,
    day_offset: int = 0,
) -> TeamGameLine:
    return TeamGameLine(
        game_id=uuid.uuid4(),
        tip_off_utc=BASE - timedelta(days=day_offset),
        opponent_team_id=uuid.uuid4(),
        is_home=is_home,
        points=points,
        points_allowed=allowed,
        possessions=possessions,
        pace=pace,
        offensive_rebounds=10,
        defensive_rebounds=25,
        turnovers=12,
        fg3a=25,
        fta=18,
    )


class TestTeamRatings:
    def test_ratings_are_per_100_possessions(self) -> None:
        """80 points on 72 possessions is 111.1, not 80."""
        lines = [_team_line(points=80, allowed=72, possessions=72.0)]
        features = summarise_team(lines, "team")
        assert features["team_off_rating"] == pytest.approx(111.1111, abs=1e-3)
        assert features["team_def_rating"] == pytest.approx(100.0)
        assert features["team_net_rating"] == pytest.approx(11.1111, abs=1e-3)

    def test_pace_normalisation_separates_speed_from_quality(self) -> None:
        """A fast team is not automatically a good team.

        Both teams below score 1.25 points per possession. Without dividing by
        possessions the faster one would look far better, and the model would
        learn that playing fast causes scoring.
        """
        slow = summarise_team([_team_line(points=75, allowed=70, possessions=60.0)], "team")
        fast = summarise_team([_team_line(points=100, allowed=93, possessions=80.0)], "team")
        assert slow["team_off_rating"] == pytest.approx(125.0)
        assert fast["team_off_rating"] == pytest.approx(125.0)

    def test_no_history_yields_none_not_zero(self) -> None:
        """An unknown rating must not look like a catastrophically bad one."""
        features = summarise_team([], "opp")
        assert features["opp_games_played"] == 0.0
        assert features["opp_off_rating"] is None
        assert features["opp_def_rating"] is None
        assert features["opp_pace"] is None

    def test_prefix_namespaces_the_families(self) -> None:
        """Team and opponent features must not collide in one flat vector."""
        team = summarise_team([_team_line(points=80, allowed=70)], "team")
        opponent = summarise_team([_team_line(points=80, allowed=70)], "opp")
        assert not set(team) & set(opponent)

    def test_win_rate_and_home_rate(self) -> None:
        lines = [
            _team_line(points=90, allowed=80, is_home=True, day_offset=1),
            _team_line(points=70, allowed=80, is_home=False, day_offset=2),
        ]
        features = summarise_team(lines, "team")
        assert features["team_win_rate"] == pytest.approx(0.5)
        assert features["team_home_rate"] == pytest.approx(0.5)

    def test_empty_and_populated_branches_agree_on_shape(self) -> None:
        """A team with no history must still produce the same columns."""
        assert set(summarise_team([], "team")) <= set(
            summarise_team([_team_line(points=80, allowed=70)], "team")
        )


def _appearance(
    game_id: uuid.UUID, player_id: uuid.UUID, minutes: float, *, day: int
) -> SquadAppearance:
    return SquadAppearance(
        game_id=game_id,
        tip_off_utc=BASE - timedelta(days=day),
        player_id=player_id,
        seconds_played=int(minutes * 60),
        did_not_play=minutes == 0,
        is_starter=minutes >= 25,
    )


class TestAvailabilityProxy:
    """The substitute for availability data we could never observe historically."""

    def test_absent_rotation_player_is_detected(self) -> None:
        """A 30-minute regular who missed the last game must show up as absent.

        This is the whole point of the family: the minutes that player usually
        absorbs are now available to everyone else, and that redistribution is
        the largest single driver of a teammate's line.
        """
        subject, regular, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        older, latest = uuid.uuid4(), uuid.uuid4()

        appearances = [
            # Most recent game: the regular is missing.
            _appearance(latest, subject, 28, day=1),
            _appearance(latest, other, 24, day=1),
            # Earlier game: everyone played.
            _appearance(older, subject, 27, day=8),
            _appearance(older, other, 22, day=8),
            _appearance(older, regular, 30, day=8),
        ]
        features = summarise_squad(appearances, player_id=subject)

        assert features["teammates_absent_last_game"] == 1.0
        assert features["teammate_minutes_absent"] == pytest.approx(30.0)
        assert features["top_teammate_absent"] == 1.0

    def test_deep_bench_absence_is_ignored(self) -> None:
        """Below the rotation threshold, absence carries no information.

        Deep bench players are in and out of the squad constantly, so counting
        them would swamp the signal with noise.
        """
        subject, fringe = uuid.uuid4(), uuid.uuid4()
        older, latest = uuid.uuid4(), uuid.uuid4()

        appearances = [
            _appearance(latest, subject, 28, day=1),
            _appearance(older, subject, 27, day=8),
            _appearance(older, fringe, ROTATION_MINUTES_THRESHOLD - 5, day=8),
        ]
        features = summarise_squad(appearances, player_id=subject)
        assert features["teammates_absent_last_game"] == 0.0
        assert features["teammate_minutes_absent"] == pytest.approx(0.0)

    def test_subject_is_excluded_from_teammate_aggregates(self) -> None:
        """A player is not their own absent teammate."""
        subject = uuid.uuid4()
        older, latest = uuid.uuid4(), uuid.uuid4()
        appearances = [
            _appearance(latest, uuid.uuid4(), 20, day=1),
            _appearance(older, subject, 30, day=8),
        ]
        features = summarise_squad(appearances, player_id=subject)
        assert features["teammates_absent_last_game"] == 0.0

    def test_tracks_the_players_own_participation(self) -> None:
        """Returning from an absence usually means restricted minutes."""
        subject = uuid.uuid4()
        older, latest = uuid.uuid4(), uuid.uuid4()
        appearances = [
            _appearance(latest, subject, 0, day=1),  # missed
            _appearance(older, subject, 30, day=8),
        ]
        features = summarise_squad(appearances, player_id=subject)
        assert features["player_played_last_game"] == 0.0
        assert features["player_missed_recent_games"] == 1.0

    def test_rotation_size(self) -> None:
        latest = uuid.uuid4()
        subject = uuid.uuid4()
        appearances = [_appearance(latest, subject, 20, day=1)] + [
            _appearance(latest, uuid.uuid4(), 15, day=1) for _ in range(8)
        ]
        features = summarise_squad(appearances, player_id=subject)
        assert features["rotation_size_last_game"] == 9.0

    def test_no_history_yields_the_same_shape(self) -> None:
        empty = summarise_squad([], player_id=uuid.uuid4())
        populated = summarise_squad(
            [_appearance(uuid.uuid4(), uuid.uuid4(), 20, day=1)], player_id=uuid.uuid4()
        )
        assert set(empty) == set(populated)
        assert empty["squad_games_observed"] == 0.0

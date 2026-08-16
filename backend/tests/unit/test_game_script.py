"""Game-script features: expected total, margin and blowout risk.

Added because the chain experiment measured minutes MAE at 4.43 against a mean
of roughly twenty played minutes, and nothing in the feature set knew whether a
game would be close. Closeness is what decides whether a starter plays the
fourth quarter.

The property most of these tests protect is that blowout risk is **unsigned**.
Benches empty at both ends of a rout -- the winning coach rests his rotation,
the losing one has nothing left to save it for -- so a signed margin would tell
the model the opposite thing about the losing side.
"""

from __future__ import annotations

import pytest

from euroleague_ai.features.team_context import (
    BLOWOUT_MARGIN,
    FALLBACK_POSSESSIONS,
    game_script_features,
)


def script(
    team_off: float | None = 110.0,
    team_def: float | None = 110.0,
    opp_off: float | None = 110.0,
    opp_def: float | None = 110.0,
    pace: float | None = 72.0,
) -> dict[str, float | None]:
    return game_script_features(
        {
            "team_off_rating": team_off,
            "team_def_rating": team_def,
            "opp_off_rating": opp_off,
            "opp_def_rating": opp_def,
            "matchup_expected_pace": pace,
        }
    )


class TestEvenMatchup:
    def test_equal_teams_produce_no_margin(self) -> None:
        assert script()["game_expected_margin"] == pytest.approx(0.0)

    def test_equal_teams_have_no_blowout_risk(self) -> None:
        assert script()["game_blowout_risk"] == pytest.approx(0.0)

    def test_the_total_is_a_plausible_basketball_score(self) -> None:
        """Two 110-rated teams at 72 possessions should land near 158."""
        total = script()["game_expected_total"]
        assert total is not None and 150.0 < total < 165.0


class TestMismatch:
    def test_the_stronger_side_is_the_favourite(self) -> None:
        result = script(team_off=120.0, team_def=100.0, opp_off=100.0, opp_def=120.0)
        assert result["game_is_favourite"] == 1.0
        assert result["game_expected_margin"] is not None
        assert result["game_expected_margin"] > 10.0

    def test_the_weaker_side_is_not(self) -> None:
        result = script(team_off=100.0, team_def=120.0, opp_off=120.0, opp_def=100.0)
        assert result["game_is_favourite"] == 0.0
        assert result["game_expected_margin"] is not None
        assert result["game_expected_margin"] < 0.0

    def test_the_margin_is_symmetric_between_the_two_sides(self) -> None:
        favourite = script(team_off=120.0, team_def=100.0, opp_off=100.0, opp_def=120.0)
        underdog = script(team_off=100.0, team_def=120.0, opp_off=120.0, opp_def=100.0)
        assert underdog["game_expected_margin"] == pytest.approx(
            -favourite["game_expected_margin"]
        )


class TestBlowoutRiskIsUnsigned:
    """The point of the whole feature."""

    def test_both_sides_of_a_rout_carry_the_same_risk(self) -> None:
        favourite = script(team_off=125.0, team_def=95.0, opp_off=95.0, opp_def=125.0)
        underdog = script(team_off=95.0, team_def=125.0, opp_off=125.0, opp_def=95.0)

        assert favourite["game_blowout_risk"] == pytest.approx(underdog["game_blowout_risk"])
        assert favourite["game_expected_abs_margin"] == pytest.approx(
            underdog["game_expected_abs_margin"]
        )
        # ...while the signed margin still distinguishes them.
        assert favourite["game_expected_margin"] != pytest.approx(
            underdog["game_expected_margin"]
        )

    def test_risk_saturates_rather_than_growing_without_bound(self) -> None:
        """A 40-point gap does not shorten a rotation twice as much as a 20.

        Beyond roughly fifteen points coaches are managing minutes rather than
        chasing a result, so the feature flattens instead of implying that a
        rout has twice the effect of a comfortable win.
        """
        big = script(team_off=140.0, team_def=90.0, opp_off=90.0, opp_def=140.0)
        assert big["game_blowout_risk"] == pytest.approx(1.0)

    def test_risk_is_bounded_to_the_unit_interval(self) -> None:
        for offence in (90.0, 110.0, 130.0, 160.0):
            risk = script(team_off=offence, opp_def=offence)["game_blowout_risk"]
            assert risk is not None and 0.0 <= risk <= 1.0

    def test_the_saturation_point_matches_the_constant(self) -> None:
        result = script()
        assert result["game_expected_abs_margin"] == pytest.approx(0.0)
        assert BLOWOUT_MARGIN == 15.0


class TestPace:
    def test_more_possessions_means_more_points_to_go_round(self) -> None:
        slow = script(pace=64.0)["game_expected_total"]
        fast = script(pace=80.0)["game_expected_total"]
        assert slow is not None and fast is not None
        assert fast > slow

    def test_pace_does_not_change_the_sign_of_the_margin(self) -> None:
        for pace in (60.0, 72.0, 85.0):
            result = script(team_off=120.0, opp_def=120.0, pace=pace)
            assert result["game_is_favourite"] == 1.0

    def test_missing_pace_falls_back_rather_than_failing(self) -> None:
        """Early in a season neither side has a measured pace yet."""
        result = script(pace=None)
        assert result["game_expected_total"] is not None
        assert FALLBACK_POSSESSIONS == 72.0


class TestMissingRatings:
    """Round one: no ratings exist, and a fabricated zero would be a lie."""

    def test_absent_ratings_give_absent_features(self) -> None:
        result = script(team_off=None, team_def=None, opp_off=None, opp_def=None, pace=None)
        assert all(value is None for value in result.values())

    def test_a_single_missing_rating_is_enough_to_abstain(self) -> None:
        result = script(opp_def=None)
        assert result["game_expected_margin"] is None

    def test_the_keys_are_always_present(self) -> None:
        """A missing feature must be NaN to the model, not an absent column."""
        full = script()
        empty = script(team_off=None, team_def=None, opp_off=None, opp_def=None)
        assert set(full) == set(empty)
        assert len(empty) == 7


class TestInternalConsistency:
    def test_the_two_sides_sum_to_the_total(self) -> None:
        result = script(team_off=118.0, team_def=104.0, opp_off=108.0, opp_def=112.0)
        assert result["game_expected_team_points"] + result[
            "game_expected_opp_points"
        ] == pytest.approx(result["game_expected_total"], abs=0.01)

    def test_the_two_sides_differ_by_the_margin(self) -> None:
        result = script(team_off=118.0, team_def=104.0, opp_off=108.0, opp_def=112.0)
        assert result["game_expected_team_points"] - result[
            "game_expected_opp_points"
        ] == pytest.approx(result["game_expected_margin"], abs=0.01)

    def test_abs_margin_is_the_magnitude_of_the_margin(self) -> None:
        result = script(team_off=100.0, team_def=120.0, opp_off=120.0, opp_def=100.0)
        assert result["game_expected_abs_margin"] == pytest.approx(
            abs(result["game_expected_margin"])
        )

    def test_a_good_defence_lowers_the_opponents_expected_points(self) -> None:
        """Sanity: the estimate must respond to defence, not only to offence."""
        leaky = script(team_def=125.0)["game_expected_opp_points"]
        stingy = script(team_def=95.0)["game_expected_opp_points"]
        assert leaky is not None and stingy is not None
        assert stingy < leaky

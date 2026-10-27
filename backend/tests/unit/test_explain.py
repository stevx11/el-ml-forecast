"""Explanation themes and driver phrasing.

The failure mode here is not a crash -- it is an explanation that reads
perfectly and attributes the wrong thing. A rule ordering mistake would report
an opponent's defensive rating as a change in the player's role, and nothing
downstream would notice. So most of these tests are about attribution.
"""

from __future__ import annotations

import pytest

from euroleague_ai.ml.explain import (
    MIN_IMPACT,
    TOP_DRIVERS,
    explain_prediction,
    phrase_for,
    render_drivers,
    theme_for,
    theme_totals,
)

# Every feature name the v3 builders actually emit, one per shape. If a builder
# adds a shape this list does not cover, `test_nothing_falls_through` is the one
# that should start failing.
FEATURES_BY_THEME: dict[str, tuple[str, ...]] = {
    "minutes": (
        "season_minutes_mean",
        "last3_minutes_mean",
        "last5_minutes_mean",
        "last10_minutes_mean",
        "last20_minutes_mean",
        "ewma_minutes",
        "std10_minutes",
        "trend10_minutes",
        "previous_game_minutes",
        "minutes_last_7d",
    ),
    "form": (
        "season_points_mean",
        "last5_points_mean",
        "last20_rebounds_mean",
        "ewma_points",
        "ewma_fga",
        "std10_assists",
        "trend10_rebounds",
        "season_fg3a_mean",
        "season_valuation_mean",
    ),
    "role": (
        "starter_last_game",
        "starter_rate_last5",
        "starter_rate_recent",
        "fga_per40",
        "fg3a_rate",
        "fta_rate",
    ),
    "schedule": ("days_rest", "is_short_rest", "games_last_7d", "games_last_14d"),
    "experience": (
        "games_played_season",
        "games_played_career_window",
        "career_window_is_current_season",
    ),
    "opponent": (
        "opp_def_rating",
        "opp_off_rating",
        "opp_pace_last5",
        "opp_win_rate",
        "opp_home_rate",
        "matchup_off_vs_def",
        "matchup_expected_pace",
    ),
    "team": (
        "team_off_rating",
        "team_def_rating",
        "team_pace",
        "team_win_rate",
        "team_home_rate",
        "is_home",
    ),
    "teammates": (
        "teammate_minutes_absent",
        "teammate_minutes_absent_share",
        "teammates_absent_last_game",
        "top_teammate_absent",
        "rotation_size_avg",
        "rotation_size_last_game",
        "player_missed_recent_games",
        "player_played_last_game",
        "squad_games_observed",
    ),
}

ALL_FEATURES = [name for names in FEATURES_BY_THEME.values() for name in names]


class TestAttribution:
    @pytest.mark.parametrize(
        ("feature", "expected"),
        [(feature, theme) for theme, features in FEATURES_BY_THEME.items() for feature in features],
    )
    def test_each_feature_lands_in_its_theme(self, feature: str, expected: str) -> None:
        assert theme_for(feature) == expected

    def test_team_and_opponent_rates_are_not_mistaken_for_role(self) -> None:
        """Regression: the role rule matches "_rate", and so do these.

        With the rules in the wrong order, a defensive rating was reported to
        the reader as a change in the player's role -- entirely plausible copy,
        entirely wrong.
        """
        assert theme_for("team_win_rate") == "team"
        assert theme_for("opp_win_rate") == "opponent"
        assert theme_for("team_home_rate") == "team"
        assert theme_for("fg3a_rate") == "role"

    def test_minutes_beats_the_general_form_rule(self) -> None:
        """``last5_minutes_mean`` matches both rules; minutes must win."""
        assert theme_for("last5_minutes_mean") == "minutes"
        assert theme_for("last5_points_mean") == "form"

    def test_teammate_features_are_not_team_features(self) -> None:
        assert theme_for("teammate_minutes_absent") == "teammates"
        assert theme_for("team_off_rating") == "team"

    def test_nothing_falls_through(self) -> None:
        unthemed = [f for f in ALL_FEATURES if theme_for(f) == "other"]
        assert not unthemed, unthemed

    def test_an_unrecognised_feature_is_surfaced_not_swallowed(self) -> None:
        """A future feature should be visibly "other", never silently dropped."""
        assert theme_for("some_new_feature_nobody_mapped") == "other"


class TestThemeTotals:
    def test_contributions_are_summed_within_a_theme(self) -> None:
        names = ["ewma_minutes", "last5_minutes_mean", "opp_def_rating"]
        assert theme_totals(names, [1.0, 0.5, -2.0]) == {
            "minutes": pytest.approx(1.5),
            "opponent": pytest.approx(-2.0),
        }

    def test_totals_preserve_the_overall_sum(self) -> None:
        """Additivity is the property that makes grouping legitimate at all."""
        contributions = [0.3 * (i % 7) - 0.9 for i in range(len(ALL_FEATURES))]
        totals = theme_totals(ALL_FEATURES, contributions)
        assert sum(totals.values()) == pytest.approx(sum(contributions))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            theme_totals(["a", "b"], [1.0])


class TestDrivers:
    def _explain(self, **impacts: float) -> dict:
        contributions = [impacts.get(name, 0.0) for name in ALL_FEATURES]
        return explain_prediction(
            ALL_FEATURES, contributions, base_value=5.0, calibration_offset=0.1
        )

    def test_the_briefs_example_phrasing_is_produced(self) -> None:
        """The spec's own example: more minutes, tougher defence."""
        result = self._explain(ewma_minutes=2.4, opp_def_rating=-0.9)
        rendered = render_drivers(result)
        assert "+ Increased projected minutes" in rendered
        assert "- Tough opponent defense" in rendered

    def test_drivers_are_ordered_by_magnitude_not_sign(self) -> None:
        result = self._explain(ewma_minutes=0.6, opp_def_rating=-2.5, ewma_points=1.2)
        assert [d["theme"] for d in result["drivers"]] == [
            "opponent",
            "form",
            "minutes",
        ]

    def test_at_most_three_drivers(self) -> None:
        result = self._explain(
            ewma_minutes=3.0,
            ewma_points=2.5,
            opp_def_rating=-2.0,
            team_off_rating=1.5,
            days_rest=1.0,
        )
        assert len(result["drivers"]) == TOP_DRIVERS

    def test_negligible_contributions_are_not_dressed_up_as_reasons(self) -> None:
        """Without a floor, rounding dust becomes a confident bullet point."""
        result = self._explain(ewma_minutes=3.0, opp_def_rating=-MIN_IMPACT / 10)
        assert [d["theme"] for d in result["drivers"]] == ["minutes"]

    def test_direction_matches_the_sign(self) -> None:
        result = self._explain(ewma_minutes=-2.0)
        driver = result["drivers"][0]
        assert driver["direction"] == "-"
        assert driver["label"] == "Reduced projected minutes"

    def test_every_theme_has_phrasing_in_both_directions(self) -> None:
        for theme in [*FEATURES_BY_THEME, "other"]:
            assert phrase_for(theme, 1.0)
            assert phrase_for(theme, -1.0)
            assert phrase_for(theme, 1.0) != phrase_for(theme, -1.0) or theme == "other"


class TestStoredPayload:
    def test_the_arithmetic_reconciles(self) -> None:
        """base + every contribution + offset is the served value.

        Recorded explicitly so a reader can check the explanation against the
        number rather than taking it on trust.
        """
        contributions = [0.1] * len(ALL_FEATURES)
        result = explain_prediction(
            ALL_FEATURES, contributions, base_value=4.0, calibration_offset=0.25
        )
        reconstructed = (
            result["base_value"] + sum(result["themes"].values()) + result["calibration_offset"]
        )
        assert reconstructed == pytest.approx(4.0 + 0.1 * len(ALL_FEATURES) + 0.25)

    def test_raw_contributions_are_kept_for_debugging(self) -> None:
        contributions = [0.0] * len(ALL_FEATURES)
        contributions[ALL_FEATURES.index("ewma_minutes")] = 9.9
        result = explain_prediction(ALL_FEATURES, contributions, base_value=1.0)
        assert result["top_features"][0] == ["ewma_minutes", 9.9]

    def test_payload_is_json_serialisable(self) -> None:
        import json

        result = explain_prediction(ALL_FEATURES, [0.2] * len(ALL_FEATURES), base_value=3.0)
        assert json.loads(json.dumps(result)) == result

    def test_render_handles_a_missing_explanation(self) -> None:
        """Predictions from older models have none, and must not crash the CLI."""
        assert render_drivers(None) == ""
        assert render_drivers({}) == ""

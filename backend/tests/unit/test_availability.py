"""The play-probability model: metrics, baselines, split, candidates, serving.

The metric tests use hand-computable cases on purpose. A gate is only as
trustworthy as the arithmetic behind it, and a log loss that is quietly off by a
sign or a clip would pass or fail models for reasons nobody could see.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from euroleague_ai.features.backfill import _PlayerGameTarget, merge_targets
from euroleague_ai.ml.availability import (
    PROBABILITY_EPSILON,
    AvailabilityRow,
    BaseRate,
    PlayedLastGame,
    RecentAppearanceRate,
    brier_score,
    chronological_split,
    classification_metrics,
    derive_feature_names,
    expected_calibration_error,
    log_loss,
    reliability_table,
    roc_auc,
    team_count_error,
    train_availability,
)
from euroleague_ai.ml.explain import PLAYS_PHRASES, explain_prediction, phrase_for
from euroleague_ai.ml.inference import LoadedModel
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.repositories.rosters import RecentAppearances, select_candidates

T0 = datetime(2024, 10, 1, 18, 0, tzinfo=UTC)


def _row(
    *,
    played: bool,
    game: uuid.UUID | None = None,
    team: uuid.UUID | None = None,
    day: int = 0,
    features: dict[str, Any] | None = None,
    recent_played: int = 5,
    recent_team_games: int = 5,
    dressed: bool | None = None,
    not_debuted: bool = False,
) -> AvailabilityRow:
    return AvailabilityRow(
        snapshot_id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=game or uuid.uuid4(),
        team_id=team or uuid.uuid4(),
        season_code="E2024",
        tip_off_utc=T0 + timedelta(days=day),
        prediction_timestamp=T0 + timedelta(days=day, hours=-1),
        features=features or {},
        played=played,
        dressed=played if dressed is None else dressed,
        recent_played=recent_played,
        recent_team_games=recent_team_games,
        not_debuted=not_debuted,
    )


class TestLogLoss:
    def test_matches_hand_computation(self) -> None:
        # -(log 0.8 + log(1 - 0.3)) / 2
        expected = -(math.log(0.8) + math.log(0.7)) / 2
        assert log_loss([0.8, 0.3], [1.0, 0.0]) == pytest.approx(expected)

    def test_certain_and_wrong_is_clipped_not_infinite(self) -> None:
        """One confident miss must not turn the average into infinity."""
        value = log_loss([1.0], [0.0])
        assert math.isfinite(value)
        assert value == pytest.approx(-math.log(PROBABILITY_EPSILON))

    def test_empty_is_nan(self) -> None:
        assert math.isnan(log_loss([], []))


class TestBrier:
    def test_matches_hand_computation(self) -> None:
        assert brier_score([0.8, 0.3], [1.0, 0.0]) == pytest.approx((0.04 + 0.09) / 2)


class TestAuc:
    def test_perfect_separation_is_one(self) -> None:
        assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_inverted_is_zero(self) -> None:
        assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)

    def test_all_tied_is_half(self) -> None:
        assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)

    def test_one_class_is_undefined(self) -> None:
        assert math.isnan(roc_auc([0.2, 0.9], [1, 1]))


class TestReliability:
    def test_bins_and_rates(self) -> None:
        table = reliability_table([0.05, 0.08, 0.95, 0.92, 0.91], [0, 1, 1, 1, 0])
        assert [row["band"] for row in table] == ["0-10%", "90-100%"]
        assert table[0]["rows"] == 2
        assert table[0]["realised"] == pytest.approx(0.5)
        assert table[1]["realised"] == pytest.approx(2 / 3, abs=1e-4)

    def test_probability_of_one_lands_in_the_top_bin(self) -> None:
        assert reliability_table([1.0], [1])[0]["band"] == "90-100%"

    def test_perfect_calibration_has_zero_ece(self) -> None:
        probabilities = [0.25] * 4 + [0.75] * 4
        labels = [1, 0, 0, 0, 1, 1, 1, 0]
        assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)


class TestTeamCount:
    def test_sums_probabilities_per_team_game(self) -> None:
        game, home, away = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        groups = [(game, home), (game, home), (game, away)]
        result = team_count_error([0.9, 0.6, 0.5], [1, 0, 1], groups)
        # home expected 1.5 vs 1 actual (+0.5); away 0.5 vs 1 (-0.5)
        assert result["team_games"] == 2
        assert result["mae"] == pytest.approx(0.5)
        assert result["bias"] == pytest.approx(0.0)

    def test_metrics_bundle_reports_it(self) -> None:
        game, team = uuid.uuid4(), uuid.uuid4()
        metrics = classification_metrics([0.9, 0.9], [1, 1], [(game, team), (game, team)])
        assert metrics.team_count_bias == pytest.approx(-0.2)
        assert metrics.as_dict()["auc"] is None  # one class: undefined, reported as null


class TestBaselines:
    def test_base_rate_is_the_training_rate(self) -> None:
        rows = [_row(played=True)] * 3 + [_row(played=False)]
        assert BaseRate().fit(rows).predict(_row(played=True)) == pytest.approx(0.75)

    def test_sparse_keys_shrink_toward_the_overall_rate(self) -> None:
        """Two rows of "played 0 of 5" must not become a confident 0%."""
        common = [_row(played=True, recent_played=5) for _ in range(100)]
        sparse = [_row(played=False, recent_played=0) for _ in range(2)]
        baseline = RecentAppearanceRate().fit(common + sparse)
        prediction = baseline.predict(_row(played=False, recent_played=0))
        assert 0.0 < prediction < 1.0
        assert prediction > 0.5  # pulled most of the way back to ~98%

    def test_unseen_key_falls_back_to_overall(self) -> None:
        rows = [_row(played=True, features={"player_played_last_game": 1.0})] * 4
        baseline = PlayedLastGame().fit(rows)
        unseen = _row(played=True, features={"player_played_last_game": 0.0})
        assert baseline.predict(unseen) == pytest.approx(1.0)

    def test_debut_is_its_own_key(self) -> None:
        signing = RecentAppearanceRate().key(_row(played=True, recent_played=0, not_debuted=True))
        fringe = RecentAppearanceRate().key(_row(played=True, recent_played=0))
        assert signing != fringe


class TestSplit:
    def test_cuts_land_on_game_boundaries(self) -> None:
        rows = []
        for day in range(40):
            game, team = uuid.uuid4(), uuid.uuid4()
            rows.extend(_row(played=True, game=game, team=team, day=day) for _ in range(7))
        train, validation, test = chronological_split(rows)

        assert len(train) + len(validation) + len(test) == len(rows)
        games = [{r.game_id for r in split} for split in (train, validation, test)]
        assert not games[0] & games[1]
        assert not games[1] & games[2]
        assert max(r.tip_off_utc for r in train) < min(r.tip_off_utc for r in validation)
        assert max(r.tip_off_utc for r in validation) < min(r.tip_off_utc for r in test)

    def test_too_few_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="not enough rows"):
            chronological_split([_row(played=True)])

    def test_metadata_never_becomes_a_feature(self) -> None:
        rows = [_row(played=True, features={"days_rest": 3.0, "round_number": 7.0})]
        assert derive_feature_names(rows) == ["days_rest"]


class TestTraining:
    def test_learns_a_real_signal_and_beats_the_base_rate(self) -> None:
        """Players who missed the last game sit out far more often. The model
        must find that and beat the constant baseline on log loss."""
        import random

        rng = random.Random(7)
        rows = []
        for day in range(400):
            game, team = uuid.uuid4(), uuid.uuid4()
            for _ in range(12):
                last = 1.0 if rng.random() < 0.85 else 0.0
                played = rng.random() < (0.95 if last else 0.35)
                rows.append(
                    _row(
                        played=played,
                        game=game,
                        team=team,
                        day=day,
                        features={"player_played_last_game": last, "noise": rng.random()},
                    )
                )

        result = train_availability(rows)

        assert result.metrics.log_loss < result.baselines["base_rate"].log_loss
        assert result.metrics.auc > 0.7
        assert result.feature_names == ["noise", "player_played_last_game"]
        summary = result.summary()
        assert summary["target"] == "PLAYS"
        assert set(summary["rows"]) == {"train", "validation", "test"}

    def test_single_class_training_is_refused(self) -> None:
        rows = [
            _row(played=True, day=day, features={"x": float(day)}) for day in range(100)
        ]
        with pytest.raises(ValueError, match="only one class"):
            train_availability(rows)


class TestCandidateSelection:
    def test_recent_players_and_unseen_signings_are_kept(self) -> None:
        regular, fringe, signing = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        appearances = RecentAppearances(
            team_games=5,
            played={regular: 5},
            debuted=frozenset({regular, fringe}),
        )
        chosen = select_candidates({regular, fringe, signing}, appearances)
        assert set(chosen) == {regular, signing}

    def test_before_any_games_the_whole_roster_is_kept(self) -> None:
        roster = {uuid.uuid4(), uuid.uuid4()}
        assert set(select_candidates(roster, RecentAppearances(team_games=0))) == roster


class TestMergeTargets:
    def test_box_score_side_wins_and_no_player_appears_twice(self) -> None:
        player, other = uuid.uuid4(), uuid.uuid4()
        home, away = uuid.uuid4(), uuid.uuid4()
        box = [_PlayerGameTarget(player, home, away, True)]
        candidates = [
            _PlayerGameTarget(player, away, home, False),
            _PlayerGameTarget(other, away, home, False),
        ]
        merged = merge_targets(box, candidates)
        assert [t.player_id for t in merged] == [player, other]
        assert merged[0].team_id == home


class TestPlaysExplanations:
    def test_blowouts_read_as_a_deeper_rotation(self) -> None:
        """The stat phrase for a negative game script says "shorter rotation";
        for P(plays) a positive one has to say the opposite."""
        assert phrase_for("game_script", 1.0, PLAYS_PHRASES) == "Game script favours a deep rotation"

    def test_log_odds_scale_and_threshold(self) -> None:
        explanation = explain_prediction(
            ["days_rest", "player_played_last_game"],
            [-0.08, 0.6],
            base_value=2.0,
            phrases=PLAYS_PHRASES,
            min_impact=0.1,
            scale="log_odds",
        )
        assert explanation["scale"] == "log_odds"
        labels = [driver["label"] for driver in explanation["drivers"]]
        assert labels == ["Recent appearances point to playing"]


class _FakeBooster:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def predict(self, x: Any, **_: Any) -> list[float]:
        return self._values[: len(x)]


class TestServingAProbability:
    def _model(self, target: PredictionTarget, values: list[float]) -> LoadedModel:
        return LoadedModel(
            model_version=None,  # type: ignore[arg-type]
            booster=_FakeBooster(values),
            feature_names=["x"],
            calibration_offset=0.7,
            target=target,
        )

    def test_probabilities_are_not_shifted_and_stay_in_unit_interval(self) -> None:
        model = self._model(PredictionTarget.PLAYS, [0.2, 1.3, -0.1])
        assert model.predict_batch([{"x": 1}, {"x": 2}, {"x": 3}]) == [0.2, 1.0, 0.0]

    def test_stat_models_keep_their_offset(self) -> None:
        model = self._model(PredictionTarget.POINTS, [5.0])
        assert model.predict_batch([{"x": 1}]) == [pytest.approx(5.7)]

    def test_a_probability_publishes_no_interval(self) -> None:
        model = self._model(PredictionTarget.PLAYS, [0.8])
        assert model.distribution_for(0.8, {}) == {}

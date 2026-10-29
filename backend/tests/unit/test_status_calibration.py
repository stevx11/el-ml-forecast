"""Per-label P(plays) offsets: the fit, the shrinkage and the round-by-round check."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from euroleague_ai.ml.status_calibration import (
    MIN_PLAYERS,
    LabelledPlayer,
    StatusCalibration,
    adjusted,
    calibrate,
    fit_offset,
    fit_offsets,
    log_loss,
)

NOW = datetime(2026, 10, 29, tzinfo=UTC)


def _round(
    key: str, group: str, model_p: float, true_p: float, n: int, rng: random.Random
) -> list[LabelledPlayer]:
    return [LabelledPlayer(group, model_p, rng.random() < true_p, key) for _ in range(n)]


class TestFit:
    def test_offset_moves_toward_what_happened_but_is_shrunk(self) -> None:
        # Round 8's probable players: P 0.33 from the model, 9 of 13 played.
        ps = [0.33] * 13
        played = [True] * 9 + [False] * 4
        offset = fit_offset(ps, played, prior_sd=1.0)
        assert 0.33 < adjusted(0.33, offset) < 9 / 13
        # With no prior it lands on the observed rate.
        unshrunk = fit_offset(ps, played, prior_sd=100.0)
        assert adjusted(0.33, unshrunk) == pytest.approx(9 / 13, abs=0.005)

    def test_a_calibrated_label_gets_no_offset(self) -> None:
        rng = random.Random(1)
        players = _round("E:1", "probable", 0.6, 0.6, 2000, rng)
        assert fit_offsets(players)["probable"] == pytest.approx(0.0, abs=0.12)

    def test_too_few_players_are_left_alone(self) -> None:
        rng = random.Random(2)
        players = _round("E:1", "doubtful", 0.5, 0.0, MIN_PLAYERS - 1, rng)
        assert "doubtful" not in fit_offsets(players)

    def test_ruled_out_and_late_reports_are_never_fitted(self) -> None:
        rng = random.Random(3)
        players = _round("E:1", "ruled out", 0.0, 0.0, 50, rng) + _round(
            "E:1", "late report", 0.8, 0.2, 50, rng
        )
        assert fit_offsets(players) == {}

    def test_zero_stays_zero(self) -> None:
        assert adjusted(0.0, 3.0) == 0.0


class TestCheck:
    def test_a_real_bias_is_switched_on(self) -> None:
        rng = random.Random(4)
        players = []
        for key in ("E:7", "E:8", "E:9"):
            players += _round(key, "probable", 0.33, 0.7, 15, rng)
            players += _round(key, "questionable", 0.2, 0.32, 30, rng)
        fit = calibrate(players, now=NOW)
        assert fit.enabled
        assert fit.offsets["probable"] > 0.8
        assert fit.held_out["adjusted"] < fit.held_out["model"]
        assert fit.labels["probable"]["mean_adjusted_p"] > fit.labels["probable"]["mean_model_p"]
        assert fit.rounds == ["E:7", "E:8", "E:9"]

    def test_no_bias_stays_off(self) -> None:
        rng = random.Random(5)
        players = []
        for key in ("E:7", "E:8"):
            players += _round(key, "questionable", 0.4, 0.4, 25, rng)
        fit = calibrate(players, now=NOW)
        # Offsets near zero; adjusting can only add noise held out.
        assert abs(fit.offsets["questionable"]) < 0.5

    def test_one_round_cannot_be_checked(self) -> None:
        rng = random.Random(6)
        fit = calibrate(_round("E:8", "probable", 0.33, 0.7, 20, rng), now=NOW)
        assert not fit.enabled and fit.held_out == {}

    def test_log_loss_with_and_without_offsets(self) -> None:
        players = [LabelledPlayer("probable", 0.3, True, "E:8")]
        assert log_loss(players, {"probable": 1.0}) < log_loss(players, {})


class TestServing:
    def test_disabled_or_unknown_label_changes_nothing(self) -> None:
        off = StatusCalibration(offsets={"probable": 1.0}, enabled=False)
        assert off.adjust(0.3, "probable") == 0.3
        on = StatusCalibration(offsets={"probable": 1.0}, enabled=True)
        assert on.adjust(0.3, "questionable") == 0.3
        assert on.adjust(0.3, "probable") == pytest.approx(adjusted(0.3, 1.0))

    def test_round_trip(self) -> None:
        fit = StatusCalibration(
            offsets={"probable": 1.2}, enabled=True, rounds=["E2026:7"], held_out={"model": 0.5}
        )
        restored = StatusCalibration.from_dict(fit.to_dict())
        assert (
            restored.offsets == fit.offsets and restored.enabled and restored.rounds == ["E2026:7"]
        )

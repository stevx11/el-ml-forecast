"""Squad minutes reconciliation: the arithmetic, the strength choice, the gate."""

from __future__ import annotations

import json
import math
import random
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from euroleague_ai.ml.reconcile import (
    CandidateMinutes,
    MinutesReconciliation,
    ReconciliationFit,
    choose_alpha,
    evaluate,
    load_reconciliation,
    paired_player_delta,
    reconciliation_path,
    scale_sides,
    squad_target,
)
from euroleague_ai.models.enums import PredictionTarget


class TestScaleSides:
    def test_full_strength_hits_the_target_per_side(self) -> None:
        home = [30.0, 28.0, 25.0, 24.0, 22.0, 20.0, 18.0, 15.0, 12.0, 6.0]
        away = [33.0, 31.0, 29.0, 27.0, 25.0, 23.0, 21.0, 17.0, 14.0, 9.0]
        minutes = home + away
        plays = [1.0] * 9 + [0.5] + [1.0] * 10
        sides = ["home"] * 10 + ["away"] * 10
        scaled = scale_sides(minutes, plays, sides, target=200.0, alpha=1.0)
        for side in ("home", "away"):
            total = sum(p * m for p, m, s in zip(plays, scaled, sides, strict=True) if s == side)
            assert total == pytest.approx(200.0)
        # One factor per side: ratios within a side are preserved.
        assert scaled[0] / scaled[1] == pytest.approx(30.0 / 28.0)

    def test_capped_minutes_are_redistributed(self) -> None:
        # Sum 100 toward 140: a flat factor 1.4 would put the star at 49.
        scaled = scale_sides(
            [35.0, 25.0, 20.0, 20.0], [1.0] * 4, ["a"] * 4, target=140.0, alpha=1.0
        )
        assert scaled[0] == 40.0
        assert sum(scaled) == pytest.approx(140.0)
        assert scaled[1] / scaled[2] == pytest.approx(25.0 / 20.0)

    def test_an_impossible_target_stops_at_the_cap(self) -> None:
        assert scale_sides([25.0, 25.0], [1.0, 1.0], ["a", "a"], target=100.0, alpha=1.0) == [
            40.0,
            40.0,
        ]

    def test_zero_strength_changes_nothing(self) -> None:
        minutes = [30.0, 20.0]
        assert scale_sides(minutes, [1.0, 1.0], ["a", "a"], target=200.0, alpha=0.0) == minutes

    def test_partial_strength_moves_part_way(self) -> None:
        # Sum 50 toward 200: full factor 4, half strength factor 2.
        scaled = scale_sides([25.0, 25.0], [1.0, 1.0], ["a", "a"], target=200.0, alpha=0.5)
        assert scaled == [pytest.approx(40.0), pytest.approx(40.0)]

    def test_cap_and_empty_side(self) -> None:
        assert scale_sides([30.0], [1.0], ["a"], target=200.0, alpha=1.0) == [40.0]
        assert scale_sides([12.0], [0.0], ["a"], target=200.0, alpha=1.0) == [12.0]


def _world(*, team_noise: float, games: int, seed: int) -> list[CandidateMinutes]:
    """Squads whose projections are all off by one shared factor per side.

    That shared factor is exactly what reconciliation can see (the squad sum is
    too high or too low) and fix. ``team_noise=0`` gives projections that are
    already right on average, where scaling has nothing to fix.
    """
    rng = random.Random(seed)
    out: list[CandidateMinutes] = []
    for g in range(games):
        side = (g, "home")
        shared = 1.0 + rng.uniform(-team_noise, team_noise)
        truths = [34, 30, 28, 25, 22, 20, 18, 12, 8, 3]
        for truth in truths:
            actual = max(0.0, truth + rng.gauss(0, 3))
            out.append(
                CandidateMinutes(side=side, predicted=truth * shared, plays=1.0, actual=actual)
            )
    return out


class TestChoosingTheStrength:
    def test_picks_scaling_when_squad_sums_carry_the_error(self) -> None:
        fit, test = (
            _world(team_noise=0.25, games=300, seed=1),
            _world(team_noise=0.25, games=300, seed=2),
        )
        target = squad_target(fit)
        alpha, table, base, scaled = choose_alpha(fit, test, target=target)
        assert alpha >= 0.7
        assert len(table) == 11
        assert scaled["player_mae"] < base["player_mae"]
        assert scaled["squad_mae"] < base["squad_mae"]

    def test_leaves_good_projections_alone(self) -> None:
        """With no shared error, scaling only adds the actual totals' noise."""
        fit, test = (
            _world(team_noise=0.0, games=300, seed=3),
            _world(team_noise=0.0, games=300, seed=4),
        )
        alpha, *_ = choose_alpha(fit, test, target=squad_target(fit) + 15.0)
        assert alpha == 0.0

    def test_takes_the_strongest_setting_that_does_not_hurt_players(self) -> None:
        """Player error flat, squad error falling: take as much as is free.

        Each squad is over-projected by a shared 5%, plus large individual
        noise, so scaling barely moves player error but fixes squad totals.
        """
        rng = random.Random(9)

        def world(seed_offset: int) -> list[CandidateMinutes]:
            out = []
            for g in range(400):
                shared = 1.05
                for truth in [34, 30, 28, 25, 22, 20, 18, 12, 8, 3]:
                    actual = max(0.0, truth + rng.gauss(0, 6))
                    out.append(
                        CandidateMinutes(
                            side=(g, seed_offset),
                            predicted=truth * shared,
                            plays=1.0,
                            actual=actual,
                        )
                    )
            return out

        fit, test = world(0), world(1)
        alpha, table, base, scaled = choose_alpha(fit, test, target=squad_target(fit))
        unscaled_mae = table[0]["player_mae"]
        chosen = next(row for row in table if row["alpha"] == alpha)
        assert chosen["player_mae"] <= unscaled_mae
        stronger = [row for row in table if row["alpha"] > alpha]
        assert all(row["player_mae"] > unscaled_mae for row in stronger)
        lowest = min(table, key=lambda row: row["player_mae"])["alpha"]
        assert alpha >= lowest
        assert scaled["squad_mae"] < base["squad_mae"]

    def test_the_grid_must_start_unscaled(self) -> None:
        with pytest.raises(ValueError, match=r"start at 0\.0"):
            choose_alpha([], [], target=200.0, grid=(0.5, 1.0))

    def test_evaluate_counts_only_players_who_played_for_player_error(self) -> None:
        side = (1, "home")
        rows = [
            CandidateMinutes(side=side, predicted=30.0, plays=0.9, actual=28.0),
            CandidateMinutes(side=side, predicted=10.0, plays=0.3, actual=None),
        ]
        result = evaluate(rows, target=30.0, alpha=0.0)
        assert result["players"] == 1
        assert result["player_mae"] == pytest.approx(2.0)
        # Squad: 0.9 * 30 + 0.3 * 10 = 30 projected vs 28 actual.
        assert result["squad_bias"] == pytest.approx(2.0)


def _fit(
    *,
    alpha: float,
    unscaled: tuple[float, float],
    scaled: tuple[float, float],
    delta: tuple[float, float] = (0.0, 0.01),
) -> ReconciliationFit:
    def metrics(player: float, squad: float) -> dict[str, float]:
        return {
            "alpha": 0.0,
            "players": 10,
            "player_mae": player,
            "player_bias": 0.0,
            "team_games": 2,
            "squad_mae": squad,
            "squad_bias": 0.0,
        }

    return ReconciliationFit(
        alpha=alpha,
        target_minutes=201.3,
        validation=[],
        test_unscaled=metrics(*unscaled),
        test_scaled=metrics(*scaled),
        player_delta={"mean": delta[0], "se": delta[1], "players": 1000, "squads": 200},
        minutes_model_id="m-id",
        minutes_model="minutes_lgbm 0.1.10",
        plays_model_id="p-id",
        plays_model="plays_lgbm 0.1.2",
        periods={},
    )


class TestPairedDelta:
    def _side(self, g: int, predicted: list[float], actual: list[float]) -> list[CandidateMinutes]:
        return [
            CandidateMinutes(side=(g, "h"), predicted=p, plays=1.0, actual=a)
            for p, a in zip(predicted, actual, strict=True)
        ]

    def test_no_scaling_means_no_change(self) -> None:
        rows = self._side(1, [30.0, 20.0], [28.0, 25.0]) + self._side(2, [30.0, 20.0], [31.0, 18.0])
        delta = paired_player_delta(rows, target=50.0, alpha=0.0)
        assert delta["mean"] == pytest.approx(0.0)
        assert delta["squads"] == 2

    def test_mean_matches_the_mae_difference(self) -> None:
        rows = self._side(1, [30.0, 20.0], [36.0, 24.0]) + self._side(2, [25.0, 25.0], [27.0, 27.0])
        delta = paired_player_delta(rows, target=60.0, alpha=1.0)
        scaled = evaluate(rows, target=60.0, alpha=1.0)["player_mae"]
        unscaled = evaluate(rows, target=60.0, alpha=0.0)["player_mae"]
        assert delta["mean"] == pytest.approx(scaled - unscaled, abs=1e-4)
        assert delta["mean"] < 0  # both squads were under-projected; scaling helps
        assert delta["se"] > 0

    def test_needs_two_squads_for_an_error_bar(self) -> None:
        rows = self._side(1, [30.0], [28.0])
        assert math.isnan(paired_player_delta(rows, target=30.0, alpha=1.0)["se"])


class TestGate:
    def test_on_when_players_hold_and_squads_improve(self) -> None:
        assert _fit(
            alpha=0.8, unscaled=(4.40, 12.0), scaled=(4.38, 3.1), delta=(-0.02, 0.01)
        ).enabled

    def test_on_when_players_are_worse_only_by_noise(self) -> None:
        """The real 0.1.10 case: +0.002 minutes, squad error 9.74 -> 4.02."""
        fit = _fit(alpha=0.8, unscaled=(4.380, 9.74), scaled=(4.382, 4.02), delta=(0.002, 0.004))
        assert not fit.players_significantly_worse
        assert fit.enabled

    def test_off_when_player_minutes_get_significantly_worse(self) -> None:
        fit = _fit(alpha=0.8, unscaled=(4.40, 12.0), scaled=(4.45, 3.1), delta=(0.05, 0.01))
        assert fit.players_significantly_worse
        assert not fit.enabled

    def test_off_when_the_error_bar_is_unknown(self) -> None:
        fit = _fit(
            alpha=0.8, unscaled=(4.40, 12.0), scaled=(4.38, 3.1), delta=(float("nan"), float("nan"))
        )
        assert not fit.enabled

    def test_off_when_squads_do_not_improve(self) -> None:
        assert not _fit(alpha=0.8, unscaled=(4.40, 3.0), scaled=(4.40, 3.0)).enabled

    def test_off_at_zero_strength(self) -> None:
        assert not _fit(alpha=0.0, unscaled=(4.4, 12.0), scaled=(4.4, 12.0)).enabled


def _models(minutes_id: str, plays_id: str) -> dict[PredictionTarget, Any]:
    return {
        PredictionTarget.MINUTES: SimpleNamespace(
            model_version=SimpleNamespace(id=uuid.UUID(minutes_id))
        ),
        PredictionTarget.PLAYS: SimpleNamespace(
            model_version=SimpleNamespace(id=uuid.UUID(plays_id))
        ),
    }


class TestSavedFit:
    def test_round_trip_and_version_matching(self, tmp_path: Path) -> None:
        minutes_id, plays_id = str(uuid.uuid4()), str(uuid.uuid4())
        fit = _fit(alpha=0.6, unscaled=(4.4, 12.0), scaled=(4.3, 4.0))
        fit.minutes_model_id, fit.plays_model_id = minutes_id, plays_id
        path = reconciliation_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fit.to_dict()))

        loaded = load_reconciliation(tmp_path)
        assert isinstance(loaded, MinutesReconciliation)
        assert loaded.alpha == 0.6
        assert loaded.applies_to(_models(minutes_id, plays_id))
        assert not loaded.applies_to(_models(str(uuid.uuid4()), plays_id))

    def test_a_failed_fit_is_never_served(self, tmp_path: Path) -> None:
        fit = _fit(alpha=0.6, unscaled=(4.4, 12.0), scaled=(4.6, 4.0), delta=(0.2, 0.01))
        path = reconciliation_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fit.to_dict()))
        assert load_reconciliation(tmp_path) is None

    def test_missing_file_means_off(self, tmp_path: Path) -> None:
        assert load_reconciliation(tmp_path) is None

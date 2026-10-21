"""Squad stat reconciliation: anchors, the pull, the gate, and serving.

The synthetic league has one thing the player models miss: game pace. Each
squad's real total scales with it, every player's expected value ignores it, so
summed lines are right on average and wrong game by game. The anchor sees pace
through the team context, which is exactly the gap it exists to close.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from euroleague_ai.ml.squad_stats import (
    ASSISTS,
    FACTOR_BOUNDS,
    POINTS,
    REBOUNDS,
    SPECS,
    THREES,
    Sides,
    StatsReconciliation,
    design_rows,
    fit_stat,
    load_stats_reconciliation,
    side_factors,
    squad_sums,
)

if TYPE_CHECKING:
    from pathlib import Path

    from euroleague_ai.models.enums import PredictionTarget


def _league(games: int = 900, seed: int = 0) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    labels: list[Any] = []
    fixtures: list[Any] = []
    features: list[dict[str, Any]] = []
    values: dict[PredictionTarget, list[float]] = {
        t: [] for t in (POINTS, ASSISTS, REBOUNDS, THREES)
    }
    actual: dict[PredictionTarget, list[float | None]] = {t: [] for t in values}
    plays: list[float] = []
    for _ in range(games):
        game = uuid.uuid4()
        pace = float(rng.uniform(62, 82))
        for home in (0, 1):
            side = (game, home)
            context = {
                "is_home": float(home),
                "matchup_expected_pace": pace,
                "game_expected_team_points": 80.0 * pace / 72.0 + float(rng.normal(0, 2)),
                "game_expected_total": 160.0 * pace / 72.0,
                "team_oreb_per_game": 10.0,
                "team_dreb_per_game": 24.0,
                "opp_oreb_per_game": 10.0,
                "opp_dreb_per_game": 24.0,
                "team_fg3a_per_game": 25.0,
            }
            for _ in range(10):
                share = float(rng.uniform(0.04, 0.16))
                labels.append(side)
                fixtures.append(game)
                features.append(context)
                plays.append(0.95)
                # Expected values: the share of an average (pace 72) game.
                values[POINTS].append(80.0 * share)
                values[ASSISTS].append(16.0 * share)
                values[REBOUNDS].append(35.0 * share)
                values[THREES].append(9.0 * share)
                # Reality: the same share of a game at this pace, plus noise.
                scale = pace / 72.0
                actual[POINTS].append(max(0.0, 80.0 * share * scale + float(rng.normal(0, 2))))
                actual[ASSISTS].append(max(0.0, 16.0 * share * scale + float(rng.normal(0, 0.8))))
                actual[REBOUNDS].append(max(0.0, 35.0 * share * scale + float(rng.normal(0, 1.2))))
                actual[THREES].append(max(0.0, 9.0 * share * scale + float(rng.normal(0, 0.6))))
    sides = Sides.build(labels, fixtures, features)
    fit_mask = [i < len(labels) // 2 for i in range(len(labels))]
    return {
        "values": values,
        "plays": plays,
        "actual": actual,
        "sides": sides,
        "fit_mask": fit_mask,
        "features": features,
        "labels": labels,
    }


def _oracle_squad_mae(world: dict[str, Any], target: PredictionTarget) -> float:
    """Squad error of the true expectation (pace known) on the test half."""
    projected: dict[Any, float] = {}
    real: dict[Any, float] = {}
    for i, label in enumerate(world["labels"]):
        if world["fit_mask"][i]:
            continue
        scale = world["features"][i]["matchup_expected_pace"] / 72.0
        projected[label] = projected.get(label, 0.0) + world["values"][target][i] * scale
        real[label] = real.get(label, 0.0) + (world["actual"][target][i] or 0.0)
    return sum(abs(projected[k] - real[k]) for k in projected) / len(projected)


@pytest.fixture(scope="module")
def league() -> dict[str, Any]:
    return _league()


class TestSides:
    def test_opponents_and_context(self) -> None:
        g = uuid.uuid4()
        sides = Sides.build([(g, 1), (g, 1), (g, 0)], [g, g, g], [{"x": 1}, {"x": 1}, {"x": 2}])
        assert sides.opponent == {(g, 1): (g, 0), (g, 0): (g, 1)}
        assert sides.members[(g, 1)] == [0, 1]
        assert sides.context[(g, 0)] == {"x": 2}

    def test_missing_context_leaves_the_side_alone(self) -> None:
        g = uuid.uuid4()
        sides = Sides.build([(g, 1), (g, 0)], [g, g], [{"game_expected_team_points": 90.0}, {}])
        sums = {POINTS: squad_sums([20.0, 20.0], [1.0, 1.0], sides)}
        design = design_rows(SPECS[POINTS], sums, sides)
        assert all(row is None for row in design.values())  # pace is missing on both
        factors = side_factors(design, sums[POINTS], [0.0, 1.0, 0.0, 0.0], alpha=1.0)
        assert set(factors.values()) == {1.0}

    def test_factors_are_bounded(self) -> None:
        g = uuid.uuid4()
        sides = Sides.build([(g, 1)], [g], [{}])
        design = {(g, 1): [1.0]}
        high = side_factors(design, {(g, 1): 10.0}, [1000.0], alpha=1.0)
        low = side_factors(design, {(g, 1): 10.0}, [0.1], alpha=1.0)
        assert high[(g, 1)] == FACTOR_BOUNDS[1] and low[(g, 1)] == FACTOR_BOUNDS[0]
        assert sides.labels == [(g, 1)]


class TestFit:
    @pytest.mark.parametrize("target", [POINTS, ASSISTS, REBOUNDS, THREES])
    def test_the_anchor_closes_the_pace_gap(
        self, league: dict[str, Any], target: PredictionTarget
    ) -> None:
        fit = fit_stat(
            target, **{k: league[k] for k in ("values", "plays", "actual", "sides", "fit_mask")}
        )
        base, new = fit.test_unscaled, fit.test_scaled
        assert fit.enabled, fit.to_dict()
        assert fit.alpha > 0.5
        assert new["squad_mae"] < base["squad_mae"]
        # As good as knowing the true pace, within noise: ten players' noise is
        # the floor no anchor can get under.
        assert new["squad_mae"] < 1.1 * _oracle_squad_mae(league, target)
        assert not fit.players_significantly_worse
        if target in (POINTS, REBOUNDS):  # pace dominates their noise here
            assert new["player_rmse"] < base["player_rmse"]
        assert len(fit.coefficients or []) == 1 + len(SPECS[target].names())

    def test_a_useless_anchor_stays_off(self) -> None:
        """No pace in the world: nothing to learn, so the gate keeps it off or harmless."""
        import numpy as np

        world = _league(games=600, seed=3)
        rng = np.random.default_rng(4)
        # Reality now ignores pace: actuals are the expected values plus noise.
        world["actual"][POINTS] = [
            max(0.0, v + float(rng.normal(0, 2))) for v in world["values"][POINTS]
        ]
        fit = fit_stat(
            POINTS, **{k: world[k] for k in ("values", "plays", "actual", "sides", "fit_mask")}
        )
        base, new = fit.test_unscaled, fit.test_scaled
        if fit.enabled:
            assert new["squad_mae"] < base["squad_mae"]
            assert not fit.players_significantly_worse
        assert new["squad_mae"] <= base["squad_mae"] * 1.05

    def test_too_few_squads_gives_no_anchor(self) -> None:
        world = _league(games=4, seed=5)
        fit = fit_stat(
            POINTS, **{k: world[k] for k in ("values", "plays", "actual", "sides", "fit_mask")}
        )
        assert fit.coefficients is None and not fit.enabled and fit.alpha == 0.0


class TestServing:
    def test_serving_reproduces_the_fit(self, league: dict[str, Any]) -> None:
        fit = fit_stat(
            POINTS, **{k: league[k] for k in ("values", "plays", "actual", "sides", "fit_mask")}
        )
        assert fit.coefficients is not None
        served = StatsReconciliation(targets={POINTS: (fit.alpha, fit.coefficients)}, model_ids={})

        # One fixture, as inference sees it: sides labelled by is_home.
        first = league["labels"][0][0]
        idx = [i for i, label in enumerate(league["labels"]) if label[0] == first]
        rows = [league["features"][i] for i in idx]
        factors = served.factors(
            {POINTS: [league["values"][POINTS][i] for i in idx]},
            [league["plays"][i] for i in idx],
            [f["is_home"] for f in rows],
            rows,
        )[POINTS]

        sums = {POINTS: squad_sums(league["values"][POINTS], league["plays"], league["sides"])}
        design = design_rows(SPECS[POINTS], sums, league["sides"])
        expected = side_factors(design, sums[POINTS], fit.coefficients, alpha=fit.alpha)
        assert factors == [pytest.approx(expected[league["labels"][i]]) for i in idx]
        assert len({round(f, 9) for f in factors}) <= 2  # one factor per side

    def test_ruled_out_players_hand_their_share_to_teammates(self) -> None:
        served = StatsReconciliation(targets={POINTS: (1.0, [0.0, 0.0, 1.0, 0.0])}, model_ids={})
        context = {"game_expected_team_points": 80.0, "matchup_expected_pace": 72.0, "is_home": 1.0}
        values = [20.0, 20.0, 20.0, 20.0]
        full = served.factors({POINTS: values}, [1.0] * 4, [1.0] * 4, [context] * 4)[POINTS]
        one_out = served.factors({POINTS: values}, [0.0, 1.0, 1.0, 1.0], [1.0] * 4, [context] * 4)[
            POINTS
        ]
        assert full == [pytest.approx(1.0)] * 4
        # 60 expected from the three who play, anchored to 80.
        assert one_out[1] == pytest.approx(80.0 / 60.0)

    def test_load_keeps_only_enabled_stats(self, tmp_path: Path) -> None:
        path = tmp_path / "reconciliation" / "stats.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model_ids": {"POINTS": "a", "PLAYS": "b"},
                    "targets": {
                        "POINTS": {
                            "enabled": True,
                            "alpha": 0.7,
                            "coefficients": [1.0, 0.9, 0.1, 0.0],
                        },
                        "ASSISTS": {
                            "enabled": False,
                            "alpha": 0.0,
                            "coefficients": [1.0, 1.0, 0.0, 0.0, 0.0],
                        },
                    },
                }
            )
        )
        loaded = load_stats_reconciliation(tmp_path)
        assert loaded is not None
        assert set(loaded.targets) == {POINTS}
        assert loaded.targets[POINTS][0] == 0.7

        path.write_text(json.dumps({"model_ids": {}, "targets": {"POINTS": {"enabled": False}}}))
        assert load_stats_reconciliation(tmp_path) is None
        assert math.isfinite(FACTOR_BOUNDS[0])

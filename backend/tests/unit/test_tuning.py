"""Hyperparameter search: sampling, and the separation of selection from reporting.

Fitting needs LightGBM and a database, so that half is exercised by
``euroleague-ai ml tune``. What is tested here is the part that would silently
produce a fake result: the sampler, and the structural guarantee that the search
never reads the test split.
"""

from __future__ import annotations

import ast
import inspect
import random

import pytest

from euroleague_ai.ml import tuning
from euroleague_ai.ml.tuning import (
    INTEGER_PARAMS,
    MIN_TRIALS,
    SEARCH_SPACE,
    Trial,
    load_params,
    params_path,
    sample_params,
    save_params,
)
from euroleague_ai.models.enums import PredictionTarget


class TestSampling:
    def test_every_parameter_is_drawn(self) -> None:
        params = sample_params(random.Random(1))
        assert set(params) == set(SEARCH_SPACE)

    def test_draws_stay_inside_the_space(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            params = sample_params(rng)
            for name, value in params.items():
                low, high = SEARCH_SPACE[name]
                # num_leaves is additionally capped by max_depth, so it can sit
                # below the nominal floor; everything else must be in range.
                if name == "num_leaves":
                    assert value <= high
                else:
                    assert low <= value <= high, (name, value)

    def test_integer_parameters_are_integers(self) -> None:
        """LightGBM rejects a float where it wants a count."""
        rng = random.Random(3)
        for _ in range(50):
            params = sample_params(rng)
            for name in INTEGER_PARAMS:
                assert isinstance(params[name], int), name

    def test_num_leaves_cannot_exceed_the_depth_limit(self) -> None:
        """Otherwise max_depth silently governs and half the space is duplicates.

        A tree of depth d has at most 2**d - 1 leaves. Sampling 200 leaves at
        depth 4 does not give a wider tree, it gives the same tree as 15 leaves
        at depth 4 -- so those draws would explore nothing.
        """
        rng = random.Random(11)
        for _ in range(200):
            params = sample_params(rng)
            assert params["num_leaves"] <= 2 ** params["max_depth"] - 1

    def test_sampling_is_reproducible(self) -> None:
        """A surprising result should be re-examinable, not merely re-rollable."""
        assert sample_params(random.Random(42)) == sample_params(random.Random(42))

    def test_different_seeds_explore_different_points(self) -> None:
        assert sample_params(random.Random(1)) != sample_params(random.Random(2))


class TestSelectionNeverSeesTest:
    """The property that decides whether the reported gain is real.

    A search that picks the best test score and then reports it is measuring the
    maximum of N noisy draws, not accuracy. With enough trials something always
    looks good. These tests pin the separation structurally, because it is the
    kind of thing a well-meaning refactor quietly undoes.
    """

    def test_the_search_function_never_reads_the_test_split(self) -> None:
        tree = ast.parse(inspect.getsource(tuning.search))
        function = tree.body[0]

        loaded = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert not {name for name in loaded if "test" in name.lower()}

    def test_the_module_never_trains_a_final_model(self) -> None:
        """Retraining and test scoring belong to the caller, after selection."""
        tree = ast.parse(inspect.getsource(tuning))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "train_target" not in called

    def test_scoring_helper_is_named_for_validation(self) -> None:
        assert hasattr(tuning, "_validation_mae")


class TestSearchGuards:
    def test_too_few_trials_is_refused(self) -> None:
        """Five random draws over eight dimensions tells you nothing."""
        with pytest.raises(ValueError, match="at least"):
            tuning.search([], PredictionTarget.POINTS, trials=MIN_TRIALS - 1)


class TestPersistence:
    def test_params_round_trip(self, tmp_path) -> None:
        params = {"num_leaves": 63, "learning_rate": 0.04}
        save_params(PredictionTarget.POINTS, params, root=tmp_path)
        assert load_params(PredictionTarget.POINTS, root=tmp_path) == params

    def test_absent_params_return_none_not_defaults(self, tmp_path) -> None:
        """So a caller can report *which* settings it used.

        Falling back silently would make a tuned run and an untuned one
        indistinguishable in the training report.
        """
        assert load_params(PredictionTarget.ASSISTS, root=tmp_path) is None

    def test_unreadable_params_degrade_to_none(self, tmp_path) -> None:
        path = params_path(PredictionTarget.REBOUNDS, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_params(PredictionTarget.REBOUNDS, root=tmp_path) is None

    def test_each_target_has_its_own_file(self, tmp_path) -> None:
        """Minutes is a rotation decision; three-pointers are closer to a coin.

        There is no reason the same settings should suit both.
        """
        save_params(PredictionTarget.MINUTES, {"num_leaves": 31}, root=tmp_path)
        save_params(PredictionTarget.POINTS, {"num_leaves": 127}, root=tmp_path)

        assert load_params(PredictionTarget.MINUTES, root=tmp_path) == {"num_leaves": 31}
        assert load_params(PredictionTarget.POINTS, root=tmp_path) == {"num_leaves": 127}


class TestTrialRecord:
    def test_a_trial_serialises(self) -> None:
        trial = Trial(params={"num_leaves": 31}, validation_mae=4.4321, best_iteration=140)
        payload = trial.as_dict()
        assert payload["validation_mae"] == 4.4321
        assert payload["best_iteration"] == 140

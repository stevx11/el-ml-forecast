"""Walk-forward evaluation: folds that never see the future, and real training per fold."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from euroleague_ai.ml.dataset import TrainingRow
from euroleague_ai.ml.walkforward import evaluate_target, save_report, season_folds
from euroleague_ai.models.enums import PredictionTarget

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2019, 10, 1, 18, 0, tzinfo=UTC)


def _rows(seasons: int = 5, games: int = 160, seed: int = 0) -> list[TrainingRow]:
    """Points that a noisy season mean predicts better than nothing, season by season."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[TrainingRow] = []
    for s in range(seasons):
        start = T0 + timedelta(days=365 * s)
        for g in range(games):
            game = uuid.uuid4()
            for home in (0.0, 1.0):
                for _ in range(8):
                    mu = rng.uniform(2.0, 16.0)
                    rows.append(
                        TrainingRow(
                            snapshot_id=uuid.uuid4(),
                            player_id=uuid.uuid4(),
                            game_id=game,
                            season_code=f"E{2019 + s}",
                            round_number=None,
                            tip_off_utc=start + timedelta(hours=g),
                            prediction_timestamp=start + timedelta(hours=g),
                            features={
                                "is_home": home,
                                "season_points_mean": mu * rng.normal(1.0, 0.3),
                                "last5_points_mean": mu * rng.normal(1.0, 0.45),
                                "signal": mu,
                                "games_played_career_window": 30.0,
                            },
                            target=float(rng.poisson(mu)),
                            actual_minutes=20.0,
                        )
                    )
    return rows


class TestFolds:
    def test_training_never_reaches_the_test_season(self) -> None:
        rows = _rows(seasons=5, games=40)
        folds = season_folds(rows, seasons=3, min_test_rows=100)
        assert [f.season for f in folds] == ["E2021", "E2022", "E2023"]
        for fold in folds:
            first_test = min(r.tip_off_utc for r in fold.test)
            assert max(r.tip_off_utc for r in fold.train + fold.validation) < first_test
            assert {r.season_code for r in fold.test} == {fold.season}
            assert fold.season not in {r.season_code for r in fold.train + fold.validation}
            # Validation is the latest slice of the earlier seasons.
            assert max(r.tip_off_utc for r in fold.train) <= min(
                r.tip_off_utc for r in fold.validation
            )
            assert 0.1 < len(fold.validation) / (len(fold.train) + len(fold.validation)) < 0.2

    def test_needs_earlier_seasons_and_enough_rows(self) -> None:
        rows = _rows(seasons=4, games=40)
        # The first two seasons can only train.
        folds = season_folds(rows, seasons=10, min_test_rows=100)
        assert [f.season for f in folds] == ["E2021", "E2022"]
        # A season with too few rows (a new season's first round) is not a fold.
        assert season_folds(rows, seasons=10, min_test_rows=10_000) == []

    def test_current_season_is_marked_partial(self) -> None:
        folds = season_folds(
            _rows(seasons=4, games=40), seasons=2, min_test_rows=100, current_season="E2022"
        )
        assert [(f.season, f.partial) for f in folds] == [("E2021", False), ("E2022", True)]


class TestEvaluate:
    def test_one_real_model_per_season(self, tmp_path: Path) -> None:
        rows = _rows(seasons=5, games=160)
        result = evaluate_target(
            rows,
            PredictionTarget.POINTS,
            seasons=3,
            current_season="E2023",
            num_rounds=150,
            early_stopping=20,
        )
        assert [f.season for f in result.folds] == ["E2021", "E2022", "E2023"]
        assert result.folds[-1].partial
        for fold in result.folds:
            assert fold.test_rows == 160 * 16
            assert fold.baseline  # judged against a real baseline on the same rows
            assert fold.beats_baseline, fold.to_dict()
            assert fold.covered_80 is not None and 0.6 < fold.covered_80 < 0.95
        summary = result.summary()
        assert summary["folds"] == 3 and summary["beats_baseline"] == 3
        assert summary["worst_improvement_pct"] <= summary["median_improvement_pct"]
        assert summary["median_improvement_complete_pct"] is not None

        path = save_report(
            [result],
            feature_set_version="v5",
            now=datetime(2026, 10, 21, tzinfo=UTC),
            root=tmp_path,
        )
        saved = json.loads(path.read_text())
        assert saved["feature_set_version"] == "v5"
        assert len(saved["targets"][0]["folds"]) == 3
        assert path.name.endswith("-v5.json")

    def test_expected_head_is_reported_when_asked(self) -> None:
        rows = _rows(seasons=4, games=120, seed=2)
        result = evaluate_target(
            rows,
            PredictionTarget.POINTS,
            seasons=1,
            fit_expected=True,
            num_rounds=100,
            early_stopping=20,
        )
        (fold,) = result.folds
        assert fold.median_rmse is not None and fold.expected_rmse is not None

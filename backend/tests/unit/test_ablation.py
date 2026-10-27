"""Ablation: a family that carries signal shows up as helping; noise does not."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from euroleague_ai.features.context_v6 import V6_FAMILIES, V6_FEATURE_NAMES
from euroleague_ai.ml.ablation import ablate, save_report, strip, variants, verdict
from euroleague_ai.models.enums import PredictionTarget
from tests.unit.test_walkforward import _rows

if TYPE_CHECKING:
    from pathlib import Path


def _world():
    """The true mean lives only in a v6 feature (opponent defence); the coach
    feature is noise; form features are noisy views of the mean."""
    import numpy as np

    rng = np.random.default_rng(7)
    out = []
    for row in _rows(seasons=3, games=150, seed=3):
        features = dict(row.features)
        features["opp_pos_points_pg"] = features.pop("signal")
        features["coach_games_in_charge"] = float(rng.integers(1, 30))
        out.append(dataclasses.replace(row, features=features))
    return out


def test_variants_and_strip() -> None:
    names = [name for name, _ in variants()]
    assert names[:2] == ["all v6", "v5 only"]
    assert names[2:] == [f"without {family}" for family in V6_FAMILIES]
    assert dict(variants())["v5 only"] == frozenset(V6_FEATURE_NAMES)
    row = _world()[0]
    (stripped,) = strip([row], frozenset({"opp_pos_points_pg"}))
    assert "opp_pos_points_pg" not in stripped.features
    assert "opp_pos_points_pg" in row.features  # the original is untouched


def test_signal_helps_and_noise_does_not(tmp_path: Path) -> None:
    report = ablate(PredictionTarget.POINTS, _world(), num_rounds=120, early_stopping=15)
    by_name = {row["variant"]: row for row in report["variants"]}
    assert by_name["all v6"]["change_pct"] == 0.0
    assert by_name["without opponent_position"]["helps"]
    assert by_name["without opponent_position"]["change_pct"] > 1.0
    assert by_name["v5 only"]["change_pct"] > 1.0
    assert not by_name["without coach"]["helps"]

    families = verdict([report])
    assert families["opponent_position"] == ["POINTS"]
    assert families["coach"] == []

    path = save_report([report], now=datetime(2026, 10, 27, tzinfo=UTC), root=tmp_path)
    assert path.is_file()


def test_excluded_families_setting() -> None:
    import pytest

    from euroleague_ai.core.config import Settings
    from euroleague_ai.features.context_v6 import excluded_feature_names

    settings = Settings(excluded_feature_families="coach, schedule_venue")  # type: ignore[arg-type]
    assert settings.excluded_feature_families == ["coach", "schedule_venue"]
    names = excluded_feature_names(settings.excluded_feature_families)
    assert names == frozenset(V6_FAMILIES["coach"] + V6_FAMILIES["schedule_venue"])
    assert excluded_feature_names([]) == frozenset()
    with pytest.raises(ValueError, match="unknown feature family coaches"):
        excluded_feature_names(["coaches"])

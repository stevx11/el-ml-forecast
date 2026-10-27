"""Activation with the per-season gate, through the real model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select

from euroleague_ai.ml.registry import register_model
from euroleague_ai.ml.train import train_target
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import ModelVersion
from tests.conftest import requires_db
from tests.unit.test_walkforward import _rows

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = [pytest.mark.integration, requires_db]


@pytest.fixture(scope="module")
def result() -> Any:
    return train_target(
        _rows(seasons=3, games=100),
        PredictionTarget.POINTS,
        num_rounds=80,
        early_stopping=10,
        fit_expected=False,
    )


def _gate(passes: bool | None) -> dict[str, Any]:
    return {
        "judged": passes is not None,
        "passes": passes,
        "complete_seasons": 3 if passes is not None else 1,
        "beats_baseline": 3 if passes else 1,
        "median_improvement_pct": 3.0 if passes else -1.0,
        "reason": "test",
        "seasons": [],
    }


async def _register(
    session: AsyncSession, result: Any, tmp_path: Path, **kwargs: Any
) -> ModelVersion:
    return await register_model(
        session, result, feature_set_version="v5", artifact_root=tmp_path, **kwargs
    )


class TestSeasonGate:
    async def test_a_failing_season_gate_blocks_activation(
        self, session: AsyncSession, result: Any, tmp_path: Path
    ) -> None:
        assert result.beats_baseline  # wins on the single split
        model = await _register(session, result, tmp_path, season_gate=_gate(False))
        assert not model.is_active
        assert model.metrics["season_gate"]["passes"] is False

    async def test_passing_activates_and_retires_the_previous_model(
        self, session: AsyncSession, result: Any, tmp_path: Path
    ) -> None:
        first = await _register(session, result, tmp_path, season_gate=_gate(True))
        second = await _register(session, result, tmp_path, season_gate=_gate(True))
        active = (
            (
                await session.execute(
                    select(ModelVersion.id).where(
                        ModelVersion.target == PredictionTarget.POINTS, ModelVersion.is_active
                    )
                )
            )
            .scalars()
            .all()
        )
        assert active == [second.id] and first.id != second.id

    async def test_an_unjudged_gate_falls_back_to_the_single_split(
        self, session: AsyncSession, result: Any, tmp_path: Path
    ) -> None:
        model = await _register(session, result, tmp_path, season_gate=_gate(None))
        assert model.is_active

    async def test_an_explicit_override_still_wins(
        self, session: AsyncSession, result: Any, tmp_path: Path
    ) -> None:
        model = await _register(session, result, tmp_path, season_gate=_gate(False), activate=True)
        assert model.is_active

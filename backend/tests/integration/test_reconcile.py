"""Squad minutes reconciliation against the real schema: fit, save, apply."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.config import Settings
from euroleague_ai.features.backfill import backfill_snapshots
from euroleague_ai.ml.availability import derive_feature_names, load_availability_rows
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.reconcile import (
    MinutesReconciliation,
    fit_minutes_reconciliation,
    reconciliation_path,
)
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import Prediction
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db
from tests.integration.test_availability import CLOCK, _game, _seed_absences
from tests.integration.test_expectation import _Constant, _model_version

pytestmark = [pytest.mark.integration, requires_db]


async def _models(session: AsyncSession, settings: Settings) -> dict[PredictionTarget, LoadedModel]:
    """Minutes: 30 for everyone. P(plays): 0.9 for everyone."""
    await backfill_snapshots(session, clock=CLOCK, settings=settings)
    rows, _ = await load_availability_rows(session, settings=settings)
    names = derive_feature_names(rows)
    return {
        PredictionTarget.MINUTES: LoadedModel(
            model_version=await _model_version(session, PredictionTarget.MINUTES),
            booster=_Constant(30.0),
            feature_names=names,
            calibration_offset=0.0,
            target=PredictionTarget.MINUTES,
        ),
        PredictionTarget.PLAYS: LoadedModel(
            model_version=await _model_version(session, PredictionTarget.PLAYS),
            booster=_Constant(0.9),
            feature_names=names,
            calibration_offset=0.0,
            target=PredictionTarget.PLAYS,
        ),
    }


async def _minutes_by_side(
    session: AsyncSession, game_id: uuid.UUID
) -> list[tuple[float, float | None]]:
    return [
        (median, expected)
        for median, expected in (
            await session.execute(
                select(Prediction.predicted_value, Prediction.expected_value).where(
                    Prediction.game_id == game_id, Prediction.target == PredictionTarget.MINUTES
                )
            )
        ).all()
    ]


class TestApplied:
    async def test_each_side_is_scaled_to_the_target(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)
        reconciliation = MinutesReconciliation(
            alpha=1.0,
            target_minutes=60.0,
            cap=40.0,
            minutes_model_id=str(models[PredictionTarget.MINUTES].model_version.id),
            plays_model_id=str(models[PredictionTarget.PLAYS].model_version.id),
        )
        game = await _game(session, 6)
        await predict_game(
            session, game, models, clock=CLOCK, settings=settings, reconciliation=reconciliation
        )

        rows = await _minutes_by_side(session, game.id)
        # Partizan: 3 x 0.9 x 30 = 81 -> 60, so 22.22 each. Baskonia: 54 -> 60, 33.33.
        assert sorted(round(e, 2) for _, e in rows) == [22.22, 22.22, 22.22, 33.33, 33.33]
        # The median is never touched: intervals and scoring are built on it.
        assert all(m == 30.0 for m, _ in rows)

    async def test_a_fit_for_other_versions_is_skipped(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)
        stale = MinutesReconciliation(
            alpha=1.0,
            target_minutes=60.0,
            cap=40.0,
            minutes_model_id=str(uuid.uuid4()),
            plays_model_id=str(uuid.uuid4()),
        )
        game = await _game(session, 6)
        await predict_game(
            session, game, models, clock=CLOCK, settings=settings, reconciliation=stale
        )
        assert all(e is None for _, e in await _minutes_by_side(session, game.id))


class TestFit:
    async def test_fit_runs_end_to_end_and_saves(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)

        fit = await fit_minutes_reconciliation(
            session,
            models=models,
            settings=settings,
            # Games 1-3 to choose the strength, games 4 and 6 to judge it.
            fit_start=datetime(2025, 10, 1, tzinfo=UTC),
            test_start=datetime(2025, 10, 20, tzinfo=UTC),
            artifact_root=tmp_path,
        )

        saved = json.loads(reconciliation_path(tmp_path).read_text())
        assert saved["alpha"] == fit.alpha
        assert saved["minutes_model_id"] == str(models[PredictionTarget.MINUTES].model_version.id)
        assert len(saved["validation"]) == 11
        # Fixture squads log 75 (Partizan) or 54 (Baskonia, Madrid) minutes:
        # the target is measured from them, not assumed to be 200.
        assert 50.0 < fit.target_minutes < 80.0
        assert saved["enabled"] == fit.enabled

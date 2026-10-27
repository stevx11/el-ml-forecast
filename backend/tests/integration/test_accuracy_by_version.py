"""Live accuracy per model version, and two versions on the same player-games."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.config import Settings
from euroleague_ai.features.backfill import backfill_snapshots
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.ml.availability import derive_feature_names, load_availability_rows
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.outcomes import accuracy_by_version, head_to_head, score_predictions
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import ModelVersion
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db
from tests.integration.test_availability import AFTER_SEASON, CLOCK, _game, _seed_absences
from tests.integration.test_expectation import _Constant

pytestmark = [pytest.mark.integration, requires_db]


async def _version(
    session: AsyncSession,
    target: PredictionTarget,
    version: str,
    *,
    active: bool,
    trained_at: datetime,
) -> ModelVersion:
    row = ModelVersion(
        id=uuid.uuid4(),
        name=f"{str(target).lower()}_test",
        version=version,
        target=target,
        algorithm="lightgbm",
        feature_set_version=FEATURE_SET_VERSION,
        hyperparameters={},
        metrics={},
        artifact_path="/nonexistent/model.txt",
        is_active=active,
        trained_at=trained_at,
    )
    session.add(row)
    await session.flush()
    return row


async def _two_versions_on_game6(session: AsyncSession, settings: Settings) -> None:
    """Version 0.0.9 says 10 points for everyone, 0.0.10 (trained later) says 12."""
    await backfill_snapshots(session, clock=CLOCK, settings=settings)
    rows, _ = await load_availability_rows(session, settings=settings)
    names = derive_feature_names(rows)
    game = await _game(session, 6)
    # Trained in the opposite order to their names: the order is by training time.
    for version, value, active, day in (("0.0.9", 10.0, False, 1), ("0.0.10", 12.0, True, 2)):
        model = LoadedModel(
            model_version=await _version(
                session,
                PredictionTarget.POINTS,
                version,
                active=active,
                trained_at=datetime(2025, 9, day, tzinfo=UTC),
            ),
            booster=_Constant(value),
            feature_names=names,
            calibration_offset=0.0,
            target=PredictionTarget.POINTS,
        )
        await predict_game(
            session, game, {PredictionTarget.POINTS: model}, clock=CLOCK, settings=settings
        )


async def test_each_version_is_scored_on_its_own(
    session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef, settings: Settings
) -> None:
    await _seed_absences(session, fixture_provider, season)
    await _two_versions_on_game6(session, settings)
    await score_predictions(session, game_codes=[6], clock=AFTER_SEASON)

    rows = await accuracy_by_version(session)
    points = [r for r in rows if r["target"] == "POINTS"]
    assert [r["version"] for r in points] == ["0.0.9", "0.0.10"]
    assert [r["active"] for r in points] == [False, True]
    assert all(r["games"] == 1 and r["metric"] == "mae" for r in points)
    # Same players, predictions 2 points apart: the biases differ by exactly 2.
    assert points[1]["bias"] - points[0]["bias"] == pytest.approx(2.0)
    # Jovanovic did not appear: left out unless asked for.
    assert points[0]["scored"] == 4
    with_absent = await accuracy_by_version(session, include_absent=True)
    assert [r["scored"] for r in with_absent if r["target"] == "POINTS"] == [5, 5]

    pair = await head_to_head(
        session, target=PredictionTarget.POINTS, newer="0.0.10", older="0.0.9"
    )
    assert pair["shared"] == 4
    assert pair["newer_error"] == pytest.approx(points[1]["error"])
    assert pair["older_error"] == pytest.approx(points[0]["error"])

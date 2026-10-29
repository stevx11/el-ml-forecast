"""Injury labels moving P(plays): applied at prediction time, read back for the next fit.

Game 6 world from ``test_injury_check``: before the prediction the table calls
Hakansson ready and Dubois questionable. Every player gets P(plays) 0.9 from
the stand-in model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from euroleague_ai.ml.outcomes import score_predictions
from euroleague_ai.ml.status_calibration import StatusCalibration, adjusted, labelled_players
from euroleague_ai.models.enums import AvailabilityStatus, EntityType, PredictionTarget
from euroleague_ai.models.prediction import Prediction
from tests.conftest import requires_db
from tests.integration.test_availability import AFTER_SEASON, _id, _seed_absences
from tests.integration.test_expectation import _predict_game6
from tests.integration.test_injury_check import _report

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

CALIBRATION = StatusCalibration(offsets={"ready": 1.0, "questionable": -1.0}, enabled=True)


async def _plays(session: AsyncSession, code: str) -> Prediction:
    player_id = await _id(session, EntityType.PLAYER, code)
    return (
        await session.execute(
            select(Prediction).where(
                Prediction.player_id == player_id, Prediction.target == PredictionTarget.PLAYS
            )
        )
    ).scalar_one()


async def _world(
    session: AsyncSession,
    fixture_provider: FixtureProvider,
    season: SeasonRef,
    settings: Settings,
    calibration: StatusCalibration | None,
) -> None:
    await _seed_absences(session, fixture_provider, season)
    await _report(session, "013001", "BAS", AvailabilityStatus.AVAILABLE)  # Hakansson
    await _report(session, "013002", "BAS", AvailabilityStatus.QUESTIONABLE)  # Dubois
    await _predict_game6(session, settings, status_calibration=calibration)


class TestApplied:
    async def test_labels_move_p_plays_and_keep_the_models_number(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season, settings, CALIBRATION)

        ready = await _plays(session, "013001")
        assert ready.predicted_value == pytest.approx(adjusted(0.9, 1.0), abs=1e-4)
        assert ready.explanation["availability"]["status"] == "AVAILABLE"
        assert ready.explanation["availability"]["model_p_plays"] == 0.9
        assert ready.explanation["drivers"][0]["theme"] == "availability"

        questionable = await _plays(session, "013002")
        assert questionable.predicted_value == pytest.approx(adjusted(0.9, -1.0), abs=1e-4)

        # A player with no label is untouched.
        assert (await _plays(session, "011001")).predicted_value == pytest.approx(0.9)

    async def test_without_a_calibration_nothing_moves(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season, settings, None)
        ready = await _plays(session, "013001")
        assert ready.predicted_value == pytest.approx(0.9)
        # Still recorded, so a later fit can use it.
        assert ready.explanation["availability"]["model_p_plays"] == 0.9


class TestReadBack:
    async def test_the_next_fit_starts_from_the_models_number(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season, settings, CALIBRATION)
        await score_predictions(session, game_codes=[6], clock=AFTER_SEASON)

        players = await labelled_players(session)
        assert sorted((p.group, p.model_p, p.played) for p in players) == [
            ("questionable", 0.9, True),
            ("ready", 0.9, True),
        ]
        assert {p.round_key.split(":")[0] for p in players} == {season.season_code}

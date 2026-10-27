"""Expected values through the real schema: predict, store, score, total, serve.

Uses the same seeded world as ``test_availability``: Jovanovic is absent from
game 6, so the squad total for Partizan has to discount him through P(plays)
to come out right.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.config import Settings
from euroleague_ai.features.backfill import backfill_snapshots
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.ml.availability import derive_feature_names, load_availability_rows
from euroleague_ai.ml.distribution import ResidualDistribution
from euroleague_ai.ml.expectation import SOURCE_SHIFTED_MEDIAN
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.outcomes import accuracy_report, score_predictions, team_totals_report
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import ModelVersion, Prediction
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db
from tests.integration.test_availability import (
    AFTER_SEASON,
    CLOCK,
    _game,
    _seed_absences,
)

pytestmark = [pytest.mark.integration, requires_db]


class _Constant:
    """A booster stand-in: every player gets the same raw output."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: object, **kwargs: object) -> list[object]:
        rows = len(x)  # type: ignore[arg-type]
        if kwargs.get("pred_contrib"):
            # No feature moves anything; the base value is the whole prediction.
            width = x.shape[1]  # type: ignore[attr-defined]
            return [[0.0] * width + [self.value] for _ in range(rows)]
        return [self.value] * rows


async def _model_version(session: AsyncSession, target: PredictionTarget) -> ModelVersion:
    version = ModelVersion(
        id=uuid.uuid4(),
        name=f"{str(target).lower()}_test",
        version="0.0.1",
        target=target,
        algorithm="lightgbm",
        feature_set_version=FEATURE_SET_VERSION,
        hyperparameters={},
        metrics={},
        artifact_path="/nonexistent/model.txt",
        is_active=True,
    )
    session.add(version)
    await session.flush()
    return version


async def _predict_game6(session: AsyncSession, settings: Settings) -> uuid.UUID:
    """POINTS: median 10, skewed residuals (+2 on average). PLAYS: 0.9 for all."""
    await backfill_snapshots(session, clock=CLOCK, settings=settings)
    rows, _ = await load_availability_rows(session, settings=settings)
    names = derive_feature_names(rows)

    points = LoadedModel(
        model_version=await _model_version(session, PredictionTarget.POINTS),
        booster=_Constant(10.0),
        feature_names=names,
        calibration_offset=0.0,
        target=PredictionTarget.POINTS,
        distribution=ResidualDistribution(
            level_edges=[], cells={}, by_history={}, pooled=[-2.0, -1.0, 0.0, 1.0, 12.0]
        ),
        expected_source=SOURCE_SHIFTED_MEDIAN,
    )
    plays = LoadedModel(
        model_version=await _model_version(session, PredictionTarget.PLAYS),
        booster=_Constant(0.9),
        feature_names=names,
        calibration_offset=0.0,
        target=PredictionTarget.PLAYS,
    )
    game = await _game(session, 6)
    await predict_game(
        session,
        game,
        {PredictionTarget.POINTS: points, PredictionTarget.PLAYS: plays},
        clock=CLOCK,
        settings=settings,
    )
    return game.id


class TestStoredAndScored:
    async def test_expected_values_are_written_beside_the_median(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        game_id = await _predict_game6(session, settings)

        rows = (
            await session.execute(
                select(
                    Prediction.target, Prediction.predicted_value, Prediction.expected_value
                ).where(Prediction.game_id == game_id)
            )
        ).all()
        points = [r for r in rows if r[0] is PredictionTarget.POINTS]
        assert len(points) == 5
        assert all(
            median == 10.0 and expected == pytest.approx(12.0) for _, median, expected in points
        )
        assert all(
            expected is None for target, _, expected in rows if target is PredictionTarget.PLAYS
        )

    async def test_team_totals_compare_median_and_expected_with_the_box_score(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await _predict_game6(session, settings)
        await score_predictions(session, game_codes=[6], clock=AFTER_SEASON)

        totals = await team_totals_report(session)
        points = totals["POINTS"]
        # Game 6: Partizan scored 60 + 18 after Jovanovic was removed (78), Baskonia 28.
        # Median: PAR 3 x 10 = 30 (-48), BAS 2 x 10 = 20 (-8)    -> bias -28
        # Expected: PAR 3 x 0.9 x 12 = 32.4, BAS 2 x 0.9 x 12 = 21.6 -> bias -26
        assert points["median"] == {"team_games": 2, "mae": 28.0, "bias": -28.0}
        assert points["expected"]["bias"] == pytest.approx(-26.0)

        report = await accuracy_report(session)
        expected = report["targets"]["POINTS"]["expected"]
        assert expected["scored"] == 4  # Jovanovic did not appear

        # Overall P(X+): every prediction counts, Jovanovic's included, at
        # 0.9 x the conditional chance; the conditional table leaves him out.
        bars = len(
            (
                await session.execute(
                    select(Prediction.threshold_probabilities).where(
                        Prediction.target == PredictionTarget.POINTS
                    )
                )
            )
            .scalars()
            .first()
            or {}
        )
        assert bars > 0
        conditional = sum(b["predictions"] for b in report["thresholds"])
        overall = sum(b["predictions"] for b in report["thresholds_overall"])
        assert (conditional, overall) == (4 * bars, 5 * bars)
        assert report["team_totals"]["POINTS"]["expected"]["team_games"] == 2


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    from euroleague_ai.db.session import get_session
    from euroleague_ai.main import create_app

    app = create_app()

    async def override() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


class TestApi:
    async def test_projection_carries_expected_lines_and_squad_totals(
        self,
        session: AsyncSession,
        client: AsyncClient,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        game_id = await _predict_game6(session, settings)

        response = await client.get(f"/api/v1/games/{game_id}/predictions")
        assert response.status_code == 200
        body = response.json()

        player = body["players"][0]
        points = player["projections"]["POINTS"]
        assert points["value"] == 10.0
        assert points["expected"] == pytest.approx(12.0)
        assert points["expected_overall"] == pytest.approx(10.8)
        # P(X+) counting the 10% chance he sits: 0.9 x the "if he plays" chance.
        conditional = points["threshold_probabilities"]
        assert conditional, "the skewed distribution publishes threshold chances"
        assert points["threshold_probabilities_overall"] == {
            bar: pytest.approx(0.9 * p, abs=1e-4) for bar, p in conditional.items()
        }
        assert body["team_totals"]["PAR"]["POINTS"] == pytest.approx(32.4)
        assert body["team_totals"]["BAS"]["POINTS"] == pytest.approx(21.6)
        assert "PLAYS" not in body["team_totals"]["PAR"]

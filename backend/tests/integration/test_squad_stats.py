"""Squad stat reconciliation through the real schema: serving and the fit's plumbing.

Same world as ``test_expectation``: POINTS median 10, expected 12, P(plays) 0.9.
Partizan field three candidates in game 6, Baskonia two.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from euroleague_ai.features.backfill import backfill_snapshots
from euroleague_ai.ml.availability import derive_feature_names, load_availability_rows
from euroleague_ai.ml.distribution import ResidualDistribution
from euroleague_ai.ml.expectation import SOURCE_SHIFTED_MEDIAN
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.squad_stats import (
    FACTOR_BOUNDS,
    STAT_TARGETS,
    StatsReconciliation,
    fit_stats_reconciliation,
)
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import Prediction
from tests.conftest import requires_db
from tests.integration.test_availability import CLOCK, _game, _seed_absences
from tests.integration.test_expectation import _Constant, _model_version

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

SKEWED = ResidualDistribution(
    level_edges=[], cells={}, by_history={}, pooled=[-2.0, -1.0, 0.0, 1.0, 12.0]
)


async def _models(session: AsyncSession, settings: Settings) -> dict[PredictionTarget, LoadedModel]:
    await backfill_snapshots(session, clock=CLOCK, settings=settings)
    rows, _ = await load_availability_rows(session, settings=settings)
    names = derive_feature_names(rows)
    models = {
        target: LoadedModel(
            model_version=await _model_version(session, target),
            booster=_Constant(10.0),
            feature_names=names,
            calibration_offset=0.0,
            target=target,
            distribution=SKEWED,
            expected_source=SOURCE_SHIFTED_MEDIAN,
        )
        for target in STAT_TARGETS
    }
    models[PredictionTarget.PLAYS] = LoadedModel(
        model_version=await _model_version(session, PredictionTarget.PLAYS),
        booster=_Constant(0.9),
        feature_names=names,
        calibration_offset=0.0,
        target=PredictionTarget.PLAYS,
    )
    return models


class TestServing:
    async def test_expected_points_move_toward_the_anchor_one_factor_per_side(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)
        game = await _game(session, 6)
        # Anchor: 1.0 x the game's expected team points; pace ignored.
        fit = StatsReconciliation(
            targets={PredictionTarget.POINTS: (1.0, [0.0, 0.0, 1.0, 0.0])}, model_ids={}
        )
        await predict_game(
            session, game, models, clock=CLOCK, settings=settings, stats_reconciliation=fit
        )

        rows = (
            await session.execute(
                select(
                    Prediction.target,
                    Prediction.predicted_value,
                    Prediction.expected_value,
                    Prediction.explanation,
                ).where(Prediction.game_id == game.id)
            )
        ).all()
        points = [r for r in rows if r[0] is PredictionTarget.POINTS]
        factors = {round(r[3]["squad_factor"], 6) for r in points}
        assert len(factors) <= 2  # one per side
        lo, hi = FACTOR_BOUNDS
        for _, median, expected, explanation in points:
            assert median == 10.0  # medians untouched
            assert expected == pytest.approx(12.0 * explanation["squad_factor"], abs=1e-3)
            assert lo <= explanation["squad_factor"] <= hi
        assert factors != {1.0}, "the fixture has ratings, so the anchor should apply"
        # Assists were not in the fit: unchanged.
        assists = [r for r in rows if r[0] is PredictionTarget.ASSISTS]
        assert all(
            r[2] == pytest.approx(12.0) and "squad_factor" not in (r[3] or {}) for r in assists
        )

    async def test_stale_fits_are_ignored(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)
        fit = StatsReconciliation(
            targets={PredictionTarget.POINTS: (1.0, [0.0, 0.0, 1.0, 0.0])},
            model_ids={"POINTS": "00000000-0000-0000-0000-000000000000"},
        )
        assert not fit.applies_to(models)


class TestFit:
    async def test_small_history_fits_nothing_but_saves_a_complete_report(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        models = await _models(session, settings)
        g3, g5 = await _game(session, 3), await _game(session, 5)
        result = await fit_stats_reconciliation(
            session,
            models=models,
            settings=settings,
            fit_start=g3.tip_off_utc,
            test_start=g5.tip_off_utc,
            artifact_root=tmp_path,
        )
        saved = json.loads((tmp_path / "reconciliation" / "stats.json").read_text())
        assert set(saved["targets"]) == {t.name for t in STAT_TARGETS}
        assert set(saved["model_ids"]) == {*(t.name for t in STAT_TARGETS), "PLAYS"}
        # A handful of squads is far too few for a trustworthy anchor.
        assert not any(f.enabled for f in result.fits.values())
        assert all(f.note for f in result.fits.values())

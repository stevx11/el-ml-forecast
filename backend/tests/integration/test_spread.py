"""The spread check through the real schema: active-style model, test period."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.ml.spread import target_spread
from euroleague_ai.models.enums import PredictionTarget
from tests.conftest import requires_db
from tests.integration.test_availability import _seed_absences
from tests.integration.test_squad_stats import _models

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]


async def test_constant_projections_have_no_slope_and_thin_tails(
    session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef, settings: Settings
) -> None:
    await _seed_absences(session, fixture_provider, season)
    models = await _models(session, settings)
    report = await target_spread(
        session,
        PredictionTarget.POINTS,
        models[PredictionTarget.POINTS],
        feature_set_version=FEATURE_SET_VERSION,
    )
    assert report.rows > 0
    # Everyone projected the same: nothing to regress on.
    assert report.verdict == "unknown"
    assert report.std["median"] == 0.0
    assert report.quantiles["expected"]["0.5"] == pytest.approx(12.0)
    assert report.big_nights, "the skewed distribution publishes threshold chances"

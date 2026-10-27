"""Injury reports through the real schema: record, resolve as of a cutoff, predict.

Fixture calendar (Partizan): game 1 Oct 3, 2 Oct 10, 3 Oct 17, 4 Oct 24,
game 5 Oct 31 (scheduled), game 6 Nov 7. Tip-offs are 18:00 UTC and the
prediction cutoff is an hour earlier.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.core.config import Settings
from euroleague_ai.ingestion.injuries import (
    InjuryInputError,
    current_rosters,
    import_reports,
    record_report,
)
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.reconcile import MinutesReconciliation
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, EntityType, PredictionTarget
from euroleague_ai.models.prediction import Prediction
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db
from tests.integration.test_availability import _game, _id
from tests.integration.test_expectation import _Constant, _model_version

pytestmark = [pytest.mark.integration, requires_db]

INGEST = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))
MONDAY_BEFORE_GAME5 = datetime(2025, 10, 27, 10, 0, tzinfo=UTC)


async def _world(session: AsyncSession, provider: FixtureProvider, season: SeasonRef) -> None:
    await ingest_all(session, provider, season, clock=INGEST)
    await session.flush()


async def _models(session: AsyncSession) -> dict[PredictionTarget, LoadedModel]:
    """Minutes 30 for everyone, P(plays) 0.9 for everyone. Features are irrelevant."""
    return {
        PredictionTarget.MINUTES: LoadedModel(
            model_version=await _model_version(session, PredictionTarget.MINUTES),
            booster=_Constant(30.0),
            feature_names=["days_rest"],
            calibration_offset=0.0,
            target=PredictionTarget.MINUTES,
        ),
        PredictionTarget.PLAYS: LoadedModel(
            model_version=await _model_version(session, PredictionTarget.PLAYS),
            booster=_Constant(0.9),
            feature_names=["days_rest"],
            calibration_offset=0.0,
            target=PredictionTarget.PLAYS,
        ),
    }


async def _record(
    session: AsyncSession, player: str, status: str, *, at: datetime, **kw: object
) -> None:
    roster = await current_rosters(session, as_of=at)
    await record_report(session, roster=roster, player=player, status=status, now=at, **kw)  # type: ignore[arg-type]
    await session.flush()


async def _plays(session: AsyncSession, game_code: int) -> dict[str, tuple[float, dict | None]]:
    game = await _game(session, game_code)
    rows = (
        await session.execute(
            select(Prediction.player_id, Prediction.predicted_value, Prediction.explanation).where(
                Prediction.game_id == game.id, Prediction.target == PredictionTarget.PLAYS
            )
        )
    ).all()
    names = {
        await _id(session, EntityType.PLAYER, code): code
        for code in ("011001", "011002", "011003", "012001", "012002", "012003")
    }
    return {
        names[player_id]: (value, (explanation or {}).get("availability"))
        for player_id, value, explanation in rows
        if player_id in names
    }


class TestRecording:
    async def test_default_scope_is_the_next_game(
        self, session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
    ) -> None:
        await _world(session, fixture_provider, season)
        await _record(session, "Jovanovic", "out", at=MONDAY_BEFORE_GAME5, reason="knee")
        event = (await session.execute(select(PlayerStatusEvent))).scalar_one()
        assert event.status is AvailabilityStatus.OUT
        assert event.game_id == (await _game(session, 5)).id
        assert event.observed_at == MONDAY_BEFORE_GAME5

    async def test_a_bad_row_means_nothing_is_imported(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        tmp_path: Path,
    ) -> None:
        await _world(session, fixture_provider, season)
        csv_path = tmp_path / "round.csv"
        csv_path.write_text(
            "player,status,team,reason\nJovanovic,out,PAR,knee\nNobody,out,,\nGarcia,maybe,,\n"
        )
        with pytest.raises(InjuryInputError) as raised:
            await import_reports(session, csv_path, now=MONDAY_BEFORE_GAME5)
        assert "line 3" in str(raised.value) and "line 4" in str(raised.value)
        assert (await session.execute(select(PlayerStatusEvent))).scalars().all() == []

        csv_path.write_text(
            "player,status,team,reason,until\nJovanovic,out,PAR,knee,2025-11-20\nGarcia,gtd,MAD,ankle,\n"
        )
        recorded = await import_reports(session, csv_path, now=MONDAY_BEFORE_GAME5)
        assert [r.scope for r in recorded] == ["until", "game"]


class TestPredictions:
    async def test_ruled_out_means_zero_and_questionable_is_only_flagged(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        await _record(session, "Jovanovic", "out", at=MONDAY_BEFORE_GAME5, reason="knee")
        await _record(session, "Garcia", "questionable", at=MONDAY_BEFORE_GAME5)
        models = await _models(session)
        await predict_game(
            session, await _game(session, 5), models, clock=INGEST, settings=settings
        )

        plays = await _plays(session, 5)
        assert plays["011002"][0] == 0.0
        assert plays["011002"][1]["status"] == "OUT"
        assert plays["012001"][0] == pytest.approx(0.9)
        assert plays["012001"][1]["status"] == "QUESTIONABLE"
        assert plays["011001"] == (pytest.approx(0.9), None)

    async def test_teammates_absorb_the_minutes_of_a_ruled_out_player(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        await _record(session, "Jovanovic", "out", at=MONDAY_BEFORE_GAME5)
        models = await _models(session)
        reconciliation = MinutesReconciliation(
            alpha=1.0,
            target_minutes=60.0,
            cap=40.0,
            minutes_model_id=str(models[PredictionTarget.MINUTES].model_version.id),
            plays_model_id=str(models[PredictionTarget.PLAYS].model_version.id),
        )
        game = await _game(session, 5)
        await predict_game(
            session, game, models, clock=INGEST, settings=settings, reconciliation=reconciliation
        )
        petrovic = await _id(session, EntityType.PLAYER, "011001")
        expected = (
            await session.execute(
                select(Prediction.expected_value).where(
                    Prediction.game_id == game.id,
                    Prediction.player_id == petrovic,
                    Prediction.target == PredictionTarget.MINUTES,
                )
            )
        ).scalar_one()
        # Partizan without Jovanovic: 2 x 0.9 x 30 = 54 weighted minutes -> 60.
        assert expected == pytest.approx(60.0 / 54.0 * 30.0, abs=0.01)

    async def test_a_report_entered_after_the_cutoff_is_invisible(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        game = await _game(session, 5)
        await _record(session, "Petrovic", "out", at=game.tip_off_utc - timedelta(minutes=30))
        await predict_game(session, game, await _models(session), clock=INGEST, settings=settings)
        assert (await _plays(session, 5))["011001"] == (pytest.approx(0.9), None)

    async def test_an_open_ended_report_ends_when_he_plays_again(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _world(session, fixture_provider, season)
        # Between games 1 and 2. He then plays games 2, 3 and 4.
        await _record(
            session, "Petrovic", "out", at=datetime(2025, 10, 5, 9, 0, tzinfo=UTC), indefinite=True
        )
        models = await _models(session)
        await predict_game(
            session, await _game(session, 2), models, clock=INGEST, settings=settings
        )
        await predict_game(
            session, await _game(session, 6), models, clock=INGEST, settings=settings
        )
        assert (await _plays(session, 2))["011001"][0] == 0.0
        assert (await _plays(session, 6))["011001"] == (pytest.approx(0.9), None)

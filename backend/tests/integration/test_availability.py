"""The play-probability model end to end against the real schema.

The fixture world has everyone playing every game, so two absences are seeded:

* game 6: Jovanovic (``011002``) is **not in Partizan's squad at all**. His box
  score row is deleted, which is exactly what an injury looks like upstream.
* game 4: Nikolic (``011003``) **dresses and is never used**.

Then the full path is exercised: backfill (with and without candidate
snapshots), the availability dataset, serving a P(plays), and scoring it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.core.config import Settings
from euroleague_ai.features.backfill import backfill_snapshots
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.ml.availability import (
    AvailabilityRow,
    derive_feature_names,
    load_availability_rows,
)
from euroleague_ai.ml.dataset import load_training_rows
from euroleague_ai.ml.inference import LoadedModel, predict_game
from euroleague_ai.ml.matrix import features_to_matrix
from euroleague_ai.ml.outcomes import availability_report, score_predictions
from euroleague_ai.models.enums import EntityType, PredictionTarget
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.models.prediction import ModelVersion, Prediction, PredictionOutcome
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

CLOCK = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))
AFTER_SEASON = FixedClock(datetime(2025, 12, 1, 12, 0, tzinfo=UTC))

LEFT_OUT = "011002"  # absent from game 6
UNUSED = "011003"  # dressed, never used in game 4


async def _id(session: AsyncSession, entity: EntityType, external_id: str) -> uuid.UUID:
    return (
        await session.execute(
            select(ExternalRef.internal_id).where(
                ExternalRef.entity_type == entity, ExternalRef.external_id == external_id
            )
        )
    ).scalar_one()


async def _game(session: AsyncSession, code: int) -> Game:
    return (await session.execute(select(Game).where(Game.game_code == code))).scalar_one()


async def _line_filter(session: AsyncSession, game_code: int, player_code: str) -> object:
    game = await _game(session, game_code)
    player_id = await _id(session, EntityType.PLAYER, player_code)
    team_id = await _id(session, EntityType.TEAM, "PAR")
    game_team_id = (
        await session.execute(
            select(GameTeam.id).where(GameTeam.game_id == game.id, GameTeam.team_id == team_id)
        )
    ).scalar_one()
    return (PlayerGameStats.game_team_id == game_team_id) & (PlayerGameStats.player_id == player_id)


async def _seed_absences(
    session: AsyncSession, provider: FixtureProvider, season: SeasonRef
) -> None:
    await ingest_all(session, provider, season, clock=CLOCK)
    await session.flush()
    await session.execute(
        delete(PlayerGameStats).where(await _line_filter(session, 6, LEFT_OUT))  # type: ignore[arg-type]
    )
    await session.execute(
        update(PlayerGameStats)
        .where(await _line_filter(session, 4, UNUSED))  # type: ignore[arg-type]
        .values(did_not_play=True, seconds_played=0)
    )
    await session.flush()


async def _find(
    session: AsyncSession, rows: list[AvailabilityRow], game_code: int, player_code: str
) -> AvailabilityRow:
    game = await _game(session, game_code)
    player_id = await _id(session, EntityType.PLAYER, player_code)
    matches = [r for r in rows if r.game_id == game.id and r.player_id == player_id]
    assert len(matches) == 1, f"expected one row for {player_code} in game {game_code}"
    return matches[0]


class TestDataset:
    async def test_absent_and_unused_players_become_negative_rows(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await backfill_snapshots(session, clock=CLOCK, settings=settings)

        rows, diagnostics = await load_availability_rows(session, settings=settings)

        left_out = await _find(session, rows, 6, LEFT_OUT)
        assert left_out.played is False
        assert left_out.dressed is False

        unused = await _find(session, rows, 4, UNUSED)
        assert unused.played is False
        assert unused.dressed is True

        assert diagnostics.missing_snapshots == 0
        assert diagnostics.wrong_side == 0
        assert diagnostics.not_in_squad == 1
        assert diagnostics.dressed_not_used == 1
        assert diagnostics.played_not_candidate == 0
        assert diagnostics.rows == diagnostics.played + 2

    async def test_recent_appearance_counts_see_only_the_past(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await backfill_snapshots(session, clock=CLOCK, settings=settings)
        rows, _ = await load_availability_rows(session, settings=settings)

        # Before game 6, Partizan had played games 1-4; Nikolic sat out game 4.
        nikolic = await _find(session, rows, 6, UNUSED)
        assert (nikolic.recent_played, nikolic.recent_team_games) == (3, 4)
        # In round one nobody has played yet, so everyone is undebuted.
        assert (await _find(session, rows, 1, UNUSED)).not_debuted is True

    async def test_without_candidate_snapshots_the_absent_player_has_no_row(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        """The reason the backfill change exists: a box-score-only backfill has
        no snapshot for an injured player, so the zero can never be learned."""
        await _seed_absences(session, fixture_provider, season)
        await backfill_snapshots(session, clock=CLOCK, settings=settings, include_candidates=False)
        _, diagnostics = await load_availability_rows(session, settings=settings)
        assert diagnostics.missing_snapshots == 1

    async def test_stat_datasets_are_unchanged_by_candidate_rows(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await backfill_snapshots(session, clock=CLOCK, settings=settings)

        rows = await load_training_rows(
            session,
            target=PredictionTarget.POINTS,
            feature_set_version=FEATURE_SET_VERSION,
        )
        game6 = await _game(session, 6)
        game4 = await _game(session, 4)
        absent = await _id(session, EntityType.PLAYER, LEFT_OUT)
        unused = await _id(session, EntityType.PLAYER, UNUSED)
        assert not [r for r in rows if r.game_id == game6.id and r.player_id == absent]
        assert not [r for r in rows if r.game_id == game4.id and r.player_id == unused]


class TestServingAndScoring:
    async def test_plays_is_predicted_scored_and_reported(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        import lightgbm as lgb
        import numpy as np

        await _seed_absences(session, fixture_provider, season)
        await backfill_snapshots(session, clock=CLOCK, settings=settings)
        rows, _ = await load_availability_rows(session, settings=settings)

        # A tiny real booster over the real feature names. Its accuracy is
        # irrelevant here; what is under test is the plumbing around it.
        names = derive_feature_names(rows)
        booster = lgb.train(
            {
                "objective": "binary",
                "min_data_in_leaf": 1,
                "min_data_in_bin": 1,
                "verbosity": -1,
                "seed": 1,
            },
            lgb.Dataset(
                features_to_matrix([r.features for r in rows], names),
                label=np.asarray([r.label for r in rows]),
            ),
            num_boost_round=5,
        )
        model_path = tmp_path / "model.txt"
        booster.save_model(str(model_path))
        model_version = ModelVersion(
            id=uuid.uuid4(),
            name="plays_lgbm",
            version="0.0.1",
            target=PredictionTarget.PLAYS,
            algorithm="lightgbm",
            feature_set_version=FEATURE_SET_VERSION,
            hyperparameters={},
            metrics={},
            artifact_path=str(model_path),
            is_active=True,
        )
        session.add(model_version)
        await session.flush()

        model = LoadedModel(
            model_version=model_version,
            booster=booster,
            feature_names=names,
            calibration_offset=0.0,
            target=PredictionTarget.PLAYS,
        )
        game6 = await _game(session, 6)
        written = await predict_game(
            session, game6, {PredictionTarget.PLAYS: model}, clock=CLOCK, settings=settings
        )
        assert written == 5  # three Partizan candidates, two Baskonia

        predictions = (
            (
                await session.execute(
                    select(Prediction).where(
                        Prediction.game_id == game6.id,
                        Prediction.target == PredictionTarget.PLAYS,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(predictions) == 5
        for prediction in predictions:
            assert 0.0 <= prediction.predicted_value <= 1.0
            assert prediction.quantiles is None
            assert (prediction.explanation or {}).get("scale") == "log_odds"

        await score_predictions(session, game_codes=[6], clock=AFTER_SEASON)
        outcomes = {
            player_id: (actual, absent)
            for player_id, actual, absent in (
                await session.execute(
                    select(
                        Prediction.player_id,
                        PredictionOutcome.actual_value,
                        PredictionOutcome.did_not_appear,
                    )
                    .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
                    .where(Prediction.target == PredictionTarget.PLAYS)
                )
            ).all()
        }
        left_out = await _id(session, EntityType.PLAYER, LEFT_OUT)
        assert outcomes[left_out] == (0.0, True)
        assert sorted(actual for actual, _ in outcomes.values()) == [0.0, 1.0, 1.0, 1.0, 1.0]

        report = await availability_report(session)
        assert report["scored"] == 5
        assert report["base_rate"] == pytest.approx(0.8)
        assert report["reliability"]

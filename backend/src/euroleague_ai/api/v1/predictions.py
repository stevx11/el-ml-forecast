"""Prediction endpoints.

Mounted only now that models exist. An endpoint returning an empty list because
the feature was never built is worse than no endpoint -- a client cannot tell
"no data" from "not implemented".
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from euroleague_ai.api.deps import SessionDep
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.game import Game
from euroleague_ai.models.people import Player
from euroleague_ai.models.prediction import ModelVersion, Prediction
from euroleague_ai.schemas.prediction import (
    PlayerProjectionOut,
    PredictionOut,
    ProjectionsOut,
)

router = APIRouter(tags=["predictions"])


async def _load(session: SessionDep, **filters: object) -> list[tuple]:
    stmt = (
        select(
            Prediction,
            Player.full_name,
            ModelVersion.name,
            ModelVersion.version,
        )
        .join(Player, Player.id == Prediction.player_id)
        .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
        .order_by(Prediction.prediction_timestamp.desc())
    )
    for column, value in filters.items():
        stmt = stmt.where(getattr(Prediction, column) == value)
    return list((await session.execute(stmt)).all())


@router.get("/games/{game_id}/predictions", response_model=ProjectionsOut)
async def game_projections(game_id: uuid.UUID, session: SessionDep) -> ProjectionsOut:
    """Projected lines for every predicted player in a game.

    One row per player with each target as a column, which is how a human reads
    a projection -- the flat per-target rows are available on the player
    endpoint.
    """
    game = (await session.execute(select(Game).where(Game.id == game_id))).scalar_one_or_none()
    if game is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Game not found")

    rows = await _load(session, game_id=game_id)

    projections: dict[uuid.UUID, PlayerProjectionOut] = {}
    for prediction, player_name, model_name, model_version in rows:
        projection = projections.setdefault(
            prediction.player_id,
            PlayerProjectionOut(
                player_id=prediction.player_id,
                player_name=player_name,
                prediction_timestamp=prediction.prediction_timestamp,
                targets={},
                models={},
            ),
        )
        projection.targets[str(prediction.target)] = prediction.predicted_value
        projection.models[str(prediction.target)] = f"{model_name} {model_version}"

    return ProjectionsOut(
        game_id=game_id,
        tip_off_utc=game.tip_off_utc,
        # Most minutes first: the players a reader cares about are at the top.
        players=sorted(
            projections.values(),
            key=lambda p: -(p.targets.get(str(PredictionTarget.MINUTES)) or 0.0),
        ),
    )


@router.get("/players/{player_id}/predictions", response_model=list[PredictionOut])
async def player_predictions(
    player_id: uuid.UUID,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[PredictionOut]:
    """A player's most recent predictions, newest first."""
    rows = await _load(session, player_id=player_id)
    return [
        PredictionOut(
            id=prediction.id,
            player_id=prediction.player_id,
            player_name=player_name,
            game_id=prediction.game_id,
            target=prediction.target,
            predicted_value=prediction.predicted_value,
            prediction_timestamp=prediction.prediction_timestamp,
            model=f"{model_name} {model_version}",
        )
        for prediction, player_name, model_name, model_version in rows[:limit]
    ]

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
    DriverOut,
    PlayerProjectionOut,
    PredictionOut,
    ProjectionsOut,
    TargetProjectionOut,
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

    home, away = await _team_labels(session, game)
    rows = await _load(session, game_id=game_id)
    # Every model shares one cutoff per game, so prediction_timestamp alone
    # cannot order them. A newer model's rows were written later, so the row's
    # own created_at breaks the tie.
    rows.sort(
        key=lambda row: (row[0].prediction_timestamp, row[0].created_at),
        reverse=True,
    )

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
        target = str(prediction.target)
        if target in projection.targets:
            # Rows arrive newest first. Keep the newest per target; an older
            # model version's projection must not overwrite the current one.
            continue
        if prediction.target is PredictionTarget.PLAYS:
            projection.play_probability = prediction.predicted_value
        projection.targets[target] = prediction.predicted_value
        projection.models[target] = f"{model_name} {model_version}"
        projection.projections[target] = TargetProjectionOut(
            value=prediction.predicted_value,
            model=f"{model_name} {model_version}",
            quantiles=prediction.quantiles,
            threshold_probabilities=prediction.threshold_probabilities,
            predicted_std=prediction.predicted_std,
            confidence=(
                float(prediction.confidence) if prediction.confidence is not None else None
            ),
            drivers=[
                DriverOut(**driver)
                for driver in (prediction.explanation or {}).get("drivers", [])
            ],
        )

    await _attach_teams(session, game, projections)

    return ProjectionsOut(
        game_id=game_id,
        tip_off_utc=game.tip_off_utc,
        home_team=home[0],
        away_team=away[0],
        home_team_code=home[1],
        away_team_code=away[1],
        # Most minutes first: the players a reader cares about are at the top.
        players=sorted(
            projections.values(),
            key=lambda p: -(p.targets.get(str(PredictionTarget.MINUTES)) or 0.0),
        ),
    )


async def _team_labels(session: SessionDep, game: Game) -> tuple[tuple[str, str], tuple[str, str]]:
    """Home and away ``(name, code)``, for the header."""
    from euroleague_ai.models.competition import Team

    rows = (
        await session.execute(
            select(Team.id, Team.name, Team.code).where(
                Team.id.in_([game.home_team_id, game.away_team_id])
            )
        )
    ).all()
    lookup = {team_id: (name, code) for team_id, name, code in rows}
    return (
        lookup.get(game.home_team_id, ("Home", "")),
        lookup.get(game.away_team_id, ("Away", "")),
    )


async def _attach_teams(
    session: SessionDep, game: Game, projections: dict[uuid.UUID, PlayerProjectionOut]
) -> None:
    """Tag each projection with the club it was made for.

    Read from the feature snapshot rather than re-resolved here. The snapshot
    records the team the prediction was actually built against, so the UI splits
    the squads exactly as inference saw them -- re-deriving it now could disagree
    after a transfer, and quietly show a player on the wrong side.
    """
    from euroleague_ai.models.competition import Team
    from euroleague_ai.models.features import PlayerFeatureSnapshot

    rows = (
        await session.execute(
            select(PlayerFeatureSnapshot.player_id, Team.id, Team.code)
            .join(Team, Team.id == PlayerFeatureSnapshot.team_id)
            .where(PlayerFeatureSnapshot.game_id == game.id)
        )
    ).all()
    for player_id, team_id, team_code in rows:
        projection = projections.get(player_id)
        if projection is not None and projection.team_id is None:
            projection.team_id = team_id
            projection.team_code = team_code


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

"""Endpoints shaped for the web client.

Separate from the resource endpoints on purpose. ``/games`` and ``/predictions``
model the domain; these model a screen. A fixtures list that needs three round
trips to render one row is a bad API for a browser however clean it looks as a
resource tree, and denormalising here keeps that pressure off the domain
endpoints.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from euroleague_ai.api.deps import SessionDep
from euroleague_ai.core.clock import SystemClock
from euroleague_ai.models.competition import Season, Team
from euroleague_ai.models.enums import GameStatus
from euroleague_ai.models.game import Game
from euroleague_ai.models.prediction import ModelVersion, Prediction

router = APIRouter(tags=["board"])


@router.get("/board/fixtures")
async def fixtures(
    session: SessionDep,
    season: str | None = Query(default=None, description='Season code, e.g. "E2026".'),
    days: int = Query(default=14, ge=1, le=365),
    include_past: int = Query(
        default=3, ge=0, le=30, description="Days of finished games to keep visible."
    ),
) -> dict[str, Any]:
    """Fixtures around now, each with whether it has been projected yet.

    The prediction count is the important field. Without it the client cannot
    tell a fixture nobody has projected from one where every player was
    filtered out, and those need different empty states.
    """
    now = SystemClock().now()
    home, away = Team.__table__.alias("home"), Team.__table__.alias("away")

    predicted = (
        select(Prediction.game_id, func.count().label("predictions"))
        .group_by(Prediction.game_id)
        .subquery()
    )

    stmt = (
        select(
            Game.id,
            Game.game_code,
            Game.round_number,
            Game.tip_off_utc,
            Game.status,
            Game.home_score,
            Game.away_score,
            home.c.name,
            home.c.code,
            away.c.name,
            away.c.code,
            Season.code,
            func.coalesce(predicted.c.predictions, 0),
        )
        .join(Season, Season.id == Game.season_id)
        .join(home, home.c.id == Game.home_team_id)
        .join(away, away.c.id == Game.away_team_id)
        .outerjoin(predicted, predicted.c.game_id == Game.id)
        .where(
            Game.tip_off_utc >= now - timedelta(days=include_past),
            Game.tip_off_utc <= now + timedelta(days=days),
        )
        .order_by(Game.tip_off_utc)
    )
    if season:
        stmt = stmt.where(Season.code == season)

    rows = (await session.execute(stmt)).all()
    return {
        "now": now.isoformat(),
        "window_days": days,
        "fixtures": [
            {
                "game_id": str(row[0]),
                "game_code": row[1],
                "round": row[2],
                "tip_off_utc": row[3].isoformat(),
                "status": str(row[4]),
                "home_score": row[5],
                "away_score": row[6],
                "home_team": row[7],
                "home_code": row[8],
                "away_team": row[9],
                "away_code": row[10],
                "season": row[11],
                "predictions": row[12],
            }
            for row in rows
        ],
    }


@router.get("/board/seasons")
async def seasons(session: SessionDep) -> dict[str, Any]:
    """Seasons that hold fixtures, newest first, for the season picker."""
    stmt = (
        select(Season.code, Season.name, func.count(Game.id))
        .join(Game, Game.season_id == Season.id)
        .group_by(Season.code, Season.name, Season.start_year)
        .order_by(Season.start_year.desc())
    )
    rows = (await session.execute(stmt)).all()
    return {"seasons": [{"code": code, "name": name, "games": games} for code, name, games in rows]}


@router.get("/board/status")
async def status(session: SessionDep) -> dict[str, Any]:
    """What the system currently knows, for the header and the empty states.

    Every one of these numbers exists to let the client explain *why* a screen is
    empty. "No projections yet" and "no models trained" look identical to a user
    and need completely different advice.
    """
    now = SystemClock().now()

    upcoming = (
        await session.execute(
            select(func.count(Game.id)).where(
                Game.status == GameStatus.SCHEDULED, Game.tip_off_utc >= now
            )
        )
    ).scalar_one()

    predictions = (await session.execute(select(func.count(Prediction.id)))).scalar_one()
    latest = (await session.execute(select(func.max(Prediction.prediction_timestamp)))).scalar_one()

    models = (
        await session.execute(
            select(ModelVersion.target, ModelVersion.name, ModelVersion.version)
            .where(ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.target)
        )
    ).all()

    return {
        "now": now.isoformat(),
        "upcoming_games": int(upcoming or 0),
        "predictions": int(predictions or 0),
        "latest_prediction": latest.isoformat() if latest else None,
        "active_models": [
            {"target": str(target), "model": f"{name} {version}"}
            for target, name, version in models
        ],
    }


@router.get("/board/accuracy")
async def accuracy(
    session: SessionDep,
    season: str | None = Query(default=None),
    include_absent: bool = Query(default=False),
) -> dict[str, Any]:
    """Realised accuracy and calibration, once fixtures have been played."""
    from euroleague_ai.ml.outcomes import accuracy_report, unscored_games

    report = await accuracy_report(session, season_code=season, include_absent=include_absent)
    report["unscored_games"] = await unscored_games(session, season_code=season)
    return report

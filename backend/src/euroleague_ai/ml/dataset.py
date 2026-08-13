"""Training dataset assembly.

Joins each persisted feature snapshot to the outcome that actually happened,
producing rows of the shape the roadmap describes::

    game_id · player_id · prediction_timestamp
    FEATURES: everything knowable before the cutoff
    TARGET:   what the player actually did

**The features are read, never recomputed.** A snapshot is the exact input a
model saw, frozen at its cutoff; recomputing here would risk producing something
subtly different from what was stored, and would defeat the point of persisting
them. The box score is touched only to supply the target.

Rows carry season code and round number so the backtest can split
chronologically. That metadata is not a feature and must never be handed to a
model -- round number correlates with everything and would let a model cheat by
memorising the calendar.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.competition import Season
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats

logger = get_logger(__name__)

#: Maps a prediction target to the box score column holding its realised value.
#: MINUTES is special-cased because storage keeps seconds.
TARGET_COLUMNS: dict[PredictionTarget, str] = {
    PredictionTarget.MINUTES: "seconds_played",
    PredictionTarget.POINTS: "points",
    PredictionTarget.REBOUNDS: "total_rebounds",
    PredictionTarget.ASSISTS: "assists",
    PredictionTarget.THREE_POINTERS_MADE: "fg3m",
    PredictionTarget.THREE_POINTERS_ATTEMPTED: "fg3a",
    PredictionTarget.STEALS: "steals",
    PredictionTarget.BLOCKS: "blocks_favour",
    PredictionTarget.TURNOVERS: "turnovers",
    PredictionTarget.FREE_THROWS_MADE: "ftm",
    PredictionTarget.FREE_THROWS_ATTEMPTED: "fta",
    PredictionTarget.VALUATION: "valuation",
}

#: Maps a prediction target to the feature-name stem used by the form family,
#: e.g. ``season_points_mean`` / ``last5_points_mean``. Baselines rely on this.
TARGET_FEATURE_STEMS: dict[PredictionTarget, str] = {
    PredictionTarget.MINUTES: "minutes",
    PredictionTarget.POINTS: "points",
    PredictionTarget.REBOUNDS: "rebounds",
    PredictionTarget.ASSISTS: "assists",
    PredictionTarget.THREE_POINTERS_MADE: "fg3m",
    PredictionTarget.THREE_POINTERS_ATTEMPTED: "fg3a",
    PredictionTarget.STEALS: "steals",
    PredictionTarget.BLOCKS: "blocks",
    PredictionTarget.TURNOVERS: "turnovers",
    PredictionTarget.FREE_THROWS_ATTEMPTED: "fta",
    PredictionTarget.VALUATION: "valuation",
}


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One player-game: pre-game features plus the realised outcome."""

    snapshot_id: uuid.UUID
    player_id: uuid.UUID
    game_id: uuid.UUID
    season_code: str
    round_number: int | None
    tip_off_utc: datetime
    prediction_timestamp: datetime
    features: dict[str, Any]
    target: float

    #: Realised minutes. Not a feature -- it is only known after the game -- but
    #: needed to stratify evaluation, because a model that looks accurate overall
    #: can be useless for the 10-20 minute rotation players where the
    #: interesting decisions actually live.
    actual_minutes: float


def target_value(target: PredictionTarget, stats_value: int) -> float:
    """Convert the stored column value into the target's natural unit."""
    if target is PredictionTarget.MINUTES:
        return stats_value / 60.0
    return float(stats_value)


async def load_training_rows(
    session: AsyncSession,
    *,
    target: PredictionTarget,
    feature_set_version: str,
    season_codes: list[str] | None = None,
    min_round: int | None = None,
    exclude_dnp: bool = True,
) -> list[TrainingRow]:
    """Load every snapshot that has a known outcome for ``target``.

    Args:
        target: Which quantity to extract as the label.
        feature_set_version: Only snapshots at this version. Mixing versions
            would train on inconsistent columns.
        season_codes: Restrict to these seasons.
        min_round: Drop early rounds where form windows are mostly empty.
        exclude_dnp: Drop games the player did not play. Defaults to ``True``
            because a DNP is an availability outcome, not a performance one --
            folding zeros into a points regression teaches the model that
            injured stars are low scorers. Availability is modelled separately.

    Returns:
        Rows sorted chronologically by tip-off, which is the order the backtest
        consumes them in.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"no box score column mapped for target {target}")

    column = getattr(PlayerGameStats, TARGET_COLUMNS[target])

    stmt = (
        select(
            PlayerFeatureSnapshot.id,
            PlayerFeatureSnapshot.player_id,
            PlayerFeatureSnapshot.game_id,
            Season.code,
            Game.round_number,
            Game.tip_off_utc,
            PlayerFeatureSnapshot.prediction_timestamp,
            PlayerFeatureSnapshot.features,
            column,
            PlayerGameStats.seconds_played,
        )
        .join(Game, Game.id == PlayerFeatureSnapshot.game_id)
        .join(Season, Season.id == Game.season_id)
        .join(GameTeam, (GameTeam.game_id == Game.id) & (GameTeam.team_id == PlayerFeatureSnapshot.team_id))
        .join(
            PlayerGameStats,
            (PlayerGameStats.game_team_id == GameTeam.id)
            & (PlayerGameStats.player_id == PlayerFeatureSnapshot.player_id),
        )
        .where(PlayerFeatureSnapshot.feature_set_version == feature_set_version)
        .order_by(Game.tip_off_utc)
    )

    if exclude_dnp:
        stmt = stmt.where(PlayerGameStats.did_not_play.is_(False))
    if season_codes:
        stmt = stmt.where(Season.code.in_(season_codes))
    if min_round is not None:
        stmt = stmt.where((Game.round_number.is_(None)) | (Game.round_number >= min_round))

    rows = (await session.execute(stmt)).all()

    result = [
        TrainingRow(
            snapshot_id=row[0],
            player_id=row[1],
            game_id=row[2],
            season_code=row[3],
            round_number=row[4],
            tip_off_utc=row[5],
            prediction_timestamp=row[6],
            features=row[7] or {},
            target=target_value(target, row[8]),
            actual_minutes=row[9] / 60.0,
        )
        for row in rows
    ]

    logger.info(
        "loaded training rows",
        extra={
            "target": str(target),
            "rows": len(result),
            "seasons": len({r.season_code for r in result}),
            "feature_set_version": feature_set_version,
        },
    )
    return result


def describe_rows(rows: list[TrainingRow]) -> dict[str, Any]:
    """Summary statistics, for sanity-checking a dataset before using it."""
    if not rows:
        return {"rows": 0}

    targets = [row.target for row in rows]
    by_season: dict[str, int] = {}
    for row in rows:
        by_season[row.season_code] = by_season.get(row.season_code, 0) + 1

    ordered = sorted(targets)
    midpoint = len(ordered) // 2

    return {
        "rows": len(rows),
        "seasons": dict(sorted(by_season.items())),
        "players": len({row.player_id for row in rows}),
        "games": len({row.game_id for row in rows}),
        "target_mean": round(sum(targets) / len(targets), 4),
        "target_median": round(ordered[midpoint], 4),
        "target_min": round(min(targets), 4),
        "target_max": round(max(targets), 4),
        "feature_count": len(rows[0].features),
    }

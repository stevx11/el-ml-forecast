"""Snapshot backfill.

Generating training data means building one feature vector per player-game:
roughly 25 players x 400 games x 10 seasons, so on the order of 100,000 rows,
each running several history queries. That is long enough that the job has to be
built like the ingestion backfill -- chunked, resumable, and safe to interrupt --
rather than as a single loop that must run to completion.

**Which rows exist.** A training row is only useful if the outcome is known, so
snapshots are built for players who actually appeared in a finished game. The
features are still computed strictly as of the pre-game cutoff; the box score is
used only to decide *that* a row should exist, never as an input to it.

That distinction matters and is easy to get wrong. Using the box score to select
rows introduces a mild survivorship effect -- players who were unavailable have
no row -- which is correct for training a "given they play, how much do they
produce" model, and is exactly why minutes and availability are modelled
separately rather than folded into the same regression.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.registry import FEATURE_SET_VERSION, build_player_features
from euroleague_ai.features.snapshot import default_prediction_timestamp
from euroleague_ai.models.competition import Season
from euroleague_ai.models.enums import GameStatus, TeamSide
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats

logger = get_logger(__name__)

#: Snapshots written before committing. Large enough to amortise the round trip,
#: small enough that an interruption costs little.
COMMIT_EVERY = 250

#: Games to report progress on.
LOG_EVERY_GAMES = 25

#: Rounds skipped at the start of each season. Form windows are empty or nearly
#: empty here, so these rows mostly teach the model how to handle missingness
#: rather than how basketball works. They are excluded from training data by
#: default; pass ``min_round=0`` to keep them.
DEFAULT_MIN_ROUND = 4


@dataclass(slots=True)
class BackfillResult:
    games_processed: int = 0
    snapshots_written: int = 0
    snapshots_skipped: int = 0
    seasons: dict[str, int] = field(default_factory=dict)

    def merge(self, other: BackfillResult) -> None:
        self.games_processed += other.games_processed
        self.snapshots_written += other.snapshots_written
        self.snapshots_skipped += other.snapshots_skipped
        for key, value in other.seasons.items():
            self.seasons[key] = self.seasons.get(key, 0) + value


@dataclass(frozen=True, slots=True)
class _PlayerGameTarget:
    """One player-game to build a snapshot for, plus the realised outcome."""

    player_id: uuid.UUID
    team_id: uuid.UUID
    opponent_team_id: uuid.UUID
    is_home: bool


async def _targets(session: AsyncSession, game: Game) -> list[_PlayerGameTarget]:
    """Players who appeared in ``game``, with their side and opponent."""
    opponent = GameTeam.__table__.alias("opponent")

    stmt = (
        select(
            PlayerGameStats.player_id,
            GameTeam.team_id,
            GameTeam.side,
            opponent.c.team_id.label("opponent_team_id"),
        )
        .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
        .join(
            opponent,
            (opponent.c.game_id == GameTeam.game_id) & (opponent.c.side != GameTeam.side),
        )
        .where(GameTeam.game_id == game.id)
    )

    return [
        _PlayerGameTarget(
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            is_home=side is TeamSide.HOME,
        )
        for player_id, team_id, side, opponent_team_id in (
            await session.execute(stmt)
        ).all()
    ]


async def _existing_snapshot_players(
    session: AsyncSession, game_id: uuid.UUID, cutoff: datetime
) -> set[uuid.UUID]:
    """Players who already have a snapshot for this game, version and cutoff.

    This is what makes the backfill resumable.
    """
    stmt = select(PlayerFeatureSnapshot.player_id).where(
        PlayerFeatureSnapshot.game_id == game_id,
        PlayerFeatureSnapshot.feature_set_version == FEATURE_SET_VERSION,
        PlayerFeatureSnapshot.prediction_timestamp == cutoff,
    )
    return set((await session.execute(stmt)).scalars().all())


async def backfill_game(
    session: AsyncSession,
    game: Game,
    *,
    clock: Clock,
    settings: Settings,
) -> BackfillResult:
    """Build and store snapshots for every player who appeared in ``game``."""
    result = BackfillResult(games_processed=1)

    cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
    already = await _existing_snapshot_players(session, game.id, cutoff)
    targets = await _targets(session, game)

    now = clock.now()
    for target in targets:
        if target.player_id in already:
            result.snapshots_skipped += 1
            continue

        features = await build_player_features(
            session,
            player_id=target.player_id,
            team_id=target.team_id,
            opponent_team_id=target.opponent_team_id,
            as_of=cutoff,
            season_id=game.season_id,
            is_home=target.is_home,
        )

        # ON CONFLICT rather than a pre-check: two concurrent backfills, or a
        # retry after a partial commit, must not collide.
        await session.execute(
            pg_insert(PlayerFeatureSnapshot)
            .values(
                id=uuid.uuid4(),
                player_id=target.player_id,
                game_id=game.id,
                team_id=target.team_id,
                opponent_team_id=target.opponent_team_id,
                prediction_timestamp=cutoff,
                feature_set_version=FEATURE_SET_VERSION,
                features=features,
                computed_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "player_id",
                    "game_id",
                    "feature_set_version",
                    "prediction_timestamp",
                ]
            )
        )
        result.snapshots_written += 1

    return result


async def _games_to_process(
    session: AsyncSession,
    *,
    season_codes: list[str] | None,
    min_round: int,
) -> list[Game]:
    """Finished games eligible for snapshot generation, oldest first."""
    stmt = (
        select(Game)
        .where(Game.status == GameStatus.FINISHED)
        .order_by(Game.tip_off_utc)
    )
    if min_round > 0:
        # NULL rounds are kept: some phases do not carry a round number, and
        # dropping them would silently remove playoff games.
        stmt = stmt.where((Game.round_number.is_(None)) | (Game.round_number >= min_round))
    if season_codes:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(
            Season.code.in_(season_codes)
        )
    return list((await session.execute(stmt)).scalars().all())


async def backfill_snapshots(
    session: AsyncSession,
    *,
    season_codes: list[str] | None = None,
    min_round: int = DEFAULT_MIN_ROUND,
    limit: int | None = None,
    clock: Clock | None = None,
    settings: Settings | None = None,
) -> BackfillResult:
    """Build feature snapshots across finished games.

    Resumable: games whose snapshots already exist at the current
    ``FEATURE_SET_VERSION`` and cutoff are skipped, so interrupting and
    re-running costs at most one commit chunk.

    Args:
        season_codes: Restrict to these seasons. ``None`` means all.
        min_round: Skip early rounds where form windows are empty.
        limit: Stop after this many games. Useful for a smoke run.
    """
    clock = clock or SystemClock()
    settings = settings or get_settings()

    games = await _games_to_process(
        session, season_codes=season_codes, min_round=min_round
    )
    if limit is not None:
        games = games[:limit]

    total = len(games)
    logger.info(
        "starting snapshot backfill",
        extra={
            "games": total,
            "feature_set_version": FEATURE_SET_VERSION,
            "seasons": season_codes or "all",
            "min_round": min_round,
        },
    )

    aggregate = BackfillResult()
    pending = 0

    for index, game in enumerate(games, start=1):
        game_result = await backfill_game(session, game, clock=clock, settings=settings)
        aggregate.merge(game_result)
        pending += game_result.snapshots_written

        if pending >= COMMIT_EVERY:
            await session.commit()
            pending = 0

        if index % LOG_EVERY_GAMES == 0 or index == total:
            logger.info(
                "snapshot progress",
                extra={
                    "games": f"{index}/{total}",
                    "written": aggregate.snapshots_written,
                    "skipped": aggregate.snapshots_skipped,
                },
            )

    await session.commit()
    logger.info(
        "snapshot backfill complete",
        extra={
            "games": aggregate.games_processed,
            "written": aggregate.snapshots_written,
            "skipped": aggregate.snapshots_skipped,
        },
    )
    return aggregate

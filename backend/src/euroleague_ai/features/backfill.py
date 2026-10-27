"""Snapshot backfill.

Generating training data means building one feature vector per player-game:
roughly 25 players x 400 games x 10 seasons, so on the order of 100,000 rows,
each running several history queries. That is long enough that the job has to be
built like the ingestion backfill -- chunked, resumable, and safe to interrupt --
rather than as a single loop that must run to completion.

**Which rows exist.** Two populations, unioned per game:

1. **Everyone in the box score**, including players who dressed and were never
   used. These are the rows the stat models train on (DNPs are filtered out at
   load time, not here).
2. **Every inference candidate** -- the players
   :func:`~euroleague_ai.repositories.rosters.candidate_players` would have
   projected at the cutoff, whether or not they then took the floor.

The second set exists for the play-probability model. Before it, a player who
was injured had no row at all, so nothing could learn *who does not play*. And
inference does not predict "the box score". It predicts the candidate list,
which always includes a few players who will not appear. Training availability
on anything other than that list would teach it a different question.

The stat datasets are unaffected. They inner-join the box score, so a candidate
who never dressed has no outcome row and never reaches a regression.

The features are still computed strictly as of the pre-game cutoff. The box
score only decides *that* a row should exist, never what goes into it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select, text
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
from euroleague_ai.repositories.rosters import candidate_players, current_club

logger = get_logger(__name__)

#: Snapshots written before committing. Large enough to amortise the round trip,
#: small enough that an interruption costs little.
COMMIT_EVERY = 250

#: Games to report progress on.
LOG_EVERY_GAMES = 25

#: Fixed key for the Postgres advisory lock that serialises backfill runs.
#: Arbitrary value; only its stability matters, so it is written as a plain
#: integer rather than derived from a hash that could change between releases.
ADVISORY_LOCK_KEY = 815_243_001

#: Rounds skipped at the start of each season.
#:
#: Under feature set v2 this was 4: season-scoped form windows were empty in
#: rounds 1-3, so those rows taught the model about missingness rather than about
#: basketball. Under v3 the windows cross season boundaries, so a returning
#: player carries real form into round one and the rows are worth training on.
#:
#: Keeping the exclusion would now be actively harmful. If the training set never
#: contains a round-one row, the model never learns how to weigh career form
#: against an empty current season -- which is exactly the situation it faces
#: every October in production. Train on the distribution you predict on.
#:
#: Rows that are still genuinely blank (a rookie's debut, or any game in the
#: earliest ingested season) stay in: LightGBM learns a split for missing and
#: falls back toward the global mean, which is the right answer for a player we
#: know nothing about.
DEFAULT_MIN_ROUND = 1


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


async def _candidate_targets(
    session: AsyncSession, game: Game, cutoff: datetime
) -> list[_PlayerGameTarget]:
    """Players inference would have projected for ``game`` at ``cutoff``.

    Same function, same cutoff, same arguments as
    :func:`~euroleague_ai.ml.inference.predict_game`, so the historical
    candidate list is the one a running system would have produced.
    """
    club_by_player = await current_club(session, as_of=cutoff)
    targets: list[_PlayerGameTarget] = []
    for team_id, opponent_id, is_home in (
        (game.home_team_id, game.away_team_id, True),
        (game.away_team_id, game.home_team_id, False),
    ):
        for player_id in await candidate_players(
            session,
            team_id=team_id,
            season_id=game.season_id,
            as_of=cutoff,
            club_by_player=club_by_player,
        ):
            targets.append(
                _PlayerGameTarget(
                    player_id=player_id,
                    team_id=team_id,
                    opponent_team_id=opponent_id,
                    is_home=is_home,
                )
            )
    return targets


def merge_targets(
    box_score: list[_PlayerGameTarget], candidates: list[_PlayerGameTarget]
) -> list[_PlayerGameTarget]:
    """Box-score players first, then candidates not already covered.

    The box score wins any disagreement about which side a player was on. It is
    what actually happened, whereas the candidate side is a roster inference
    that can be wrong around a transfer. A snapshot is keyed by player and game,
    so a player can only have one.
    """
    seen = {target.player_id for target in box_score}
    merged = list(box_score)
    for target in candidates:
        if target.player_id not in seen:
            seen.add(target.player_id)
            merged.append(target)
    return merged


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
    include_candidates: bool = True,
) -> BackfillResult:
    """Build and store snapshots for one game's box score and candidate list.

    Args:
        include_candidates: Also snapshot inference candidates who did not
            appear. Needed by the play-probability model; turn it off only
            for a quick stat-model backfill.
    """
    result = BackfillResult(games_processed=1)

    cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
    already = await _existing_snapshot_players(session, game.id, cutoff)
    targets = await _targets(session, game)
    if include_candidates:
        targets = merge_targets(targets, await _candidate_targets(session, game, cutoff))

    now = clock.now()
    # One cache per game. Team ratings and squad data are identical for every
    # player on a side, so they are fetched once here instead of once per
    # player -- roughly three times fewer queries for the same output.
    cache: dict[Any, Any] = {}
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
            cache=cache,
            game=game,
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
    """Games eligible for snapshot generation, oldest first.

    Restricted to games that actually have box scores. A finished game with no
    box score yields no training rows -- there is no outcome to learn from --
    and walking it costs a query to discover nothing. Excluding them up front
    makes the progress counter mean what it appears to mean, which matters when
    the job runs for an hour.
    """
    has_box_score = (
        select(GameTeam.game_id)
        .join(PlayerGameStats, PlayerGameStats.game_team_id == GameTeam.id)
        .distinct()
    )
    stmt = (
        select(Game)
        .where(Game.status == GameStatus.FINISHED, Game.id.in_(has_box_score))
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
    include_candidates: bool = True,
) -> BackfillResult:
    """Build feature snapshots across finished games.

    Resumable: games whose snapshots already exist at the current
    ``FEATURE_SET_VERSION`` and cutoff are skipped, so interrupting and
    re-running costs at most one commit chunk.

    Args:
        season_codes: Restrict to these seasons. ``None`` means all.
        min_round: Skip rounds below this number. See ``DEFAULT_MIN_ROUND``.
        limit: Stop after this many games. Useful for a smoke run.
        include_candidates: Also snapshot inference candidates who did not
            play. See the module docstring.
    """
    clock = clock or SystemClock()
    settings = settings or get_settings()

    # Session-scoped advisory lock. Released automatically when the connection
    # closes, so a crashed run cannot leave the job permanently blocked.
    acquired = (
        await session.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": ADVISORY_LOCK_KEY}
        )
    ).scalar_one()
    if not acquired:
        raise RuntimeError(
            "another snapshot backfill is already running against this database. "
            "Concurrent runs duplicate work and contend on the same rows -- wait "
            "for it to finish, or stop it first."
        )

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
        game_result = await backfill_game(
            session,
            game,
            clock=clock,
            settings=settings,
            include_candidates=include_candidates,
        )
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

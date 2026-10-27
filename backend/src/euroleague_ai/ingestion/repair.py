"""In-place repair of ``observed_at`` on already-ingested rows.

Backfilled history was originally stamped with the ingest wall-clock, which
placed every historical fact in the future relative to its own game and made it
invisible to every feature query. See :mod:`euroleague_ai.ingestion.knowability`
for the full explanation.

This rewrites the affected timestamps from data already in the database. No API
calls, so it costs seconds rather than another fight with the rate limiter.

**Only moves timestamps backwards.** A row whose ``observed_at`` is already
earlier than the derived knowable time is left alone -- that row was ingested
live and its timestamp is the truth. Moving it forward would be inventing
knowledge we did not have.

Intervals are passed as a number of seconds multiplied by ``INTERVAL '1 second'``
rather than as a bound ``timedelta``. asyncpg sends a bare ``timedelta`` as an
untyped parameter, and Postgres cannot infer that ``tip_off_utc - $1`` is
timestamp arithmetic -- it fails with "operator does not exist: timestamp with
time zone > interval". Multiplying an explicit interval literal removes the
ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ingestion.knowability import (
    BOX_SCORE_AVAILABLE_AFTER,
    FIXTURE_KNOWN_BEFORE,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class RepairResult:
    games: int = 0
    player_game_stats: int = 0
    team_game_stats: int = 0
    stints: int = 0

    @property
    def total(self) -> int:
        return self.games + self.player_game_stats + self.team_game_stats + self.stints

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "player_game_stats": self.player_game_stats,
            "team_game_stats": self.team_game_stats,
            "player_team_stints": self.stints,
            "total": self.total,
        }


# Each repair is a matched (count, update) pair sharing one predicate, written
# out rather than derived from each other. Deriving the SELECT from the UPDATE by
# string surgery is how the type bug above stayed invisible until runtime.

_LEAD = ":lead_seconds * INTERVAL '1 second'"
_DELAY = ":delay_seconds * INTERVAL '1 second'"

_FIXTURE_PREDICATE = f"observed_at > tip_off_utc - ({_LEAD})"
_FIXTURE_COUNT = f"SELECT count(*) FROM games WHERE {_FIXTURE_PREDICATE}"
_FIXTURE_UPDATE = f"""
UPDATE games
SET observed_at = tip_off_utc - ({_LEAD})
WHERE {_FIXTURE_PREDICATE}
"""

_PLAYER_STATS_COUNT = f"""
SELECT count(*)
FROM player_game_stats s
JOIN game_teams gt ON gt.id = s.game_team_id
JOIN games g ON g.id = gt.game_id
WHERE s.observed_at > g.tip_off_utc + ({_DELAY})
"""
_PLAYER_STATS_UPDATE = f"""
UPDATE player_game_stats s
SET observed_at = g.tip_off_utc + ({_DELAY})
FROM game_teams gt, games g
WHERE gt.id = s.game_team_id
  AND g.id = gt.game_id
  AND s.observed_at > g.tip_off_utc + ({_DELAY})
"""

_TEAM_STATS_COUNT = f"""
SELECT count(*)
FROM team_game_stats s
JOIN game_teams gt ON gt.id = s.game_team_id
JOIN games g ON g.id = gt.game_id
WHERE s.observed_at > g.tip_off_utc + ({_DELAY})
"""
_TEAM_STATS_UPDATE = f"""
UPDATE team_game_stats s
SET observed_at = g.tip_off_utc + ({_DELAY})
FROM game_teams gt, games g
WHERE gt.id = s.game_team_id
  AND g.id = gt.game_id
  AND s.observed_at > g.tip_off_utc + ({_DELAY})
"""

_STINT_COUNT = "SELECT count(*) FROM player_team_stints WHERE observed_at > effective_from"
_STINT_UPDATE = """
UPDATE player_team_stints
SET observed_at = effective_from
WHERE observed_at > effective_from
"""


async def _rows_changed(session: AsyncSession, sql: str, params: dict[str, Any]) -> int:
    """Run an UPDATE or DELETE and return how many rows it touched."""
    result = await session.execute(text(sql), params)
    assert isinstance(result, CursorResult)
    return result.rowcount or 0


def _params() -> dict[str, float]:
    return {
        "lead_seconds": FIXTURE_KNOWN_BEFORE.total_seconds(),
        "delay_seconds": BOX_SCORE_AVAILABLE_AFTER.total_seconds(),
    }


async def repair_observed_at(session: AsyncSession, *, dry_run: bool = False) -> RepairResult:
    """Rewrite ``observed_at`` to the derived knowable time where it is too late.

    Args:
        dry_run: Count affected rows without modifying them.

    Returns:
        Rows changed (or that would change) per table.
    """
    params = _params()

    if dry_run:
        counts = {}
        for label, sql in (
            ("games", _FIXTURE_COUNT),
            ("player_game_stats", _PLAYER_STATS_COUNT),
            ("team_game_stats", _TEAM_STATS_COUNT),
            ("stints", _STINT_COUNT),
        ):
            counts[label] = int((await session.execute(text(sql), params)).scalar() or 0)

        result = RepairResult(
            games=counts["games"],
            player_game_stats=counts["player_game_stats"],
            team_game_stats=counts["team_game_stats"],
            stints=counts["stints"],
        )
        logger.info("observed_at repair (dry run)", extra=result.as_dict())
        return result

    # Games first: the stats repairs read tip_off_utc, which this does not touch.
    games = await _rows_changed(session, _FIXTURE_UPDATE, params)
    player_stats = await _rows_changed(session, _PLAYER_STATS_UPDATE, params)
    team_stats = await _rows_changed(session, _TEAM_STATS_UPDATE, params)
    stints = await _rows_changed(session, _STINT_UPDATE, params)

    result = RepairResult(
        games=games,
        player_game_stats=player_stats,
        team_game_stats=team_stats,
        stints=stints,
    )
    logger.info("observed_at repaired", extra=result.as_dict())
    return result


async def purge_feature_snapshots(
    session: AsyncSession, *, feature_set_version: str | None = None
) -> int:
    """Delete feature snapshots so they can be rebuilt.

    Needed after a repair: snapshots are immutable by design, so correcting the
    data underneath them does not update them. They have to be discarded and
    recomputed.

    Predictions reference snapshots, so those are removed first. There are none
    yet -- no model exists -- but the ordering is written now so this does not
    become a foreign key failure later.
    """
    params: dict[str, Any] = {}
    snapshot_filter = ""
    if feature_set_version is not None:
        snapshot_filter = " WHERE feature_set_version = :version"
        params["version"] = feature_set_version

    await session.execute(
        text(
            "DELETE FROM prediction_outcomes WHERE prediction_id IN ("
            " SELECT p.id FROM predictions p"
            " WHERE p.player_feature_snapshot_id IN ("
            "  SELECT id FROM player_feature_snapshots" + snapshot_filter + "))"
        ),
        params,
    )
    await session.execute(
        text(
            "DELETE FROM predictions WHERE player_feature_snapshot_id IN ("
            " SELECT id FROM player_feature_snapshots" + snapshot_filter + ")"
        ),
        params,
    )
    deleted = await _rows_changed(
        session, "DELETE FROM player_feature_snapshots" + snapshot_filter, params
    )

    logger.info(
        "purged feature snapshots",
        extra={"deleted": deleted, "feature_set_version": feature_set_version or "all"},
    )
    return deleted

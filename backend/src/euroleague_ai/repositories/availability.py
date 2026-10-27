"""As-of reads for injury and availability reports.

Same rule as :mod:`euroleague_ai.repositories.history`: every read takes
``as_of`` and filters ``observed_at <= as_of``. A report entered at 18:30 does
not exist for a prediction cut off at 17:00, even though it describes the
afternoon.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, IngestionStatus, StatusSourceType
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.ingestion import IngestionRun

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class StatusReport:
    """One stored availability observation, flattened for resolution code."""

    id: uuid.UUID
    player_id: uuid.UUID
    game_id: uuid.UUID | None
    status: AvailabilityStatus
    reason: str | None
    expected_return: datetime | None
    source: StatusSourceType
    event_timestamp: datetime
    observed_at: datetime
    #: ``None`` for authoritative reports (manual, provider); 0-1 for news.
    confidence: float | None = None
    #: For a player's newest injury-table row: the latest table fetch, known at
    #: ``as_of``, that listed his club. The importer writes only rows that
    #: change, so a row can be days old while every fetch since has repeated
    #: it. Ordering by this time, not the row's own, is what stops an older
    #: news reading from outranking the table's current word.
    confirmed_at: datetime | None = None

    @property
    def effective_at(self) -> datetime:
        """When this report was last known to hold, for newest-first ordering."""
        if self.confirmed_at is not None and self.confirmed_at > self.event_timestamp:
            return self.confirmed_at
        return self.event_timestamp


async def status_reports(
    session: AsyncSession, player_ids: Iterable[uuid.UUID], *, as_of: datetime
) -> dict[uuid.UUID, list[StatusReport]]:
    """Every report for these players that was known at ``as_of``, newest first."""
    as_of = ensure_utc(as_of, field="as_of")
    ids = list(player_ids)
    if not ids:
        return {}
    rows = (
        await session.execute(
            select(PlayerStatusEvent)
            .where(
                PlayerStatusEvent.player_id.in_(ids),
                PlayerStatusEvent.observed_at <= as_of,
                PlayerStatusEvent.event_timestamp <= as_of,
            )
            .order_by(
                PlayerStatusEvent.event_timestamp.desc(),
                PlayerStatusEvent.observed_at.desc(),
            )
        )
    ).scalars()
    events = list(rows)
    fetches = await table_fetches(session, as_of=as_of)
    # The newest table row per player, by when it was observed.
    newest_table: dict[uuid.UUID, PlayerStatusEvent] = {}
    for event in events:
        if event.source_type is not StatusSourceType.REPORT:
            continue
        seen = newest_table.get(event.player_id)
        if seen is None or event.observed_at > seen.observed_at:
            newest_table[event.player_id] = event

    out: dict[uuid.UUID, list[StatusReport]] = {}
    for event in events:
        confirmed = None
        if newest_table.get(event.player_id) is event:
            confirmed = _confirmation(fetches, event)
        out.setdefault(event.player_id, []).append(
            StatusReport(
                id=event.id,
                player_id=event.player_id,
                game_id=event.game_id,
                status=event.status,
                reason=event.reason,
                expected_return=event.expected_return_date,
                source=event.source_type,
                event_timestamp=event.event_timestamp,
                observed_at=event.observed_at,
                confidence=float(event.confidence) if event.confidence is not None else None,
                confirmed_at=confirmed,
            )
        )
    return out


#: Recorded by every successful injury-table import, as an ``ingestion_runs``
#: row: when the table was fetched, and which clubs it listed.
TABLE_FETCH_JOB = "injury_report.fetch"


@dataclass(frozen=True, slots=True)
class TableFetch:
    fetched_at: datetime
    team_ids: frozenset[str]


async def table_fetches(
    session: AsyncSession, *, as_of: datetime, limit: int = 40
) -> list[TableFetch]:
    """Successful table fetches known at ``as_of``, newest first."""
    rows = (
        await session.execute(
            select(IngestionRun.started_at, IngestionRun.details)
            .where(
                IngestionRun.job_name == TABLE_FETCH_JOB,
                IngestionRun.status == IngestionStatus.SUCCEEDED,
                IngestionRun.started_at <= as_of,
            )
            .order_by(IngestionRun.started_at.desc())
            .limit(limit)
        )
    ).all()
    return [
        TableFetch(started_at, frozenset(str(t) for t in (details or {}).get("team_ids", [])))
        for started_at, details in rows
    ]


def _confirmation(fetches: list[TableFetch], event: PlayerStatusEvent) -> datetime | None:
    """The latest fetch after this row was written that listed the player's club."""
    team = str(event.team_id) if event.team_id is not None else None
    for fetch in fetches:  # newest first
        if fetch.fetched_at < event.observed_at:
            return None
        if team is not None and team in fetch.team_ids:
            return fetch.fetched_at
    return None


async def last_appearances(
    session: AsyncSession, player_ids: Iterable[uuid.UUID], *, as_of: datetime
) -> dict[uuid.UUID, datetime]:
    """Tip-off of each player's most recent game with playing time, as known at ``as_of``.

    Used to retire an open-ended "out" report: a player who has played since
    the report was made is evidently back.
    """
    as_of = ensure_utc(as_of, field="as_of")
    ids = list(player_ids)
    if not ids:
        return {}
    rows = (
        await session.execute(
            select(PlayerGameStats.player_id, func.max(Game.tip_off_utc))
            .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            .join(Game, Game.id == GameTeam.game_id)
            .where(
                PlayerGameStats.player_id.in_(ids),
                PlayerGameStats.did_not_play.is_(False),
                PlayerGameStats.observed_at <= as_of,
                Game.tip_off_utc < as_of,
            )
            .group_by(PlayerGameStats.player_id)
        )
    ).all()
    return dict(rows)

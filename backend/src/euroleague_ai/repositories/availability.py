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
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats

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
    out: dict[uuid.UUID, list[StatusReport]] = {}
    for event in rows:
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
            )
        )
    return out


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
    return {player_id: tip_off for player_id, tip_off in rows}

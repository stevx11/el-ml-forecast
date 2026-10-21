"""After a round: were the injury statuses the predictions used right?

For every finished game, take the P(plays) predictions as they were last
written and the injury report each one used (stored in its explanation), and
set it beside what the box score says he did.

Two kinds of status are not in a prediction's explanation, so they are
re-resolved as of the cutoff instead:

* **ready**: an "available" report clears a player rather than flagging him, so
  predictions do not record it;
* **late**: a report that reached us after the last prediction run but before
  the cutoff. The prediction could not use it; seeing how often this happens
  says whether the scheduler runs late enough.

The question it answers is narrow and the one that matters for trusting the
pipeline: a player we set to P(plays) = 0 should not have played, and a player
the report called ready should have. Anything else (questionable, doubtful,
probable) is reported as a play rate, because those labels promise nothing;
after a season they become calibrated features.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from euroleague_ai.features.injury_status import resolve_status
from euroleague_ai.models.competition import Season, Team
from euroleague_ai.models.enums import AvailabilityStatus, GameStatus, PredictionTarget
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.people import Player
from euroleague_ai.models.prediction import Prediction
from euroleague_ai.repositories.availability import last_appearances, status_reports

if TYPE_CHECKING:  # pragma: no cover - typing only
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

#: Display order, most to least serious.
GROUPS: tuple[str, ...] = (
    "ruled out",
    "out (news, not applied)",
    "doubtful",
    "questionable",
    "probable",
    "ready",
    "late report",
)


@dataclass(frozen=True, slots=True)
class CheckedPlayer:
    game_code: int
    team: str
    player: str
    group: str
    status: str
    source: str
    plays: float
    #: Minutes played; ``None`` if he did not take the floor.
    minutes: float | None
    #: In the box score at all (dressed), whether or not he played.
    dressed: bool
    reason: str | None

    @property
    def played(self) -> bool:
        return self.minutes is not None

    @property
    def mismatch(self) -> bool:
        """What the status promised and the box score contradicts."""
        if self.group == "ruled out":
            return self.played
        if self.group == "ready":
            return not self.played
        return False


@dataclass(slots=True)
class RoundCheck:
    games: int = 0
    players: list[CheckedPlayer] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        by_group: dict[str, list[CheckedPlayer]] = defaultdict(list)
        for player in self.players:
            by_group[player.group].append(player)
        for group in GROUPS:
            members = by_group.get(group, [])
            if not members:
                continue
            played = sum(p.played for p in members)
            out[group] = {
                "players": len(members),
                "played": played,
                "play_rate": round(played / len(members), 3),
                "mean_p_plays": round(sum(p.plays for p in members) / len(members), 3),
                "mismatches": sum(p.mismatch for p in members),
            }
        return out

    @property
    def mismatches(self) -> list[CheckedPlayer]:
        return [p for p in self.players if p.mismatch]


def _group(status: AvailabilityStatus, *, applied: bool) -> str:
    if status in (AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED):
        return "ruled out" if applied else "out (news, not applied)"
    return {
        AvailabilityStatus.DOUBTFUL: "doubtful",
        AvailabilityStatus.QUESTIONABLE: "questionable",
        AvailabilityStatus.PROBABLE: "probable",
        AvailabilityStatus.AVAILABLE: "ready",
    }.get(status, "questionable")


async def finished_games(
    session: AsyncSession,
    *,
    now: datetime,
    days: int | None = None,
    round_number: int | None = None,
    season_code: str | None = None,
) -> list[Game]:
    """Finished games in the last ``days``, or in one round of a season."""
    stmt = select(Game).where(Game.status == GameStatus.FINISHED).order_by(Game.tip_off_utc)
    if round_number is not None:
        stmt = stmt.where(Game.round_number == round_number)
        if season_code:
            stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    else:
        stmt = stmt.where(
            Game.tip_off_utc >= now - timedelta(days=days or 7), Game.tip_off_utc <= now
        )
    return list((await session.execute(stmt)).scalars())


async def check_games(session: AsyncSession, games: list[Game]) -> RoundCheck:
    """Status at the cutoff against the box score, for every predicted player."""
    result = RoundCheck(games=len(games))
    for game in games:
        rows = (
            await session.execute(
                select(
                    Prediction.player_id,
                    Prediction.predicted_value,
                    Prediction.prediction_timestamp,
                    Prediction.created_at,
                    Prediction.explanation,
                    PlayerFeatureSnapshot.team_id,
                )
                .join(
                    PlayerFeatureSnapshot,
                    PlayerFeatureSnapshot.id == Prediction.player_feature_snapshot_id,
                )
                .where(Prediction.game_id == game.id, Prediction.target == PredictionTarget.PLAYS)
            )
        ).all()
        if not rows:
            continue
        # One prediction per player: the latest cutoff, then the latest write.
        latest: dict[uuid.UUID, Any] = {}
        for row in rows:
            seen = latest.get(row.player_id)
            if seen is None or (row.prediction_timestamp, row.created_at) > (
                seen.prediction_timestamp,
                seen.created_at,
            ):
                latest[row.player_id] = row
        cutoff = max(r.prediction_timestamp for r in latest.values())
        player_ids = list(latest)

        reports = await status_reports(session, player_ids, as_of=cutoff)
        played_before = await last_appearances(session, player_ids, as_of=cutoff)
        box = {
            player_id: (did_not_play, seconds)
            for player_id, did_not_play, seconds in (
                await session.execute(
                    select(
                        PlayerGameStats.player_id,
                        PlayerGameStats.did_not_play,
                        PlayerGameStats.seconds_played,
                    )
                    .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                    .where(GameTeam.game_id == game.id)
                )
            ).all()
        }
        names = dict(
            (
                await session.execute(
                    select(Player.id, Player.full_name).where(Player.id.in_(player_ids))
                )
            ).all()
        )
        teams = dict(
            (
                await session.execute(
                    select(Team.id, Team.code).where(
                        Team.id.in_({r.team_id for r in latest.values()})
                    )
                )
            ).all()
        )

        for player_id, row in latest.items():
            used = (row.explanation or {}).get("availability")
            if used:
                status = AvailabilityStatus(used["status"])
                group = _group(status, applied=bool(used.get("applied")))
                source, reason = str(used.get("source")), used.get("reason")
            else:
                resolved = resolve_status(
                    reports.get(player_id, []),
                    game_id=game.id,
                    tip_off=game.tip_off_utc,
                    last_played=played_before.get(player_id),
                    keep_available=True,
                )
                if resolved is None:
                    continue
                status = resolved.report.status
                group = "ready" if status is AvailabilityStatus.AVAILABLE else "late report"
                source, reason = str(resolved.report.source), resolved.report.reason
            entry = box.get(player_id)
            minutes = None
            if entry is not None and not entry[0]:
                minutes = round((entry[1] or 0) / 60.0, 1)
            result.players.append(
                CheckedPlayer(
                    game_code=game.game_code,
                    team=teams.get(row.team_id, "?"),
                    player=names.get(player_id, "?"),
                    group=group,
                    status=str(status),
                    source=source,
                    plays=float(row.predicted_value),
                    minutes=minutes,
                    dressed=entry is not None,
                    reason=reason,
                )
            )
    result.players.sort(key=lambda p: (GROUPS.index(p.group), p.game_code, p.team, p.player))
    return result

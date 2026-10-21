"""Recording injury and availability reports.

**Why start now, and by hand.** EuroLeague publishes an official injury report
every round, built from the clubs' own announcements, but as an article: no
feed, no structure, no timestamps. Nothing in this system has ever recorded one,
and bitemporal history cannot be backfilled. A report nobody stored before tip-off
is gone for good. So the first step is the simplest dependable one: enter each
round's report with the CLI. Every entry is stamped with when it was recorded,
which is exactly what a future model needs to learn from it honestly. Automated
extraction from news feeds comes next and writes to the same table.

Players are matched by name against current club rosters, so "Petrovic" or
"Marko Petrovic" works. Accents and case are ignored, and EuroLeague's
"SURNAME, FORENAME" order doesn't matter. An ambiguous name is refused with the
candidates listed rather than guessed.
"""

from __future__ import annotations

import csv
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import TYPE_CHECKING

from sqlalchemy import or_, select

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.competition import Team
from euroleague_ai.models.enums import AvailabilityStatus, EntityType, GameStatus, StatusSourceType
from euroleague_ai.models.game import Game
from euroleague_ai.models.identity import ExternalRef
from euroleague_ai.models.people import Player
from euroleague_ai.repositories.rosters import current_club

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

#: Words people actually use, mapped to the stored status.
STATUS_ALIASES: dict[str, AvailabilityStatus] = {
    "out": AvailabilityStatus.OUT,
    "ruled-out": AvailabilityStatus.OUT,
    "injured": AvailabilityStatus.OUT,
    "suspended": AvailabilityStatus.SUSPENDED,
    "doubtful": AvailabilityStatus.DOUBTFUL,
    "questionable": AvailabilityStatus.QUESTIONABLE,
    "gtd": AvailabilityStatus.QUESTIONABLE,
    "game-time": AvailabilityStatus.QUESTIONABLE,
    "day-to-day": AvailabilityStatus.QUESTIONABLE,
    "probable": AvailabilityStatus.PROBABLE,
    "available": AvailabilityStatus.AVAILABLE,
    "fit": AvailabilityStatus.AVAILABLE,
    "back": AvailabilityStatus.AVAILABLE,
}


class InjuryInputError(ValueError):
    """A report that cannot be recorded as given. The message says how to fix it."""


def parse_status(raw: str) -> AvailabilityStatus:
    key = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if key in STATUS_ALIASES:
        return STATUS_ALIASES[key]
    valid = ", ".join(sorted(STATUS_ALIASES))
    raise InjuryInputError(f"unknown status {raw!r}. Use one of: {valid}")


def normalise(text: str) -> str:
    """Lowercase, accents stripped, punctuation to spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in plain.lower()).split())


@dataclass(frozen=True, slots=True)
class RosterPlayer:
    player_id: uuid.UUID
    full_name: str
    team_id: uuid.UUID
    team_code: str
    code: str | None


async def current_rosters(session: AsyncSession, *, as_of: datetime) -> list[RosterPlayer]:
    """Every player registered to a club at ``as_of``, with names and codes."""
    clubs = await current_club(session, as_of=as_of)
    if not clubs:
        return []
    players = {
        p.id: p
        for p in (
            await session.execute(select(Player).where(Player.id.in_(list(clubs))))
        ).scalars()
    }
    teams = {
        t.id: t.code
        for t in (
            await session.execute(select(Team).where(Team.id.in_(set(clubs.values()))))
        ).scalars()
    }
    codes = {
        internal: external.strip()
        for internal, external in (
            await session.execute(
                select(ExternalRef.internal_id, ExternalRef.external_id).where(
                    ExternalRef.entity_type == EntityType.PLAYER,
                    ExternalRef.internal_id.in_(list(clubs)),
                )
            )
        ).all()
    }
    return [
        RosterPlayer(
            player_id=pid,
            full_name=players[pid].full_name,
            team_id=team_id,
            team_code=teams.get(team_id, "?"),
            code=codes.get(pid),
        )
        for pid, team_id in clubs.items()
        if pid in players
    ]


def match_player(
    roster: list[RosterPlayer], query: str, *, team_code: str | None = None
) -> RosterPlayer:
    """The one rostered player ``query`` names, or an error listing the options.

    A player code matches exactly. Otherwise every word of the query must appear
    among the words of the name.
    """
    pool = [
        p for p in roster if team_code is None or p.team_code.upper() == team_code.upper()
    ]
    stripped = query.strip()
    by_code = [p for p in pool if p.code and p.code.lstrip("P0") == stripped.lstrip("P0")]
    if stripped and by_code:
        return by_code[0]

    words = normalise(stripped).split()
    if not words:
        raise InjuryInputError("empty player name")
    matches = [p for p in pool if set(words) <= set(normalise(p.full_name).split())]
    if not matches:
        # Prefix match, for "Petro" or a shortened first name.
        matches = [
            p
            for p in pool
            if all(any(part.startswith(w) for part in normalise(p.full_name).split()) for w in words)
        ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        scope = f" at {team_code}" if team_code else ""
        raise InjuryInputError(f"no rostered player{scope} matches {query!r}")
    options = "; ".join(f"{p.full_name} ({p.team_code})" for p in matches[:8])
    raise InjuryInputError(
        f"{query!r} matches {len(matches)} players: {options}. Add --team or more of the name."
    )


async def next_game(
    session: AsyncSession, team_id: uuid.UUID, *, after: datetime
) -> Game | None:
    """The team's next scheduled fixture after ``after``."""
    return (
        await session.execute(
            select(Game)
            .where(
                or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                Game.status == GameStatus.SCHEDULED,
                Game.tip_off_utc > after,
            )
            .order_by(Game.tip_off_utc)
            .limit(1)
        )
    ).scalar_one_or_none()


@dataclass(frozen=True, slots=True)
class RecordedReport:
    player: RosterPlayer
    status: AvailabilityStatus
    scope: str
    game_code: int | None
    until: str | None
    reason: str | None


async def record_report(
    session: AsyncSession,
    *,
    roster: list[RosterPlayer],
    player: str,
    status: str,
    now: datetime,
    team_code: str | None = None,
    reason: str | None = None,
    until: str | None = None,
    indefinite: bool = False,
    game_code: int | None = None,
    season_id: uuid.UUID | None = None,
    source: StatusSourceType = StatusSourceType.MANUAL,
) -> RecordedReport:
    """Validate and add one report. Raises :class:`InjuryInputError` on bad input.

    Scope, in order of precedence: ``game_code`` (that fixture), ``until`` (every
    game before that date), ``indefinite`` (until he plays again or is reported
    back). With none of them the report covers his team's next game, which is
    what a round's injury report usually means.
    """
    now = ensure_utc(now, field="now")
    parsed = parse_status(status)
    match = match_player(roster, player, team_code=team_code)

    game: Game | None = None
    expected_return: datetime | None = None
    if game_code is not None:
        stmt = select(Game).where(Game.game_code == game_code)
        if season_id is not None:
            stmt = stmt.where(Game.season_id == season_id)
        game = (await session.execute(stmt.order_by(Game.tip_off_utc.desc()).limit(1))).scalar_one_or_none()
        if game is None:
            raise InjuryInputError(f"no game with code {game_code}")
        if match.team_id not in (game.home_team_id, game.away_team_id):
            raise InjuryInputError(
                f"{match.full_name} ({match.team_code}) is not playing in game {game_code}"
            )
        scope = "game"
    elif until:
        try:
            day = datetime.strptime(until, "%Y-%m-%d").date()
        except ValueError as exc:
            raise InjuryInputError(f"--until must be YYYY-MM-DD, got {until!r}") from exc
        expected_return = datetime.combine(day, time(0, 0), tzinfo=UTC)
        if expected_return <= now:
            raise InjuryInputError(f"--until {until} is not in the future")
        scope = "until"
    elif indefinite:
        scope = "open"
    else:
        game = await next_game(session, match.team_id, after=now)
        if game is None:
            raise InjuryInputError(
                f"{match.team_code} has no scheduled game; use --until or --indefinite"
            )
        scope = "game"

    session.add(
        PlayerStatusEvent(
            id=uuid.uuid4(),
            player_id=match.player_id,
            team_id=match.team_id,
            game_id=game.id if game is not None else None,
            status=parsed,
            reason=(reason or None),
            expected_return_date=expected_return,
            source_type=source,
            confidence=None,
            event_timestamp=now,
            observed_at=now,
        )
    )
    return RecordedReport(
        player=match,
        status=parsed,
        scope=scope,
        game_code=game.game_code if game is not None else None,
        until=until if scope == "until" else None,
        reason=reason or None,
    )


#: Columns accepted by :func:`import_reports`. Only ``player`` and ``status`` are required.
CSV_COLUMNS = ("player", "status", "team", "reason", "until", "game_code", "indefinite")


async def import_reports(
    session: AsyncSession,
    path: Path,
    *,
    now: datetime,
    season_id: uuid.UUID | None = None,
) -> list[RecordedReport]:
    """Record every row of a CSV, or nothing if any row is invalid.

    All-or-nothing so a half-imported round can't leave some players reported and
    others silently missing. Errors list every bad row, not just the first.
    """
    now = ensure_utc(now, field="now")
    roster = await current_rosters(session, as_of=now)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"player", "status"} - set(reader.fieldnames or [])
        if missing:
            raise InjuryInputError(f"CSV needs a header row with: {', '.join(sorted(missing))}")
        rows = list(reader)

    recorded: list[RecordedReport] = []
    errors: list[str] = []
    async with session.begin_nested():
        for line, row in enumerate(rows, start=2):
            try:
                code = (row.get("game_code") or "").strip()
                recorded.append(
                    await record_report(
                        session,
                        roster=roster,
                        player=row.get("player") or "",
                        status=row.get("status") or "",
                        now=now,
                        team_code=(row.get("team") or "").strip() or None,
                        reason=(row.get("reason") or "").strip() or None,
                        until=(row.get("until") or "").strip() or None,
                        indefinite=(row.get("indefinite") or "").strip().lower() in ("1", "true", "yes", "y"),
                        game_code=int(code) if code else None,
                        season_id=season_id,
                    )
                )
            except (InjuryInputError, ValueError) as exc:
                errors.append(f"line {line}: {exc}")
        if errors:
            raise InjuryInputError("nothing was recorded:\n  " + "\n  ".join(errors))
    return recorded

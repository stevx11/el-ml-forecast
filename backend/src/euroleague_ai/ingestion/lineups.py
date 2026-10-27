"""When do starting lineups become knowable? A probe, before any feature.

Knowing the starting five changes a minutes projection more than any form
feature (roadmap p6-03). But a lineup feature is only worth building if the
lineups are published *before* a prediction can still be made, and nothing in
the data says when that is. The game stats endpoint carries a ``startFive``
flag per player once the game sheet exists; whether that sheet exists 30
minutes before tip-off, 5 minutes before, or only at tip-off is an empirical
question, and the answer decides the design:

* well before tip-off: a second, later prediction run with lineups is viable;
* only at tip-off: lineups are useless before the game, and the step is dropped.

:func:`probe` reads the endpoint fresh (never from the cache) for every game
near tip-off and records, per side, how many players are listed and how many
are flagged as starters. Run in a loop (``lineups probe --watch``), the log
shows the first moment each game's starters were visible, relative to tip-off.
Nothing is stored in the database: this is a measurement, not a data source yet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.competition import Season
from euroleague_ai.models.game import Game

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.euroleague import EuroLeagueProvider

logger = get_logger(__name__)

REPORTS_DIR = Path("reports") / "lineups"

#: Keep polling a game this long after its tip-off, so the log shows the
#: moment the starters appear even when that is only at tip-off.
AFTER_TIP_OFF = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class SideLineup:
    team: str | None
    players: int
    starters: int
    starter_names: tuple[str, ...]


def side_lineup(side: Any) -> SideLineup:
    """Players listed and players flagged ``startFive`` on one side."""
    if not isinstance(side, dict):
        return SideLineup(None, 0, 0, ())
    team: str | None = None
    players = 0
    names: list[str] = []
    for entry in side.get("players") or []:
        if not isinstance(entry, dict):
            continue
        player = entry.get("player") or {}
        if team is None:
            club = player.get("club") or {}
            team = club.get("code") if isinstance(club, dict) else None
        players += 1
        stats = entry.get("stats") or {}
        if isinstance(stats, dict) and stats.get("startFive"):
            person = player.get("person") or {}
            names.append(str(person.get("name") or "?") if isinstance(person, dict) else "?")
    return SideLineup(team, players, len(names), tuple(names))


@dataclass(frozen=True, slots=True)
class LineupProbe:
    """One look at one game's lineup."""

    fetched_at: datetime
    game_code: int
    tip_off_utc: datetime
    found: bool
    home: SideLineup
    away: SideLineup

    @property
    def minutes_to_tip_off(self) -> float:
        return round((self.tip_off_utc - self.fetched_at).total_seconds() / 60.0, 1)

    @property
    def complete(self) -> bool:
        """Both starting fives are on the sheet."""
        return self.home.starters == 5 and self.away.starters == 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at.isoformat(),
            "game_code": self.game_code,
            "tip_off_utc": self.tip_off_utc.isoformat(),
            "minutes_to_tip_off": self.minutes_to_tip_off,
            "found": self.found,
            "complete": self.complete,
            "home": asdict(self.home),
            "away": asdict(self.away),
        }


def read_lineup(
    game_code: int, tip_off_utc: datetime, payload: dict[str, Any] | None, *, now: datetime
) -> LineupProbe:
    """Turn one response (``None`` for a 404) into a probe record."""
    if payload is None:
        empty = SideLineup(None, 0, 0, ())
        return LineupProbe(now, game_code, tip_off_utc, False, empty, empty)
    return LineupProbe(
        now,
        game_code,
        tip_off_utc,
        True,
        side_lineup(payload.get("local")),
        side_lineup(payload.get("road")),
    )


async def games_near_tip_off(
    session: AsyncSession, *, season_code: str, now: datetime, hours: float
) -> list[Game]:
    """This season's games tipping off in the next ``hours``, or just tipped off."""
    stmt = (
        select(Game)
        .join(Season, Season.id == Game.season_id)
        .where(
            Season.code == season_code,
            Game.tip_off_utc >= now - AFTER_TIP_OFF,
            Game.tip_off_utc <= now + timedelta(hours=hours),
        )
        .order_by(Game.tip_off_utc)
    )
    return list((await session.execute(stmt)).scalars().all())


async def probe(
    provider: EuroLeagueProvider,
    season: SeasonRef,
    games: list[tuple[int, datetime]],
    *,
    now: datetime,
) -> list[LineupProbe]:
    """One fresh look at every game in ``games`` (game code, tip-off)."""
    out: list[LineupProbe] = []
    for game_code, tip_off in games:
        payload = await provider.fetch_game_stats_payload(season, game_code)
        out.append(read_lineup(game_code, tip_off, payload, now=now))
    return out


def append_log(records: list[LineupProbe], *, root: Path, now: datetime) -> Path:
    """Append records to the day's JSON-lines log, one line per look."""
    directory = root / REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y%m%d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict()) + "\n")
    return path


def first_complete(records: list[LineupProbe]) -> dict[int, tuple[float, float | None]]:
    """Per game: minutes to tip-off at the first look, and when both fives first showed.

    The second value is ``None`` when they never did in these records, and
    negative when they only showed after tip-off. When it equals the first look,
    the starters were already there: the true time is that early or earlier.
    """
    out: dict[int, tuple[float, float | None]] = {}
    for record in sorted(records, key=lambda r: r.fetched_at):
        first_look, seen = out.get(record.game_code, (record.minutes_to_tip_off, None))
        if record.complete and seen is None:
            seen = record.minutes_to_tip_off
        out[record.game_code] = (first_look, seen)
    return out

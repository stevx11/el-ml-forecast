"""The BasketNews EuroLeague injury report: one table, every club, updated daily.

Unlike an article, the page is a **complete snapshot**: every player a club is
missing or unsure of, with a status (Out, Doubtful, Game-time, Uncertain,
Expected, Ready), the rounds it covers and a comment. That makes two things
possible that prose cannot give:

* a player who **drops off** the table is back, so his earlier report is
  cleared rather than left to expire;
* the **Round** column says how long a status lasts: "Round 7" is that one game,
  "Round 7-9" runs through round 9, "Indefinitely" / "Long-term" / "Out for
  season" stay until he plays again or the row changes.

Only changes are written. Each fetch is compared with the last report this
source made for each player, and a new :class:`PlayerStatusEvent` is added only
when the status, the scope or the comment differ. An unchanged table writes
nothing, so the history reads as "what the report said, and from when".

Every fetch whose table differs from the last one is kept as a
:class:`NewsArticle`, so each report can point at the exact table it came from.

``robots.txt`` allows the article (checked 2026-10-21). The page is fetched once
per scheduled run: twice a day.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy import or_, select

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.core.logging import get_logger
from euroleague_ai.ingestion.injuries import RosterPlayer, current_rosters, normalise
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.models.game import Game
from euroleague_ai.models.news import NewsArticle
from euroleague_ai.news.store import ensure_source

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings

logger = get_logger(__name__)

#: Fewer rows than this means the page layout changed or the fetch returned
#: something else. Importing it would clear every injured player in the league.
MIN_ROWS = 20
MIN_TEAMS = 10

#: The page's status labels. "Uncertain" is mostly fringe players the coach has
#: not been using; it is flagged like a game-time decision and changes nothing.
STATUS_BY_LABEL: dict[str, AvailabilityStatus] = {
    "ready": AvailabilityStatus.AVAILABLE,
    "expected": AvailabilityStatus.PROBABLE,
    "probable": AvailabilityStatus.PROBABLE,
    "game time": AvailabilityStatus.QUESTIONABLE,
    "uncertain": AvailabilityStatus.QUESTIONABLE,
    "questionable": AvailabilityStatus.QUESTIONABLE,
    "doubtful": AvailabilityStatus.DOUBTFUL,
    "out": AvailabilityStatus.OUT,
    "suspended": AvailabilityStatus.SUSPENDED,
}

#: Round-scoped rows look this far around ``now`` for the team's game in that
#: round. Wide enough for "Round 4-12", narrow enough not to reach last season.
_ROUND_WINDOW_BACK = timedelta(days=21)
_ROUND_WINDOW_AHEAD = timedelta(days=120)

_TABLE = re.compile(r'<table[^>]*class="[^"]*table_02[^"]*"[^>]*>(.*?)</table>', re.S)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S)
_TEAM_CELL = re.compile(r"injury_reports__team-cell")
_ANCHOR = re.compile(r"<a\b[^>]*>(.*?)</a>", re.S)
_TAG = re.compile(r"<[^>]+>")
_MODIFIED = re.compile(r'"dateModified"\s*:\s*"([^"]+)"')
_ROUNDS = re.compile(r"round\s*(\d+)(?:\s*[-\u2013]\s*(\d+))?", re.I)
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


class ReportFormatError(ValueError):
    """The page did not contain a table we can trust."""


@dataclass(frozen=True, slots=True)
class ReportRow:
    team: str
    position: str
    name: str
    status_label: str
    rounds: str
    comment: str

    @property
    def status(self) -> AvailabilityStatus | None:
        return STATUS_BY_LABEL.get(normalise(self.status_label))

    @property
    def round_range(self) -> tuple[int, int] | None:
        """``(first, last)`` round the row covers, or ``None`` for open-ended."""
        match = _ROUNDS.search(self.rounds)
        if match is None:
            return None
        first = int(match.group(1))
        return first, int(match.group(2) or first)

    def line(self) -> str:
        return " | ".join(
            (self.team, self.position, self.name, self.status_label, self.rounds, self.comment)
        )


@dataclass(slots=True)
class ParsedReport:
    rows: list[ReportRow]
    modified_at: datetime | None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256("\n".join(r.line() for r in self.rows).encode()).hexdigest()


def _clean(fragment: str) -> str:
    return " ".join(html_lib.unescape(_TAG.sub(" ", fragment)).split())


def parse_injury_report(
    page: str, *, min_rows: int = MIN_ROWS, min_teams: int = MIN_TEAMS
) -> ParsedReport:
    """The table's rows, in page order. Raises if it does not look complete."""
    table = _TABLE.search(page)
    if table is None:
        raise ReportFormatError("no injury table on the page")
    rows: list[ReportRow] = []
    team: str | None = None
    for row in _ROW.findall(table.group(1)):
        if _TEAM_CELL.search(row):
            anchor = _ANCHOR.search(row)
            team = _clean(anchor.group(1) if anchor else row)
            continue
        cells = _CELL.findall(row)
        if len(cells) != 5 or team is None:
            continue
        position, name, status, rounds, comment = (_clean(c) for c in cells)
        if name:
            rows.append(ReportRow(team, position, name, status, rounds, comment))
    teams = {r.team for r in rows}
    if len(rows) < min_rows or len(teams) < min_teams:
        raise ReportFormatError(
            f"only {len(rows)} rows across {len(teams)} teams: the layout may have changed"
        )
    modified = _MODIFIED.search(page)
    modified_at = None
    if modified:
        try:
            modified_at = datetime.strptime(modified.group(1), "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            modified_at = None
    return ParsedReport(rows, modified_at)


# -- Matching ----------------------------------------------------------------


def _name_words(name: str) -> list[str]:
    return [w for w in normalise(name).split() if w not in _NAME_SUFFIXES]


def _surname_words(roster_name: str) -> list[str]:
    """Roster names are "SURNAME, FIRST"; the part before the comma."""
    surname = roster_name.split(",", 1)[0]
    return _name_words(surname)


def _full_matches(name: str, pool: Sequence[RosterPlayer]) -> list[RosterPlayer]:
    words = set(_name_words(name))
    return [p for p in pool if words and words == set(_name_words(p.full_name))]


def _surname_matches(name: str, pool: Sequence[RosterPlayer]) -> list[RosterPlayer]:
    """Players whose whole surname appears in ``name``: "Nikos Rogkavopoulos"
    finds ROGKAVOPOULOS, NIKOLAOS; "P.J. Dozier" finds DOZIER, PJ."""
    words = _name_words(name)
    out = []
    for player in pool:
        surname = _surname_words(player.full_name)
        if surname and len(surname) <= len(words) and words[-len(surname) :] == surname:
            out.append(player)
    return out


@dataclass(slots=True)
class MatchedRow:
    row: ReportRow
    player: RosterPlayer | None


def match_rows(rows: Sequence[ReportRow], roster: Sequence[RosterPlayer]) -> list[MatchedRow]:
    """Each row's rostered player, or ``None``.

    The page names clubs by sponsor ("Crvena Zvezda Meridianbet Belgrade"), so
    the club is found from its players instead: whichever team most of the
    block's unambiguous full-name matches belong to. Rows are then matched
    within that club, by full name first and then by surname, which is what
    catches "Nikos" for "Nikolaos" without guessing across the league.
    """
    blocks: dict[str, list[ReportRow]] = {}
    for row in rows:
        blocks.setdefault(row.team, []).append(row)

    out: list[MatchedRow] = []
    for block_rows in blocks.values():
        votes: Counter[uuid.UUID] = Counter()
        for row in block_rows:
            found = _full_matches(row.name, roster) or _surname_matches(row.name, roster)
            if len({p.team_id for p in found}) == 1:
                votes[found[0].team_id] += 1
        team_id = votes.most_common(1)[0][0] if votes else None
        pool = [p for p in roster if p.team_id == team_id] if team_id else []
        for row in block_rows:
            found = _full_matches(row.name, pool) or _surname_matches(row.name, pool)
            out.append(MatchedRow(row, found[0] if len(found) == 1 else None))
    return out


# -- Import ------------------------------------------------------------------


@dataclass(slots=True)
class ReportImportResult:
    rows: int = 0
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    written: int = 0
    cleared: int = 0
    unchanged: int = 0
    changes: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "matched": self.matched,
            "unmatched": len(self.unmatched),
            "written": self.written,
            "cleared": self.cleared,
            "unchanged": self.unchanged,
        }


async def _round_games(
    session: AsyncSession, team_ids: set[uuid.UUID], *, now: datetime
) -> dict[tuple[uuid.UUID, int], Game]:
    """Each team's game in each round, the one nearest ``now`` if a round repeats."""
    games = (
        await session.execute(
            select(Game).where(
                or_(Game.home_team_id.in_(team_ids), Game.away_team_id.in_(team_ids)),
                Game.round_number.is_not(None),
                Game.tip_off_utc >= now - _ROUND_WINDOW_BACK,
                Game.tip_off_utc <= now + _ROUND_WINDOW_AHEAD,
            )
        )
    ).scalars()
    out: dict[tuple[uuid.UUID, int], Game] = {}
    for game in games:
        assert game.round_number is not None
        for team in (game.home_team_id, game.away_team_id):
            key = (team, game.round_number)
            if team in team_ids and (
                key not in out or abs(game.tip_off_utc - now) < abs(out[key].tip_off_utc - now)
            ):
                out[key] = game
    return out


async def _latest_reports(
    session: AsyncSession, player_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, PlayerStatusEvent]:
    rows = (
        await session.execute(
            select(PlayerStatusEvent)
            .where(
                PlayerStatusEvent.source_type == StatusSourceType.REPORT,
                PlayerStatusEvent.player_id.in_(list(player_ids)),
            )
            .order_by(
                PlayerStatusEvent.observed_at.desc(), PlayerStatusEvent.event_timestamp.desc()
            )
        )
    ).scalars()
    latest: dict[uuid.UUID, PlayerStatusEvent] = {}
    for event in rows:
        latest.setdefault(event.player_id, event)
    return latest


async def _snapshot_article(
    session: AsyncSession, parsed: ParsedReport, *, url: str, now: datetime
) -> NewsArticle:
    """The stored copy of this exact table, created the first time it is seen."""
    digest = parsed.content_hash
    snapshot_url = f"{url}#table-{digest[:16]}"
    existing = (
        await session.execute(select(NewsArticle).where(NewsArticle.url == snapshot_url))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    source = await ensure_source(session, url)
    article = NewsArticle(
        id=uuid.uuid4(),
        source_id=source.id,
        url=snapshot_url,
        title="EuroLeague injury report",
        body="\n".join(r.line() for r in parsed.rows),
        language="en",
        published_at=parsed.modified_at,
        content_hash=digest,
        observed_at=now,
    )
    session.add(article)
    await session.flush()
    return article


def _same(
    event: PlayerStatusEvent,
    status: AvailabilityStatus,
    game_id: uuid.UUID | None,
    until: datetime | None,
    reason: str | None,
) -> bool:
    return (
        event.status is status
        and event.game_id == game_id
        and event.expected_return_date == until
        and (event.reason or None) == (reason or None)
    )


async def import_injury_report(
    session: AsyncSession,
    page: str,
    *,
    url: str,
    now: datetime,
    min_rows: int = MIN_ROWS,
    min_teams: int = MIN_TEAMS,
) -> ReportImportResult:
    """Read the page and write what changed since the last import.

    Raises :class:`ReportFormatError` without writing anything if the page does
    not look like a complete report.
    """
    now = ensure_utc(now, field="now")
    parsed = parse_injury_report(page, min_rows=min_rows, min_teams=min_teams)
    roster = await current_rosters(session, as_of=now)
    matched = match_rows(parsed.rows, roster)

    result = ReportImportResult(rows=len(parsed.rows))
    result.unmatched = [f"{m.row.name} ({m.row.team})" for m in matched if m.player is None]
    found = [m for m in matched if m.player is not None]
    result.matched = len(found)

    listed_teams = {m.player.team_id for m in found if m.player}
    games = await _round_games(session, listed_teams, now=now)
    # The publisher's last edit, never later than when we saw it.
    event_time = (
        min(ensure_utc(parsed.modified_at, field="modified_at"), now) if parsed.modified_at else now
    )

    team_players = [p.player_id for p in roster if p.team_id in listed_teams]
    latest = await _latest_reports(session, team_players)
    article: NewsArticle | None = None

    async def write(
        player: RosterPlayer,
        status: AvailabilityStatus,
        *,
        game: Game | None,
        until: datetime | None,
        reason: str | None,
        row: ReportRow | None,
    ) -> None:
        nonlocal article
        article = article or await _snapshot_article(session, parsed, url=url, now=now)
        session.add(
            PlayerStatusEvent(
                id=uuid.uuid4(),
                player_id=player.player_id,
                team_id=player.team_id,
                game_id=game.id if game else None,
                status=status,
                reason=(reason or "")[:512] or None,
                expected_return_date=until,
                source_type=StatusSourceType.REPORT,
                source_article_id=article.id,
                confidence=None,
                event_timestamp=event_time,
                observed_at=now,
            )
        )
        result.changes.append(
            {
                "team": player.team_code,
                "player": player.full_name,
                "status": str(status),
                "scope": "until" if until else "game" if game else "open",
                "game_code": game.game_code if game else None,
                "until": until.date().isoformat() if until else None,
                "rounds": row.rounds if row else None,
                "reason": reason,
            }
        )

    seen: set[uuid.UUID] = set()
    for m in found:
        player = m.player
        assert player is not None
        status = m.row.status
        if status is None or player.player_id in seen:
            continue
        seen.add(player.player_id)
        game: Game | None = None
        until: datetime | None = None
        span = m.row.round_range
        if span is not None:
            first, last = span
            if first == last:
                game = games.get((player.team_id, first))
            elif (end := games.get((player.team_id, last))) is not None:
                # Applies to every game up to and including the last round's.
                until = end.tip_off_utc + timedelta(minutes=1)
        reason = m.row.comment or None
        previous = latest.get(player.player_id)
        if previous is not None and _same(
            previous, status, game.id if game else None, until, reason
        ):
            result.unchanged += 1
            continue
        await write(player, status, game=game, until=until, reason=reason, row=m.row)
        result.written += 1

    # Off the table: back. Only for clubs that appear on this page, so a club
    # whose block failed to parse is not cleared wholesale.
    by_id = {p.player_id: p for p in roster}
    for player_id, previous in latest.items():
        if player_id in seen or previous.status is AvailabilityStatus.AVAILABLE:
            continue
        player = by_id.get(player_id)
        if player is None or player.team_id not in listed_teams:
            continue
        await write(
            player,
            AvailabilityStatus.AVAILABLE,
            game=None,
            until=None,
            reason="No longer on the injury report",
            row=None,
        )
        result.cleared += 1

    logger.info("injury report imported", extra=result.as_dict())
    return result


async def fetch_injury_report(url: str, *, settings: Settings) -> str:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.news_fetch_timeout_seconds),
        headers={"User-Agent": settings.injury_report_user_agent},
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def collect_injury_report(
    session: AsyncSession, *, settings: Settings, now: datetime
) -> ReportImportResult:
    """Fetch the configured report page and import it."""
    if not settings.injury_report_url:
        raise ReportFormatError("INJURY_REPORT_URL is not set")
    page = await fetch_injury_report(settings.injury_report_url, settings=settings)
    return await import_injury_report(session, page, url=settings.injury_report_url, now=now)

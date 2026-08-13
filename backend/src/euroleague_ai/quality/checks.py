"""Data quality checks over ingested seasons.

Run this before trusting the data for modelling. Ten seasons of real data always
contains surprises -- competition format changes, missing box scores, arithmetic
that does not reconcile -- and every one of them becomes a silent bias in a
trained model if it goes unnoticed.

Each check returns findings with a severity:

* ``ERROR``   -- the data is wrong. Fix before training.
* ``WARNING`` -- suspicious, may be legitimate. Investigate.
* ``INFO``    -- context, no action needed.

Checks are read-only and deliberately expressed as plain SQL against the stored
schema rather than through the ORM, so what they assert is obvious on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.logging import get_logger

logger = get_logger(__name__)


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(slots=True)
class Finding:
    """One issue, or one piece of context."""

    check: str
    severity: Severity
    message: str
    count: int = 0
    sample: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


async def _scalar(session: AsyncSession, sql: str, **params: object) -> int:
    result = await session.execute(text(sql), params)
    value = result.scalar()
    return int(value or 0)


async def _rows(session: AsyncSession, sql: str, limit: int = 5, **params: object) -> list[str]:
    result = await session.execute(text(sql), params)
    return [" | ".join(str(v) for v in row) for row in result.fetchmany(limit)]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

_SEASON_SUMMARY = """
SELECT s.code,
       count(DISTINCT g.id)                                              AS games,
       count(DISTINCT g.id) FILTER (WHERE g.status = 'FINISHED')         AS finished,
       count(DISTINCT gt.game_id) FILTER (WHERE ps.id IS NOT NULL)       AS with_box,
       count(ps.id)                                                      AS player_games
FROM seasons s
LEFT JOIN games g          ON g.season_id = s.id
LEFT JOIN game_teams gt    ON gt.game_id = g.id
LEFT JOIN player_game_stats ps ON ps.game_team_id = gt.id
GROUP BY s.code
ORDER BY s.code
"""


async def check_coverage(session: AsyncSession, report: QualityReport) -> None:
    """Per-season game and box score counts, and gaps between them."""
    result = await session.execute(text(_SEASON_SUMMARY))
    rows = result.fetchall()

    if not rows:
        report.add(Finding("coverage", Severity.ERROR, "no seasons ingested"))
        return

    summary = [
        f"{code}: {games} games, {finished} finished, {with_box} with box scores, "
        f"{player_games} player-games"
        for code, games, finished, with_box, player_games in rows
    ]
    total_player_games = sum(row[4] for row in rows)
    report.add(
        Finding(
            "coverage",
            Severity.INFO,
            f"{len(rows)} seasons, {total_player_games:,} player-games total",
            count=total_player_games,
            sample=summary,
        )
    )

    # A finished game with no box score is a hole in every form feature that
    # spans it, and holes are invisible once the features are computed.
    missing = [
        f"{code}: {finished - with_box} finished games lack box scores"
        for code, _games, finished, with_box, _pg in rows
        if finished > with_box
    ]
    if missing:
        report.add(
            Finding(
                "coverage.missing_box_scores",
                Severity.WARNING,
                "finished games without box scores -- re-run `ingest stats` for these seasons",
                count=len(missing),
                sample=missing,
            )
        )

    # The 30-round round-robin format started in 2016-17. Wildly different game
    # counts usually mean an older format, not corrupt data -- but schedule
    # features are not comparable across the boundary.
    unusual = [
        f"{code}: {games} games"
        for code, games, *_ in rows
        if games and not (150 <= games <= 500)
    ]
    if unusual:
        report.add(
            Finding(
                "coverage.unusual_schedule_size",
                Severity.INFO,
                "season sizes outside the typical range -- likely a different competition format",
                count=len(unusual),
                sample=unusual,
            )
        )


# ---------------------------------------------------------------------------
# Box score arithmetic
# ---------------------------------------------------------------------------


async def check_scoring_arithmetic(session: AsyncSession, report: QualityReport) -> None:
    """Points must equal 2*FG2M + 3*FG3M + FTM.

    If this fails, either the mapping is wrong or the upstream feed is
    inconsistent. Either way every points-derived feature inherits the error.
    """
    sql = """
    SELECT count(*) FROM player_game_stats
    WHERE points <> (2 * fg2m + 3 * fg3m + ftm)
    """
    count = await _scalar(session, sql)
    if count:
        sample = await _rows(
            session,
            """
            SELECT p.full_name, s.points, s.fg2m, s.fg3m, s.ftm
            FROM player_game_stats s JOIN players p ON p.id = s.player_id
            WHERE s.points <> (2 * s.fg2m + 3 * s.fg3m + s.ftm)
            LIMIT 5
            """,
        )
        report.add(
            Finding(
                "arithmetic.points",
                Severity.ERROR,
                "points do not equal 2*FG2M + 3*FG3M + FTM",
                count=count,
                sample=sample,
            )
        )


async def check_rebound_totals(session: AsyncSession, report: QualityReport) -> None:
    """Total rebounds must equal offensive + defensive."""
    count = await _scalar(
        session,
        "SELECT count(*) FROM player_game_stats "
        "WHERE total_rebounds <> (offensive_rebounds + defensive_rebounds)",
    )
    if count:
        report.add(
            Finding(
                "arithmetic.rebounds",
                Severity.ERROR,
                "total rebounds do not equal offensive + defensive",
                count=count,
            )
        )


async def check_made_not_exceeding_attempted(
    session: AsyncSession, report: QualityReport
) -> None:
    """Makes cannot exceed attempts. A violation means the mapping is misaligned."""
    count = await _scalar(
        session,
        "SELECT count(*) FROM player_game_stats "
        "WHERE fg2m > fg2a OR fg3m > fg3a OR ftm > fta",
    )
    if count:
        report.add(
            Finding(
                "arithmetic.made_gt_attempted",
                Severity.ERROR,
                "made shots exceed attempts -- likely a column mapping error",
                count=count,
            )
        )


async def check_team_totals_reconcile(session: AsyncSession, report: QualityReport) -> None:
    """Team totals should match the sum of their players' lines.

    Small differences are legitimate: team rebounds and team turnovers are not
    attributed to any player. Points, however, must reconcile exactly.
    """
    sql = """
    WITH player_sums AS (
        SELECT gt.id AS game_team_id, sum(s.points) AS player_points
        FROM game_teams gt JOIN player_game_stats s ON s.game_team_id = gt.id
        GROUP BY gt.id
    )
    SELECT count(*)
    FROM team_game_stats tgs JOIN player_sums ps ON ps.game_team_id = tgs.game_team_id
    WHERE tgs.points <> ps.player_points
    """
    count = await _scalar(session, sql)
    if count:
        report.add(
            Finding(
                "reconciliation.team_points",
                Severity.ERROR,
                "team total points differ from the sum of player points",
                count=count,
            )
        )


async def check_game_scores_match_stats(session: AsyncSession, report: QualityReport) -> None:
    """The scoreboard on the game row should agree with the box score."""
    sql = """
    SELECT count(*)
    FROM games g
    JOIN game_teams gt      ON gt.game_id = g.id
    JOIN team_game_stats t  ON t.game_team_id = gt.id
    WHERE g.status = 'FINISHED'
      AND ((gt.side = 'HOME' AND g.home_score IS NOT NULL AND g.home_score <> t.points)
        OR (gt.side = 'AWAY' AND g.away_score IS NOT NULL AND g.away_score <> t.points))
    """
    count = await _scalar(session, sql)
    if count:
        report.add(
            Finding(
                "reconciliation.scoreboard",
                Severity.WARNING,
                "game scoreboard disagrees with box score totals",
                count=count,
            )
        )


# ---------------------------------------------------------------------------
# Playing time
# ---------------------------------------------------------------------------


async def check_minutes_plausible(session: AsyncSession, report: QualityReport) -> None:
    """No player can log more than the game's length.

    2400s is regulation (40 min); overtime adds 5 min each, so the bound is
    generous rather than exact.
    """
    count = await _scalar(
        session, "SELECT count(*) FROM player_game_stats WHERE seconds_played > 3600"
    )
    if count:
        report.add(
            Finding(
                "minutes.implausible",
                Severity.ERROR,
                "player logged more than 60 minutes in a single game",
                count=count,
            )
        )


async def check_dnp_consistency(session: AsyncSession, report: QualityReport) -> None:
    """A DNP must have zero minutes, and vice versa.

    This distinction drives whether a game is excluded from form windows or
    averaged in as a zero, so an inconsistency here silently biases every
    per-game average.
    """
    count = await _scalar(
        session,
        "SELECT count(*) FROM player_game_stats "
        "WHERE (did_not_play AND seconds_played > 0) "
        "   OR (NOT did_not_play AND seconds_played = 0)",
    )
    if count:
        report.add(
            Finding(
                "minutes.dnp_flag",
                Severity.WARNING,
                "did_not_play disagrees with seconds_played",
                count=count,
            )
        )


async def check_starters_per_game(session: AsyncSession, report: QualityReport) -> None:
    """Basketball has five starters per side."""
    sql = """
    SELECT count(*) FROM (
        SELECT gt.id, count(*) FILTER (WHERE s.is_starter) AS starters
        FROM game_teams gt
        JOIN player_game_stats s ON s.game_team_id = gt.id
        GROUP BY gt.id
        HAVING count(*) FILTER (WHERE s.is_starter) <> 5
    ) bad
    """
    count = await _scalar(session, sql)
    if count:
        sample = await _rows(
            session,
            """
            SELECT g.game_code, gt.side, count(*) FILTER (WHERE s.is_starter) AS starters
            FROM game_teams gt
            JOIN games g ON g.id = gt.game_id
            JOIN player_game_stats s ON s.game_team_id = gt.id
            GROUP BY g.game_code, gt.side
            HAVING count(*) FILTER (WHERE s.is_starter) <> 5
            LIMIT 5
            """,
        )
        report.add(
            Finding(
                "lineup.starter_count",
                Severity.WARNING,
                "team-games without exactly five starters -- affects starter features",
                count=count,
                sample=sample,
            )
        )


# ---------------------------------------------------------------------------
# Structure and identity
# ---------------------------------------------------------------------------


async def check_two_sides_per_game(session: AsyncSession, report: QualityReport) -> None:
    """Invariant 4 from the data model: exactly one HOME and one AWAY."""
    count = await _scalar(
        session,
        "SELECT count(*) FROM (SELECT game_id FROM game_teams "
        "GROUP BY game_id HAVING count(*) <> 2) bad",
    )
    if count:
        report.add(
            Finding(
                "structure.sides", Severity.ERROR, "games without exactly two sides", count=count
            )
        )


async def check_duplicate_players(session: AsyncSession, report: QualityReport) -> None:
    """Two player rows with the same name and birth date are one person.

    Identity resolution failures split a career in half, which halves every
    form window and makes a veteran look like a rookie.
    """
    sql = """
    SELECT count(*) FROM (
        SELECT full_name, birth_date FROM players
        WHERE birth_date IS NOT NULL
        GROUP BY full_name, birth_date HAVING count(*) > 1
    ) dupes
    """
    count = await _scalar(session, sql)
    if count:
        sample = await _rows(
            session,
            """
            SELECT full_name, birth_date, count(*)
            FROM players WHERE birth_date IS NOT NULL
            GROUP BY full_name, birth_date HAVING count(*) > 1
            ORDER BY count(*) DESC LIMIT 5
            """,
        )
        report.add(
            Finding(
                "identity.duplicate_players",
                Severity.ERROR,
                "same name and birth date stored as multiple players",
                count=count,
                sample=sample,
            )
        )


async def check_player_in_both_sides(session: AsyncSession, report: QualityReport) -> None:
    """Nobody plays for both teams in one game."""
    sql = """
    SELECT count(*) FROM (
        SELECT gt.game_id, s.player_id
        FROM player_game_stats s JOIN game_teams gt ON gt.id = s.game_team_id
        GROUP BY gt.game_id, s.player_id HAVING count(DISTINCT gt.side) > 1
    ) bad
    """
    count = await _scalar(session, sql)
    if count:
        report.add(
            Finding(
                "identity.player_both_sides",
                Severity.ERROR,
                "a player appears on both sides of the same game",
                count=count,
            )
        )


async def check_orphan_stats(session: AsyncSession, report: QualityReport) -> None:
    """Box scores attached to games that are not marked finished."""
    sql = """
    SELECT count(*) FROM player_game_stats s
    JOIN game_teams gt ON gt.id = s.game_team_id
    JOIN games g ON g.id = gt.game_id
    WHERE g.status <> 'FINISHED'
    """
    count = await _scalar(session, sql)
    if count:
        report.add(
            Finding(
                "structure.stats_for_unfinished_game",
                Severity.WARNING,
                "box scores exist for games not marked FINISHED",
                count=count,
            )
        )


async def check_temporal_integrity(session: AsyncSession, report: QualityReport) -> None:
    """observed_at must be present and not absurdly before the event.

    The leakage machinery depends entirely on observed_at being meaningful.
    """
    backdated = await _scalar(
        session,
        """
        SELECT count(*) FROM player_game_stats s
        JOIN game_teams gt ON gt.id = s.game_team_id
        JOIN games g ON g.id = gt.game_id
        WHERE s.observed_at < g.tip_off_utc
        """,
    )
    if backdated:
        report.add(
            Finding(
                "temporal.observed_before_event",
                Severity.INFO,
                "box scores recorded with observed_at before tip-off. Expected for a "
                "historical backfill (we ingested old games today, stamped with the "
                "ingest clock) -- but it means these rows carry no real 'when we knew "
                "it' signal, so availability-style features cannot be reconstructed "
                "for them.",
                count=backdated,
            )
        )

    duplicate_tipoffs = await _scalar(
        session,
        """
        SELECT count(*) FROM (
            SELECT season_id, home_team_id, away_team_id, tip_off_utc
            FROM games GROUP BY 1, 2, 3, 4 HAVING count(*) > 1
        ) dupes
        """,
    )
    if duplicate_tipoffs:
        report.add(
            Finding(
                "temporal.duplicate_fixtures",
                Severity.WARNING,
                "same matchup at the same tip-off appears more than once",
                count=duplicate_tipoffs,
            )
        )


async def check_roster_intervals(session: AsyncSession, report: QualityReport) -> None:
    """Players should have a roster stint, and stints should not overlap."""
    without_stint = await _scalar(
        session,
        "SELECT count(*) FROM players p "
        "WHERE NOT EXISTS (SELECT 1 FROM player_team_stints s WHERE s.player_id = p.id)",
    )
    if without_stint:
        report.add(
            Finding(
                "roster.players_without_stints",
                Severity.WARNING,
                "players with no roster registration -- usually mid-season signings first "
                "seen in a box score",
                count=without_stint,
            )
        )

    overlapping = await _scalar(
        session,
        """
        SELECT count(*) FROM player_team_stints a
        JOIN player_team_stints b
          ON a.player_id = b.player_id AND a.season_id = b.season_id
         AND a.id <> b.id AND a.team_id <> b.team_id
        WHERE a.effective_from < coalesce(b.effective_until, 'infinity'::timestamptz)
          AND b.effective_from < coalesce(a.effective_until, 'infinity'::timestamptz)
        """,
    )
    if overlapping:
        report.add(
            Finding(
                "roster.overlapping_stints",
                Severity.WARNING,
                "a player is registered to two teams at the same time in one season",
                count=overlapping,
            )
        )


ALL_CHECKS = (
    check_coverage,
    check_scoring_arithmetic,
    check_rebound_totals,
    check_made_not_exceeding_attempted,
    check_team_totals_reconcile,
    check_game_scores_match_stats,
    check_minutes_plausible,
    check_dnp_consistency,
    check_starters_per_game,
    check_two_sides_per_game,
    check_duplicate_players,
    check_player_in_both_sides,
    check_orphan_stats,
    check_temporal_integrity,
    check_roster_intervals,
)


async def run_quality_checks(session: AsyncSession) -> QualityReport:
    """Run every check and return the combined report."""
    report = QualityReport()
    for check in ALL_CHECKS:
        try:
            await check(session, report)
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            logger.exception("quality check failed", extra={"check": check.__name__})
            report.add(
                Finding(check.__name__, Severity.ERROR, f"check itself failed: {exc}")
            )
    return report

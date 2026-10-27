"""Operational CLI.

The scheduling story today: this plus cron. Every command is a thin wrapper
around a job function in :mod:`euroleague_ai.ingestion.jobs`, so promoting them
to Celery tasks later is a decorator, not a rewrite.

    euroleague-ai ingest all --season E2025
    euroleague-ai ingest games --season E2025 --round 12
    euroleague-ai features build --game-code 1 --season E2025
    euroleague-ai news collect
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Annotated, Any

import typer

from euroleague_ai.core.config import get_settings
from euroleague_ai.core.logging import configure_logging, get_logger
from euroleague_ai.db.session import session_scope
from euroleague_ai.ingestion.jobs import (
    ingest_all,
    ingest_game_stats,
    ingest_games,
    ingest_players,
    ingest_teams,
)
from euroleague_ai.providers.factory import build_provider, season_ref

if TYPE_CHECKING:  # pragma: no cover - typing only
    from euroleague_ai.ingestion.injuries import RosterPlayer
    from euroleague_ai.quality.checks import QualityReport
    from euroleague_ai.repositories.availability import StatusReport

logger = get_logger(__name__)

app = typer.Typer(help="EuroLeague AI operations", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingestion jobs", no_args_is_help=True)
features_app = typer.Typer(help="Feature computation", no_args_is_help=True)
news_app = typer.Typer(help="News collection", no_args_is_help=True)
quality_app = typer.Typer(help="Data quality", no_args_is_help=True)
ml_app = typer.Typer(help="Datasets, baselines, evaluation", no_args_is_help=True)
maintenance_app = typer.Typer(help="Data repair", no_args_is_help=True)
pipeline_app = typer.Typer(help="Scheduled end-to-end runs", no_args_is_help=True)
injuries_app = typer.Typer(
    help="Injury and availability reports. Enter each round's report before tip-off.",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(news_app, name="news")
app.add_typer(quality_app, name="quality")
app.add_typer(ml_app, name="ml")
app.add_typer(maintenance_app, name="maintenance")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(injuries_app, name="injuries")

SeasonOption = Annotated[str, typer.Option("--season", "-s", help="Season code, e.g. E2025.")]


def _season_default() -> str:
    return get_settings().euroleague_default_season


def _run[T](factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async job with logging configured and the provider always closed."""
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    return asyncio.run(factory())


def _report(label: str, result: object) -> None:
    typer.echo(f"{label}: {result}")


@ingest_app.command("teams")
def cmd_ingest_teams(season: SeasonOption = "") -> None:
    """Fetch and store every club registered in the season."""

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings)
        try:
            async with session_scope() as session:
                return await ingest_teams(
                    session, provider, season_ref(season or _season_default(), settings)
                )
        finally:
            await provider.aclose()

    _report("teams", _run(job))


@ingest_app.command("players")
def cmd_ingest_players(season: SeasonOption = "") -> None:
    """Fetch and store player biography and roster intervals.

    Run after ``ingest teams``: registrations referencing an unknown club are
    skipped rather than inventing a team row.
    """

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings)
        try:
            async with session_scope() as session:
                return await ingest_players(
                    session, provider, season_ref(season or _season_default(), settings)
                )
        finally:
            await provider.aclose()

    _report("players", _run(job))


@ingest_app.command("games")
def cmd_ingest_games(
    season: SeasonOption = "",
    round_number: Annotated[
        int | None, typer.Option("--round", "-r", help="Restrict to one round.")
    ] = None,
) -> None:
    """Fetch and store the schedule."""

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings)
        try:
            async with session_scope() as session:
                return await ingest_games(
                    session,
                    provider,
                    season_ref(season or _season_default(), settings),
                    round_number=round_number,
                )
        finally:
            await provider.aclose()

    _report("games", _run(job))


@ingest_app.command("stats")
def cmd_ingest_stats(
    season: SeasonOption = "",
    game_code: Annotated[
        list[int] | None,
        typer.Option("--game-code", "-g", help="Specific game codes. Repeatable."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help="Re-fetch games that already have box scores (after a stat correction).",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help="Sit through long rate-limit cooldowns instead of stopping. "
            "For unattended backfills; can block for many minutes.",
        ),
    ] = False,
) -> None:
    """Fetch and store box scores for finished games.

    Resumes by default: games that already have box scores are skipped, so
    re-running after an interrupted backfill continues where it stopped.
    Idempotent either way.
    """

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings, wait_out_cooldowns=wait)
        try:
            async with session_scope() as session:
                return await ingest_game_stats(
                    session,
                    provider,
                    season_ref(season or _season_default(), settings),
                    game_codes=list(game_code) if game_code else None,
                    refresh_existing=refresh,
                )
        finally:
            await provider.aclose()

    _report("stats", _run(job))


@ingest_app.command("all")
def cmd_ingest_all(
    season: SeasonOption = "",
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help="Sit through long rate-limit cooldowns instead of stopping. "
            "Use for unattended historical backfills.",
        ),
    ] = False,
) -> None:
    """Run every ingestion job in dependency order.

    This is the "make it work" command: teams, then players, then games, then
    box scores. Each stage commits before the next, and box scores resume, so
    interrupting and re-running is always safe.
    """

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings, wait_out_cooldowns=wait)
        try:
            async with session_scope() as session:
                return await ingest_all(
                    session, provider, season_ref(season or _season_default(), settings)
                )
        finally:
            await provider.aclose()

    _report("all", _run(job))


@ingest_app.command("seasons")
def cmd_ingest_seasons(
    start: Annotated[int, typer.Option("--from", help="First season start year.")] = 2016,
    end: Annotated[int, typer.Option("--to", help="Last season start year.")] = 2025,
    wait: Annotated[bool, typer.Option("--wait", help="Sit through rate-limit cooldowns.")] = True,
) -> None:
    """Backfill a range of seasons, newest first.

    Newest first on purpose: recent seasons matter most for modelling, so if the
    backfill is interrupted you keep the data you care about. Everything is
    resumable, so re-running continues where it stopped.

    The default range starts at 2016 because the 2016-17 season introduced the
    current 30-round round-robin format. Earlier seasons used a group stage plus
    Top 16, which gives different schedule density and rest patterns -- usable,
    but not directly comparable for schedule features.
    """
    codes = [f"E{year}" for year in range(end, start - 1, -1)]
    typer.echo(f"Backfilling {len(codes)} seasons: {', '.join(codes)}")

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings, wait_out_cooldowns=wait)
        summary: dict[str, str] = {}
        try:
            for code in codes:
                typer.echo(f"--- {code} ---")
                try:
                    async with session_scope() as session:
                        result = await ingest_all(session, provider, season_ref(code, settings))
                    summary[code] = f"{result.rows_written} rows"
                except Exception as exc:
                    # One bad season must not abandon the rest of the range.
                    logger.exception("season failed", extra={"season": code})
                    summary[code] = f"FAILED: {type(exc).__name__}"
                typer.echo(f"    {code}: {summary[code]}")
        finally:
            await provider.aclose()
        return summary

    _report("seasons", _run(job))


@features_app.command("backfill")
def cmd_features_backfill(
    start: Annotated[
        int | None, typer.Option("--from", help="First season start year, e.g. 2016.")
    ] = None,
    end: Annotated[int | None, typer.Option("--to", help="Last season start year.")] = None,
    min_round: Annotated[
        int,
        typer.Option(
            "--min-round",
            help="Skip rounds below this. Since feature set v3 the form windows "
            "cross season boundaries, so early rounds carry real signal and are "
            "kept by default. Mirrors features.backfill.DEFAULT_MIN_ROUND.",
        ),
    ] = 1,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Stop after N games. Use for a smoke run first."),
    ] = None,
    candidates: Annotated[
        bool,
        typer.Option(
            "--candidates/--no-candidates",
            help="Also snapshot roster candidates who did not play. The "
            "play-probability model needs these rows.",
        ),
    ] = True,
) -> None:
    """Build feature snapshots across finished games.

    This is the Phase 1 deliverable: roughly 100k training rows over ten seasons.
    Resumable -- games already snapshotted at the current feature set version are
    skipped, so interrupting and re-running is safe and cheap.

    Start with a smoke run before committing to the full pass::

        euroleague-ai features backfill --limit 20
    """
    from euroleague_ai.features.backfill import backfill_snapshots

    codes: list[str] | None = None
    if start is not None or end is not None:
        first = start if start is not None else 2000
        last = end if end is not None else 2025
        codes = [f"E{year}" for year in range(first, last + 1)]

    async def job() -> object:
        settings = get_settings()
        async with session_scope() as session:
            return await backfill_snapshots(
                session,
                season_codes=codes,
                min_round=min_round,
                limit=limit,
                settings=settings,
                include_candidates=candidates,
            )

    _report("snapshots", _run(job))


@features_app.command("names")
def cmd_feature_names() -> None:
    """Print the feature vector's column names and count."""
    from euroleague_ai.features.registry import FEATURE_SET_VERSION, feature_names

    async def job() -> list[str]:
        async with session_scope() as session:
            return await feature_names(session)

    names = _run(job)
    typer.echo(f"feature set {FEATURE_SET_VERSION}: {len(names)} features\n")
    for name in names:
        typer.echo(f"  {name}")


@features_app.command("build")
def cmd_build_features(
    season: SeasonOption = "",
    game_code: Annotated[
        int, typer.Option("--game-code", "-g", help="Game to build snapshots for.")
    ] = 0,
) -> None:
    """Build and persist feature snapshots for every player in a game.

    Uses the default cutoff (``tip_off - PREDICTION_LEAD_MINUTES``), so the
    snapshot reflects only what was knowable before that instant.
    """
    from sqlalchemy import select

    from euroleague_ai.features.snapshot import build_and_store_player_snapshot
    from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats

    async def job() -> object:
        settings = get_settings()
        async with session_scope() as session:
            season_code = season or _season_default()
            game = (
                await session.execute(
                    select(Game)
                    .join(Game.season)
                    .where(Game.game_code == game_code)
                    .where(Game.season.has(code=season_code))
                )
            ).scalar_one_or_none()
            if game is None:
                raise typer.BadParameter(f"game {game_code} not found in season {season_code}")

            rows = (
                await session.execute(
                    select(PlayerGameStats.player_id, GameTeam.team_id)
                    .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                    .where(GameTeam.game_id == game.id)
                )
            ).all()

            built = 0
            for player_id, team_id in rows:
                await build_and_store_player_snapshot(
                    session,
                    player_id=player_id,
                    game_id=game.id,
                    team_id=team_id,
                    settings=settings,
                )
                built += 1
            return {"snapshots": built}

    _report("features", _run(job))


@news_app.command("collect")
def cmd_collect_news() -> None:
    """Poll every configured feed, store new articles, read injuries out of them.

    Set NEWS_FEED_URLS, e.g. https://www.eurohoops.net/en/euroleague/feed/.
    Runs automatically in the scheduled pipeline, before predictions.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.news.pipeline import collect_news

    settings = get_settings()
    if not settings.news_feed_urls:
        typer.secho(
            "  No feeds configured. Set NEWS_FEED_URLS in .env, e.g.\n"
            "  NEWS_FEED_URLS=https://www.eurohoops.net/en/euroleague/feed/",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            result = await collect_news(session, settings=settings, now=SystemClock().now())
            await session.commit()
            return result.as_dict()

    r = _run(job)
    for url, count in r["feeds"].items():
        failed = url in r["failed_feeds"]
        typer.secho(
            f"  {url}: {'could not be read' if failed else f'{count} new article(s)'}",
            fg=typer.colors.RED if failed else None,
        )
    typer.echo(
        f"  {r['readings']} injury reading(s), {r['ruled_out']} confident enough to rule a player "
        "out"
    )
    if r["readings"]:
        typer.echo("  see them with: euroleague-ai news injuries")


@news_app.command("extract")
def cmd_news_extract(
    days: Annotated[
        int, typer.Option("--days", help="Re-read articles from the last N days.")
    ] = 14,
    replace: Annotated[
        bool,
        typer.Option(
            "--replace",
            help=(
                "Re-read every article in the window and drop the readings taken from them before."
            ),
        ),
    ] = False,
) -> None:
    """Re-read stored articles that the current rules haven't read yet.

    With --replace, every article in the window is read again and its earlier
    readings are deleted, so a misreading fixed in the rules disappears.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.news.pipeline import extract_articles, reread_articles, unread_articles

    async def job() -> tuple[int, int, int]:
        now = SystemClock().now()
        async with session_scope() as session:
            if replace:
                result = await reread_articles(session, now=now, days=days)
            else:
                articles = await unread_articles(session, now=now, days=days)
                result = (len(articles), *await extract_articles(session, articles, now=now))
            await session.commit()
            return result

    count, readings, ruled_out = _run(job)
    typer.echo(
        f"  read {count} article(s): {readings} injury reading(s), {ruled_out} ruling a player out"
    )


@news_app.command("injuries")
def cmd_news_injuries(
    days: Annotated[int, typer.Option("--days", help="How far back to list.")] = 7,
) -> None:
    """Injury readings taken from news, with the sentence each came from.

    The date is when the article was published; the scope says how long the
    reading lasts: one game, until a date, or open (until he plays again).
    """
    from datetime import timedelta

    from sqlalchemy import select

    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.features.injury_status import NEWS_RULE_OUT_CONFIDENCE, RULES_OUT
    from euroleague_ai.models.competition import Team
    from euroleague_ai.models.news import NewsArticle, NewsEvent
    from euroleague_ai.models.people import Player

    async def job() -> object:
        since = SystemClock().now() - timedelta(days=days)
        async with session_scope() as session:
            return (
                await session.execute(
                    select(NewsEvent, Player.full_name, Team.code, NewsArticle.title)
                    .join(Player, Player.id == NewsEvent.player_id)
                    .join(NewsArticle, NewsArticle.id == NewsEvent.article_id)
                    .outerjoin(Team, Team.id == NewsEvent.team_id)
                    .where(NewsEvent.observed_at >= since)
                    .order_by(NewsEvent.observed_at.desc())
                )
            ).all()

    rows = _run(job)
    if not rows:
        typer.echo(f"  no injury readings in the last {days} day(s)")
        return
    from euroleague_ai.models.enums import AvailabilityStatus

    for event, name, team_code, title in rows:  # type: ignore[attr-defined]
        payload = event.payload
        status = AvailabilityStatus(payload.get("status", "UNKNOWN"))
        applies = status in RULES_OUT and float(event.confidence) >= NEWS_RULE_OUT_CONFIDENCE
        published = (payload.get("published_at") or event.observed_at.isoformat())[5:10]
        scope = payload.get("scope") or "open"
        if scope == "game" and payload.get("game_code") is not None:
            scope = f"game {payload['game_code']}"
        elif scope == "until" and payload.get("until"):
            scope = f"until {payload['until'][5:]}"
        typer.secho(
            f"  {published}  {team_code or '?':<4} {name:<26} "
            f"{status!s:<13} {float(event.confidence):.2f}  {scope:<12}"
            + ("  rules out" if applies else ""),
            fg=typer.colors.RED if applies else None,
        )
        typer.echo(f'      "{event.payload.get("sentence", "")[:150]}"')
        typer.echo(f"      {title[:90]}")


async def _season_id(session: Any, season: str) -> Any:
    from sqlalchemy import select

    from euroleague_ai.models.competition import Season

    return (
        await session.execute(select(Season.id).where(Season.code == (season or _season_default())))
    ).scalar_one_or_none()


def _print_report(recorded: Any) -> None:
    scope = (
        f"game {recorded.game_code}"
        if recorded.scope == "game"
        else f"until {recorded.until}"
        if recorded.scope == "until"
        else "until he plays again"
    )
    reason = f"  ({recorded.reason})" if recorded.reason else ""
    typer.echo(
        f"  {recorded.player.team_code:<4} {recorded.player.full_name:<28} "
        f"{recorded.status!s:<13} {scope}{reason}"
    )


@injuries_app.command("set")
def cmd_injuries_set(
    player: Annotated[
        str, typer.Argument(help='Name or player code, e.g. "Petrovic" or "Marko Petrovic".')
    ],
    status: Annotated[
        str,
        typer.Argument(
            help="out, suspended, doubtful, questionable (gtd), probable, available (back)."
        ),
    ],
    team: Annotated[
        str | None,
        typer.Option("--team", "-t", help="Club code, to pick between players with one name."),
    ] = None,
    reason: Annotated[
        str | None, typer.Option("--reason", "-r", help='e.g. "knee", "illness".')
    ] = None,
    until: Annotated[
        str | None,
        typer.Option("--until", help="Out until this date (YYYY-MM-DD), for every game before it."),
    ] = None,
    indefinite: Annotated[
        bool,
        typer.Option(
            "--indefinite", help="No return date: applies until he plays again or is set back."
        ),
    ] = False,
    game_code: Annotated[
        int | None,
        typer.Option("--game-code", "-g", help="Only this fixture. Default: his team's next game."),
    ] = None,
    season: SeasonOption = "",
) -> None:
    """Record one player's status, stamped with the time you entered it.

    With no --game-code, --until or --indefinite, it covers his team's next game.
    A player recorded as out or suspended gets P(plays) = 0 in the next
    `ml predict`, and his minutes go to his teammates. Enter it before the
    prediction cutoff (60 minutes before tip-off) to count.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.ingestion.injuries import InjuryInputError, current_rosters, record_report

    async def job() -> object:
        now = SystemClock().now()
        async with session_scope() as session:
            roster = await current_rosters(session, as_of=now)
            recorded = await record_report(
                session,
                roster=roster,
                player=player,
                status=status,
                now=now,
                team_code=team,
                reason=reason,
                until=until,
                indefinite=indefinite,
                game_code=game_code,
                season_id=await _season_id(session, season),
            )
            await session.commit()
            return recorded

    try:
        recorded = _run(job)
    except InjuryInputError as exc:
        typer.secho(f"  {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("  recorded:", fg=typer.colors.GREEN)
    _print_report(recorded)


@injuries_app.command("import")
def cmd_injuries_import(
    path: Annotated[str, typer.Argument(help="CSV with a header row.")],
    season: SeasonOption = "",
) -> None:
    """Record a whole round's report from a CSV, all or nothing.

    Columns: player, status (required); team, reason, until (YYYY-MM-DD),
    game_code, indefinite (yes/no) (optional). Example:

        player,status,team,reason,until
        Petrovic,out,PAR,knee,
        Hezonja,questionable,MAD,ankle,
        Mirotic,out,,hamstring,2026-11-15
    """
    from pathlib import Path

    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.ingestion.injuries import InjuryInputError, import_reports

    async def job() -> object:
        async with session_scope() as session:
            recorded = await import_reports(
                session,
                Path(path),
                now=SystemClock().now(),
                season_id=await _season_id(session, season),
            )
            await session.commit()
            return recorded

    try:
        recorded = _run(job)
    except (InjuryInputError, FileNotFoundError) as exc:
        typer.secho(f"  {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"  recorded {len(recorded)} report(s):", fg=typer.colors.GREEN)  # type: ignore[arg-type]
    for item in recorded:  # type: ignore[attr-defined]
        _print_report(item)


@injuries_app.command("report")
def cmd_injuries_report(
    file: Annotated[
        str | None,
        typer.Option("--file", "-f", help="Read a saved copy of the page instead of fetching it."),
    ] = None,
    show: Annotated[
        bool, typer.Option("--unmatched", help="List rows with no rostered player.")
    ] = False,
) -> None:
    """Read the daily injury table (INJURY_REPORT_URL) and record what changed.

    Only changes are written: a new status, scope or comment, or a player who
    has dropped off the table (recorded as back). Running it twice in a row
    writes nothing the second time.
    """
    from pathlib import Path

    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.news.injury_report import (
        ReportFormatError,
        fetch_injury_report,
        import_injury_report,
    )

    settings = get_settings()
    url = settings.injury_report_url
    if not url and not file:
        typer.secho(
            "  INJURY_REPORT_URL is not set (see .env.example), and no --file given.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    async def job() -> object:
        page = (
            Path(file).read_text(encoding="utf-8")
            if file
            else await fetch_injury_report(url or "", settings=settings)
        )
        async with session_scope() as session:
            imported = await import_injury_report(
                session,
                page,
                url=url or "https://basketnews.com/injury-report",
                now=SystemClock().now(),
            )
            await session.commit()
            return imported

    try:
        result = _run(job)
    except ReportFormatError as exc:
        typer.secho(f"  {exc}. Nothing was written.", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    r: Any = result
    typer.echo(
        f"  {r.rows} rows, {r.matched} matched to rostered players, {len(r.unmatched)} not rostered"
    )
    typer.echo(f"  {r.written} changed, {r.cleared} back (off the table), {r.unchanged} unchanged")
    for change in r.changes:
        scope = {
            "game": f"game {change['game_code']}",
            "until": f"until {change['until']}",
            "open": "until back",
        }[change["scope"]]
        rules_out = change["status"] in ("OUT", "SUSPENDED")
        typer.secho(
            f"    {change['team']:<5}{change['player']:<28}{change['status']:<14}{scope:<18}"
            f"{(change['reason'] or '')[:70]}",
            fg=typer.colors.RED if rules_out else None,
        )
    if show and r.unmatched:
        typer.echo("  not on a current roster (usually players outside the EuroLeague squad):")
        for name in r.unmatched:
            typer.echo(f"    {name}")


@injuries_app.command("check")
def cmd_injuries_check(
    round_number: Annotated[
        int | None, typer.Option("--round", "-r", help="Check one round of the season.")
    ] = None,
    days: Annotated[
        int, typer.Option("--days", help="Otherwise, finished games in the last N days.")
    ] = 7,
    season: SeasonOption = "",
    show_all: Annotated[
        bool, typer.Option("--all", help="List every player, not only mismatches.")
    ] = False,
) -> None:
    """After a round: did the statuses the predictions used hold up?

    Every player with a status in a finished game, grouped by what the
    prediction used, against the box score. A ruled-out player who played, or a
    ready player who did not, is a mismatch. Run after `ml score` (box scores
    must be ingested).
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.ml.injury_check import check_games, finished_games

    async def job() -> object:
        async with session_scope() as session:
            games = await finished_games(
                session,
                now=SystemClock().now(),
                days=days,
                round_number=round_number,
                season_code=(season or _season_default()) if round_number is not None else None,
            )
            return await check_games(session, games)

    result: Any = _run(job)
    scope = f"round {round_number}" if round_number is not None else f"the last {days} day(s)"
    if not result.players:
        typer.echo(f"  no predicted players with a status in finished games from {scope}")
        return
    typer.echo("")
    typer.secho(f"Injury statuses vs box scores - {result.games} game(s), {scope}", bold=True)
    typer.echo(
        f"  {'status used':<26}{'players':>8}{'played':>8}{'play rate':>11}{'mean P(plays)':>15}"
        f"{'mismatches':>12}"
    )
    for group, row in result.summary().items():
        bad = row["mismatches"]
        typer.secho(
            f"  {group:<26}{row['players']:>8}{row['played']:>8}{row['play_rate']:>10.0%} "
            f"{row['mean_p_plays']:>14.2f}{bad if group in ('ruled out', 'ready') else '-':>12}",
            fg=typer.colors.RED if bad else None,
        )
    listed = result.players if show_all else result.mismatches
    if listed:
        typer.echo("")
        typer.echo("  every player:" if show_all else "  mismatches:")
        for p in listed:
            played = f"{p.minutes:.0f} min" if p.played else ("DNP" if p.dressed else "not dressed")
            typer.secho(
                "    game "
                f"{p.game_code:<5}{p.team:<5}{p.player:<28}{p.group:<26}{p.source.lower():<8}"
                f"{played:<12}{(p.reason or '')[:60]}",
                fg=typer.colors.RED if p.mismatch else None,
            )
    elif not show_all:
        typer.secho(
            "  no mismatches: every ruled-out player sat, every ready player played",
            fg=typer.colors.GREEN,
        )


@injuries_app.command("list")
def cmd_injuries_list() -> None:
    """Reports that apply to each team's next game right now."""
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.features.injury_status import resolve_status
    from euroleague_ai.ingestion.injuries import current_rosters, next_game
    from euroleague_ai.repositories.availability import last_appearances, status_reports

    async def job() -> object:
        now = SystemClock().now()
        async with session_scope() as session:
            roster = await current_rosters(session, as_of=now)
            by_id = {p.player_id: p for p in roster}
            reports = await status_reports(session, list(by_id), as_of=now)
            played = await last_appearances(session, list(reports), as_of=now)
            rows = []
            games: dict[Any, Any] = {}
            for player_id, items in reports.items():
                member = by_id[player_id]
                if member.team_id not in games:
                    games[member.team_id] = await next_game(session, member.team_id, after=now)
                game = games[member.team_id]
                if game is None:
                    continue
                resolved = resolve_status(
                    items,
                    game_id=game.id,
                    tip_off=game.tip_off_utc,
                    last_played=played.get(player_id),
                )
                if resolved is not None:
                    rows.append(
                        (member.team_code, member.full_name, game.game_code, resolved.as_dict())
                    )
            return sorted(rows, key=lambda r: (r[0], r[1]))

    rows = _run(job)
    if not rows:
        typer.echo(
            "  no reports apply to upcoming games. Add one with `euroleague-ai injuries set`."
        )
        return
    typer.echo("")
    typer.echo(
        f"  {'team':<5}{'player':<28}{'status':<14}{'next game':<10}{'scope':<12}"
        f"{'source':<12}{'applied':<9}reason"
    )
    for team_code, name, code, info in rows:  # type: ignore[attr-defined]
        scope = {"game": "this game", "until": f"to {info['until']}", "open": "until back"}[
            info["scope"]
        ]
        source = info["source"].lower()
        if info.get("confidence") is not None:
            source += f" {float(info['confidence']):.2f}"
        applied = "P=0" if info.get("applied") else "-"
        typer.echo(
            f"  {team_code:<5}{name:<28}{info['status']:<14}{code!s:<10}{scope:<12}"
            f"{source:<12}{applied:<9}{info['reason'] or ''}"
        )


@injuries_app.command("history")
def cmd_injuries_history(
    player: Annotated[str, typer.Argument(help="Name or player code.")],
    team: Annotated[str | None, typer.Option("--team", "-t")] = None,
) -> None:
    """Every report recorded for one player, newest first."""
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.ingestion.injuries import InjuryInputError, current_rosters, match_player
    from euroleague_ai.repositories.availability import status_reports

    async def job() -> tuple[RosterPlayer, list[StatusReport]]:
        now = SystemClock().now()
        async with session_scope() as session:
            match = match_player(await current_rosters(session, as_of=now), player, team_code=team)
            reports = (await status_reports(session, [match.player_id], as_of=now)).get(
                match.player_id, []
            )
            return match, reports

    try:
        match, reports = _run(job)
    except InjuryInputError as exc:
        typer.secho(f"  {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"  {match.full_name} ({match.team_code})", bold=True)
    if not reports:
        typer.echo("  no reports recorded")
    for r in reports:
        scope = (
            "one game"
            if r.game_id
            else (f"until {r.expected_return.date()}" if r.expected_return else "until back")
        )
        typer.echo(
            f"  {r.observed_at:%Y-%m-%d %H:%M}  {r.status!s:<13} {scope:<18} {r.source!s:<8} "
            f"{r.reason or ''}"
        )


@app.command("version")
def cmd_version() -> None:
    """Print the application version."""
    from euroleague_ai import __version__

    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()


@quality_app.command("check")
def cmd_quality_check() -> None:
    """Run integrity checks over the ingested data.

    Exits non-zero if any ERROR-level finding is present, so this can gate a
    training pipeline rather than merely informing a human.
    """
    from euroleague_ai.quality.checks import Severity, run_quality_checks

    async def job() -> QualityReport:
        async with session_scope() as session:
            return await run_quality_checks(session)

    report = _run(job)

    colours = {
        Severity.ERROR: typer.colors.RED,
        Severity.WARNING: typer.colors.YELLOW,
        Severity.INFO: typer.colors.CYAN,
    }
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}

    typer.echo("")
    for finding in sorted(report.findings, key=lambda f: order[f.severity]):
        tag = typer.style(f"{finding.severity:<7}", fg=colours[finding.severity], bold=True)
        suffix = f"  [{finding.count:,}]" if finding.count else ""
        typer.echo(f"{tag} {finding.check}{suffix}")
        typer.echo(f"        {finding.message}")
        for line in finding.sample:
            typer.echo(f"          - {line}")
        typer.echo("")

    errors, warnings = len(report.errors), len(report.warnings)
    summary = f"{errors} error(s), {warnings} warning(s)"
    if report.ok:
        typer.secho(f"PASS - {summary}", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho(f"FAIL - {summary}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


def _print_spread_detail(coverage: dict[str, Any]) -> None:
    """Per-player spread: what was chosen, coverage per volatility third, Brier."""
    scaling = coverage.get("spread_scaling") or {}
    feature = scaling.get("volatility_feature")
    typer.echo(
        f"    spread         scaled by {feature} ^ {scaling.get('scale_power')}"
        if feature
        else "    spread         unscaled (no volatility scale beat it on validation)"
    )
    groups = coverage.get("by_volatility") or {}
    reference = coverage.get("reference_by_volatility") or {}
    if groups:
        parts = []
        for name in ("low", "mid", "high"):
            if name not in groups:
                continue
            part = f"{name} {float(groups[name]['covered_80']):.1%}"
            if name in reference:
                part += f" (was {float(reference[name]['covered_80']):.1%})"
            parts.append(part)
        typer.echo("    80% by swing   " + ", ".join(parts))
    brier = coverage.get("threshold_brier")
    if brier is not None:
        was = coverage.get("reference_threshold_brier")
        typer.echo(
            f"    P(X+) Brier    {float(brier):.4f}"
            + (f" (unscaled {float(was):.4f})" if was is not None else "")
        )


def _print_pipeline(result: dict[str, Any]) -> None:
    """Render one pipeline run."""
    if result.get("skipped"):
        typer.secho(
            "  Another run holds the lock; this slot was skipped.",
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo("")
    for step in result.get("steps") or []:
        if step["ok"]:
            typer.secho(f"  OK    {step['name']}", fg=typer.colors.GREEN)
            for key, value in (step.get("detail") or {}).items():
                if isinstance(value, dict):
                    continue
                typer.echo(f"          {key:<22}{value}")
        else:
            typer.secho(f"  FAIL  {step['name']}: {step['error']}", fg=typer.colors.RED)

    if result.get("ok"):
        typer.secho("\n  Run completed.", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho(
            "\n  Run completed with failures - recorded in ingestion_runs.",
            fg=typer.colors.YELLOW,
            bold=True,
        )


@pipeline_app.command("run")
def cmd_pipeline_run(
    season: SeasonOption = "",
    days: Annotated[
        int, typer.Option("--days", "-d", help="Project fixtures within this many days.")
    ] = 0,
    skip_ingest: Annotated[
        bool,
        typer.Option("--skip-ingest", help="Score and predict from data already held."),
    ] = False,
) -> None:
    """Run ingest, then score, then predict, once.

    Ingest first so box scores exist to score against; score before predicting
    so a run that dies partway through has still recorded how yesterday went --
    the one part that cannot be recovered from upstream later.

    Safe to re-run: every step is idempotent, and a Postgres advisory lock stops
    two runs overlapping.
    """
    from euroleague_ai.scheduling.runner import run_daily

    settings = get_settings()

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            result = await run_daily(
                session,
                season_code=season or None,
                within_days=days or settings.pipeline_within_days,
                skip_ingest=skip_ingest,
            )
            return result.as_dict()

    result = _run(job)
    _print_pipeline(result)
    if not result.get("ok") and not result.get("skipped"):
        raise typer.Exit(code=1)


@pipeline_app.command("serve")
def cmd_pipeline_serve(
    season: SeasonOption = "",
    at: Annotated[
        str,
        typer.Option("--at", help='UTC run times, e.g. "06:00,16:00". Default: config.'),
    ] = "",
    once: Annotated[
        bool,
        typer.Option("--once", help="Wait for the next slot, run, then exit."),
    ] = False,
) -> None:
    """Run the pipeline on a schedule, forever.

    A loop and a sleep, not a task queue. The work is one job twice a day
    against one database; a broker, a result backend and a worker pool would add
    failure modes rather than remove them.

    Intended as a long-running container. Restart-safe: it holds no state
    between runs, and every run records itself in ingestion_runs.
    """
    import asyncio

    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.scheduling.runner import run_daily
    from euroleague_ai.scheduling.schedule import (
        next_run_at,
        parse_run_times,
        seconds_until,
    )

    settings = get_settings()
    try:
        run_times = parse_run_times(at or settings.pipeline_run_at)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    clock = SystemClock()
    typer.secho(
        "  Scheduler started. Run times (UTC): "
        + ", ".join(t.strftime("%H:%M") for t in run_times),
        fg=typer.colors.GREEN,
        bold=True,
    )

    async def loop() -> None:
        while True:
            target = next_run_at(clock.now(), run_times)
            delay = seconds_until(clock.now(), target)
            typer.echo(f"  next run {target:%Y-%m-%d %H:%M} UTC (in {delay / 60:.0f} min)")
            await asyncio.sleep(delay)

            async with session_scope() as session:
                result = await run_daily(
                    session,
                    season_code=season or None,
                    within_days=settings.pipeline_within_days,
                )
            _print_pipeline(result.as_dict())

            if once:
                return
            # A run finishing inside its own minute would otherwise re-enter the
            # same slot; next_run_at is strictly-after, so a second is enough.
            await asyncio.sleep(1)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        typer.echo("\n  Scheduler stopped.")


@pipeline_app.command("history")
def cmd_pipeline_history(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """Recent scheduled runs and how they went.

    The point of an unattended job is that nobody watches it, which is exactly
    why it needs a trail that can be read afterwards.
    """
    from euroleague_ai.scheduling.runner import recent_runs

    async def job() -> list[dict[str, Any]]:
        async with session_scope() as session:
            runs = await recent_runs(session, limit=limit)
            return [
                {
                    "started": r.started_at.isoformat(timespec="minutes"),
                    "status": str(r.status),
                    "season": r.season_code,
                    "error": r.error_message,
                    "steps": (r.details or {}).get("steps") or [],
                }
                for r in runs
            ]

    runs = _run(job)
    if not runs:
        typer.echo("no scheduled runs yet - try `euroleague-ai pipeline run`")
        return

    typer.echo("")
    typer.echo(f"  {'started (UTC)':<20}{'status':<10}{'steps':<28}season")
    for run in runs:
        marks = " ".join(f"{s['name']}{'+' if s['ok'] else '!'}" for s in run["steps"])
        typer.secho(
            f"  {run['started']:<20}{run['status']:<10}{marks:<28}{run['season'] or ''}",
            fg=typer.colors.RED if run["status"] == "FAILED" else None,
        )
        if run["error"]:
            typer.secho(f"      {run['error']}", fg=typer.colors.RED)


@ml_app.command("baselines")
def cmd_ml_baselines(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Single target, e.g. POINTS. Default: all five."),
    ] = None,
    start: Annotated[int | None, typer.Option("--from", help="First season start year.")] = None,
    end: Annotated[int | None, typer.Option("--to", help="Last season start year.")] = None,
    min_train_rounds: Annotated[
        int,
        typer.Option(
            "--min-train-rounds",
            help="Rounds that must precede a fold. Early rounds have empty form windows.",
        ),
    ] = 5,
) -> None:
    """Backtest the baselines. This is the bar Phase 3 must clear.

    Walk-forward by season and round -- train on rounds 1..k, evaluate k+1.
    Never a random split: that would put a player's adjacent games on both sides
    and measure interpolation rather than forecasting.
    """
    from euroleague_ai.ml.report import DEFAULT_TARGETS, run_baseline_report
    from euroleague_ai.models.enums import PredictionTarget

    if target:
        try:
            targets: tuple[PredictionTarget, ...] = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in PredictionTarget)
            raise typer.BadParameter(f"unknown target {target!r}. Valid: {valid}") from exc
        if targets[0] is PredictionTarget.PLAYS:
            raise typer.BadParameter(
                "PLAYS is a classifier with its own baselines; they are reported by "
                "`euroleague-ai ml train --target PLAYS`."
            )
    else:
        targets = DEFAULT_TARGETS

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            return await run_baseline_report(
                session,
                targets=targets,
                season_codes=codes,
                min_train_rounds=min_train_rounds,
            )

    report = _run(job)

    typer.echo("")
    typer.secho(
        f"Baseline backtest - feature set {report['feature_set_version']}",
        bold=True,
    )
    for name, block in report["targets"].items():
        typer.echo("")
        if "error" in block:
            typer.secho(f"{name}: {block['error']}", fg=typer.colors.RED)
            continue

        data = block["dataset"]
        typer.secho(f"  {name}", fg=typer.colors.CYAN, bold=True)
        typer.echo(
            f"    {data['rows']:,} rows | {data['players']:,} players | "
            f"{data['games']:,} games | mean {data['target_mean']}"
        )
        typer.echo(f"    {'predictor':<22}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'bias':>9}{'cover':>8}")
        for rank, entry in enumerate(block["ranking"]):
            style = typer.colors.GREEN if rank == 0 else None
            line = (
                f"    {entry['predictor']:<22}{entry['mae']:>9.3f}{entry['rmse']:>9.3f}"
                f"{entry['r2']:>9.3f}{entry['bias']:>9.3f}{entry['coverage']:>8.2f}"
            )
            typer.secho(line, fg=style, bold=rank == 0)

        # Stratified view of the winner. An aggregate MAE can look healthy while
        # the model is useless on the 10-25 minute rotation players, which is
        # exactly where the interesting decisions are.
        winner = block["ranking"][0]["predictor"] if block["ranking"] else None
        buckets = block["detail"].get(winner, {}).get("by_minutes") if winner else None
        if buckets:
            typer.echo("")
            typer.echo(f"      {winner} by realised minutes:")
            typer.echo(f"        {'minutes':<10}{'n':>8}{'MAE':>9}{'bias':>9}")
            for label, stats in buckets.items():
                if stats["n"]:
                    typer.echo(
                        f"        {label:<10}{stats['n']:>8,}{stats['mae']:>9.3f}"
                        f"{stats['bias']:>9.3f}"
                    )

        best = block["best"]
        bar = block["bar_to_beat"]
        typer.echo("")
        if bar is None or bar != bar:  # NaN
            # Every baseline covering nothing means the features are empty, not
            # that the baselines are bad. Saying so beats printing NaN.
            typer.secho(
                "    NO BASELINE COULD PREDICT ANY ROW.\n"
                "    The snapshots exist but their features are null, which means the\n"
                "    history queries found nothing at each cutoff. Almost always this is\n"
                "    observed_at sitting in the future relative to the games it describes\n"
                "    (the classic backfill mistake). Diagnose and fix with:\n"
                "      euroleague-ai maintenance repair-observed-at --dry-run",
                fg=typer.colors.RED,
                bold=True,
            )
        else:
            typer.secho(f"    bar to beat: {best} at MAE {bar:.3f}", fg=typer.colors.YELLOW)

    typer.echo("")
    typer.secho(
        "Phase 3 gate: a model must beat the best baseline above, per target, "
        "on walk-forward validation.",
        bold=True,
    )


@ml_app.command("describe")
def cmd_ml_describe(
    target: Annotated[str, typer.Option("--target", "-t")] = "POINTS",
) -> None:
    """Summarise the training dataset for one target."""
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.dataset import describe_rows, load_training_rows
    from euroleague_ai.models.enums import PredictionTarget

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            rows = await load_training_rows(
                session,
                target=PredictionTarget[target.upper()],
                feature_set_version=FEATURE_SET_VERSION,
            )
            return describe_rows(rows)

    summary = _run(job)
    typer.echo("")
    for key, value in summary.items():
        typer.echo(f"  {key:<16} {value}")


@maintenance_app.command("repair-observed-at")
def cmd_repair_observed_at(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Count affected rows without changing them.")
    ] = False,
) -> None:
    """Rewrite observed_at to the time each fact became knowable.

    Backfilled history was stamped with the ingest wall-clock, placing every
    historical fact in the future relative to its own game and hiding it from
    every feature query. This rewrites those timestamps from data already in the
    database -- no API calls.

    Only moves timestamps backwards: rows ingested live keep their real values.
    """
    from euroleague_ai.ingestion.repair import repair_observed_at

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            return (await repair_observed_at(session, dry_run=dry_run)).as_dict()

    result = _run(job)
    typer.echo("")
    for key, value in result.items():
        typer.echo(f"  {key:<20} {value:,}")
    if dry_run:
        typer.secho("\n  dry run - nothing was modified", fg=typer.colors.YELLOW)
    else:
        typer.secho(
            "\n  Snapshots built before this repair are stale. Purge and rebuild:\n"
            "    euroleague-ai maintenance purge-snapshots\n"
            "    euroleague-ai features backfill --from 2016 --to 2025",
            fg=typer.colors.YELLOW,
        )


@maintenance_app.command("purge-snapshots")
def cmd_purge_snapshots(
    version: Annotated[
        str | None,
        typer.Option("--version", help="Only this feature set version. Default: all."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete feature snapshots so they can be rebuilt.

    Snapshots are immutable by design, so correcting the data underneath them
    does not update them -- they must be discarded and recomputed.
    """
    from euroleague_ai.ingestion.repair import purge_feature_snapshots

    if not yes:
        typer.confirm(f"Delete feature snapshots ({version or 'all versions'})?", abort=True)

    async def job() -> object:
        async with session_scope() as session:
            return await purge_feature_snapshots(session, feature_set_version=version)

    typer.echo(f"deleted {_run(job):,} snapshots")


@ml_app.command("tune")
def cmd_ml_tune(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Single target. Default: all five."),
    ] = None,
    trials: Annotated[int, typer.Option("--trials", "-n", help="Configurations to sample.")] = 40,
    start: Annotated[int | None, typer.Option("--from", help="First season start year.")] = None,
    end: Annotated[int | None, typer.Option("--to", help="Last season start year.")] = None,
    save: Annotated[
        bool,
        typer.Option("--save", help="Write the winner so `ml train` picks it up."),
    ] = False,
) -> None:
    """Search for better LightGBM settings.

    The current values were chosen when the feature set was smaller and have
    never been re-tuned, which is the likeliest reason recent feature additions
    made some targets slightly worse.

    Every trial is scored on the validation split and the winner is chosen
    there; only then is it retrained once and measured on test against the
    current defaults. Selecting on test would make the reported gain the maximum
    of N noisy draws rather than an estimate -- so the number printed here is
    one you can believe.

    Nothing is saved unless --save is passed.
    """
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.pipeline import MIN_ROWS
    from euroleague_ai.ml.report import DEFAULT_TARGETS
    from euroleague_ai.ml.train import train_target
    from euroleague_ai.ml.tuning import save_params, search
    from euroleague_ai.models.enums import PredictionTarget

    if target:
        try:
            targets: tuple[PredictionTarget, ...] = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in DEFAULT_TARGETS)
            raise typer.BadParameter(f"unknown target {target!r}. Valid: {valid}") from exc
    else:
        targets = DEFAULT_TARGETS

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> list[dict[str, Any]]:
        out = []
        async with session_scope() as session:
            for item in targets:
                rows = await load_training_rows(
                    session,
                    target=item,
                    feature_set_version=FEATURE_SET_VERSION,
                    season_codes=codes,
                )
                if len(rows) < MIN_ROWS:
                    out.append({"target": str(item), "error": f"only {len(rows)} rows"})
                    continue

                found = search(rows, item, trials=trials)

                # The honest confirmation: retrain both on the same split and
                # compare on test, which the search never touched.
                # MAE comparison only; the expectation head would double the cost.
                tuned = train_target(rows, item, params=found.best.params, fit_expected=False)
                default = train_target(rows, item, fit_expected=False)
                record = found.as_dict()
                record["test_mae_tuned"] = round(tuned.metrics.mae, 4)
                record["test_mae_default"] = round(default.metrics.mae, 4)
                record["test_gain_pct"] = round(
                    (default.metrics.mae - tuned.metrics.mae) / default.metrics.mae * 100.0,
                    2,
                )
                if save and record["test_gain_pct"] > 0:
                    record["saved_to"] = str(save_params(item, found.best.params))
                out.append(record)
        return out

    results = _run(job)
    typer.echo("")
    typer.secho(f"  Hyperparameter search - {trials} trials per target", bold=True)
    typer.echo(f"  {'target':<22}{'default':>9}{'tuned':>9}{'test gain':>11}   validation gain")

    improved = 0
    for row in results:
        if "error" in row:
            typer.secho(f"  {row['target']:<22}{row['error']}", fg=typer.colors.YELLOW)
            continue
        better = row["test_gain_pct"] > 0
        improved += better
        typer.secho(
            f"  {row['target']:<22}{row['test_mae_default']:>9.4f}"
            f"{row['test_mae_tuned']:>9.4f}{row['test_gain_pct']:>+10.2f}%"
            f"{row['validation_gain_pct']:>+15.2f}%",
            fg=typer.colors.GREEN if better else None,
        )
        if row.get("saved_to"):
            typer.secho(f"    {'':<20}saved -> {row['saved_to']}", fg=typer.colors.GREEN)

    typer.echo("")
    scored = [r for r in results if "error" not in r]
    if not scored:
        return
    if improved:
        typer.secho(
            f"  {improved} of {len(scored)} improved on test."
            + ("" if save else " Re-run with --save to keep them."),
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        typer.secho(
            "  No target improved on test. The defaults are already reasonable;\n"
            "  a validation gain that does not survive to test was selection noise.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    typer.echo("")
    for row in scored:
        typer.echo(f"  {row['target']} best params: {row['best_params']}")


@ml_app.command("chain")
def cmd_ml_chain(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Single target. Default: all chainable."),
    ] = None,
    start: Annotated[int | None, typer.Option("--from", help="First season start year.")] = None,
    end: Annotated[int | None, typer.Option("--to", help="Last season start year.")] = None,
) -> None:
    """Test whether predicting minutes then rates beats predicting the stat.

    Points are minutes times scoring rate. Asking one model for points makes it
    learn both at once; splitting them lets each learn the thing it is good at.
    The counter-argument is that errors multiply -- minutes 10% high and rate
    10% high gives points 21% high.

    Which effect wins is measurable, not arguable, so this trains both on
    identical rows and prints the answer. Nothing is registered or activated:
    read the numbers first.
    """
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.chain import CHAINABLE, compare_chain
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.pipeline import MIN_ROWS
    from euroleague_ai.models.enums import PredictionTarget

    if target:
        # Lookup by member name, matching the other commands. A StrEnum accepts
        # either, but two conventions in one file is how a typo becomes a
        # confusing error rather than an obvious one.
        try:
            targets: tuple[PredictionTarget, ...] = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in CHAINABLE)
            raise typer.BadParameter(f"unknown target {target!r}. Chainable: {valid}") from exc
        if targets[0] not in CHAINABLE:
            raise typer.BadParameter(
                f"{targets[0]} is not chainable -- minutes is the multiplier, "
                "so it cannot be expressed as a rate of itself."
            )
    else:
        targets = CHAINABLE

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> list[dict[str, Any]]:
        results = []
        async with session_scope() as session:
            for item in targets:
                rows = await load_training_rows(
                    session,
                    target=item,
                    feature_set_version=FEATURE_SET_VERSION,
                    season_codes=codes,
                )
                if len(rows) < MIN_ROWS:
                    results.append({"target": str(item), "error": f"only {len(rows)} rows"})
                    continue
                results.append(compare_chain(rows, item).as_dict())
        return results

    results = _run(job)
    typer.echo("")
    typer.secho("  Chained (minutes x rate) vs direct", bold=True)
    typer.echo(f"  {'target':<22}{'direct':>9}{'chained':>9}{'change':>9}{'baseline':>10}  verdict")

    wins = 0
    for row in results:
        if "error" in row:
            typer.secho(f"  {row['target']:<22}{row['error']}", fg=typer.colors.YELLOW)
            continue
        better = row["chain_wins"]
        wins += better
        typer.secho(
            f"  {row['target']:<22}{row['direct_mae']:>9.4f}{row['chained_mae']:>9.4f}"
            f"{row['delta_pct']:>+8.2f}%{row['baseline_mae']:>10.4f}  "
            f"{'CHAIN' if better else 'direct'}",
            fg=typer.colors.GREEN if better else None,
        )
        typer.echo(
            f"    {'':<20}minutes MAE {row['minutes_mae']:.3f} · "
            f"rate MAE {row['rate_mae']:.3f} per 40"
        )
        for note in row.get("notes") or []:
            typer.secho(f"    {'':<20}{note}", fg=typer.colors.YELLOW)

    typer.echo("")
    scored = [r for r in results if "error" not in r]
    if wins and wins == len(scored):
        typer.secho(
            "  The chain wins on every target. Worth wiring into serving.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    elif wins:
        typer.secho(
            f"  The chain wins on {wins} of {len(scored)}. Mixed - chain only where it wins.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    else:
        typer.secho(
            "  Direct wins everywhere. Error compounding beat the stability gain;\n"
            "  the honest move is to leave the models as they are.",
            fg=typer.colors.YELLOW,
            bold=True,
        )


def _print_expectation(expectation: dict[str, Any] | None) -> None:
    """Render how a target's expected value was chosen and whether it adds up."""
    if not expectation:
        return
    chosen = expectation["chosen"]
    test = expectation["test"]
    totals = expectation["team_totals"]
    published = expectation["source"] != "none"
    typer.echo(
        f"    expected     {chosen.replace('_', ' ')}: RMSE {test[chosen]['rmse']:.3f} "
        f"vs median {test['median']['rmse']:.3f}, bias {test[chosen]['bias']:+.3f} "
        f"vs {test['median']['bias']:+.3f}"
    )
    typer.secho(
        f"    squad totals bias per team-game: median {totals['median']['bias']:+.2f}"
        f" -> expected {totals[chosen]['bias']:+.2f}   "
        f"{'published' if published else 'FAIL - no expected values published'}",
        fg=typer.colors.GREEN if published else typer.colors.RED,
    )
    if not expectation.get("beats_baseline", True):
        typer.secho(
            f"      note: RMSE does not beat the best baseline "
            f"({expectation['baseline']} {expectation['baseline_rmse']:.3f})",
            fg=typer.colors.YELLOW,
        )


def _print_availability_block(block: dict[str, Any]) -> bool:
    """Render the PLAYS training report. Returns whether it passed the gate."""
    s = block["summary"]
    model, base = s["model"], s["baseline_metrics"]
    data = block.get("dataset", {})
    ok = bool(s["beats_baseline"])

    def fmt(value: object, spec: str = ".4f") -> str:
        return "n/a" if value is None else format(value, spec)

    typer.secho("  PLAYS  (probability of taking the floor)", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"    rows         {data.get('rows', 0):,} candidates over {data.get('games', 0):,} games, "
        f"{fmt(data.get('base_rate'), '.1%')} played"
    )
    typer.echo(
        f"                 {data.get('dressed_not_used', 0):,} dressed but unused, "
        f"{data.get('not_in_squad', 0):,} not in the squad"
    )
    typer.echo(
        f"    model        log loss {fmt(model['log_loss'])}   Brier {fmt(model['brier'])}   "
        f"AUC {fmt(model['auc'], '.3f')}   ECE {fmt(model['ece'], '.3f')}"
    )
    typer.echo(
        f"    baseline     log loss {fmt(base['log_loss'])}   Brier {fmt(base['brier'])}   "
        f"({s['baseline']})"
    )
    typer.secho(
        f"    improvement  {fmt(s['improvement_pct'], '+.2f')}%   "
        f"{'PASS' if ok else 'FAIL - not activated'}",
        fg=typer.colors.GREEN if ok else typer.colors.RED,
        bold=True,
    )
    typer.echo(
        f"    players used per team-game: off by {fmt(model['team_count_mae'], '.2f')} "
        f"on average (bias {fmt(model['team_count_bias'], '+.2f')})"
    )
    by_outcome = block.get("by_outcome") or {}
    if by_outcome:
        typer.echo(
            "    mean P(plays) by what happened: "
            + ", ".join(
                f"{key.replace('_', ' ')} {fmt(value.get('mean_predicted'), '.2f')}"
                for key, value in by_outcome.items()
            )
        )
    recall = data.get("candidate_recall")
    if recall is not None and recall < 0.98:
        typer.secho(
            f"    candidate list caught {recall:.1%} of players who played; the rest are "
            "never projected at all",
            fg=typer.colors.YELLOW,
        )
    if data.get("missing_snapshots"):
        typer.secho(
            f"    {data['missing_snapshots']:,} candidates had no snapshot and were skipped. "
            "Run `euroleague-ai features backfill` to add them.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("    top features: " + ", ".join(n for n, _ in block["top_features"][:5]))
    typer.echo(f"    artifact:     {block['model']['artifact']}")
    return ok


@ml_app.command("spread")
def cmd_ml_spread(
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Single stat target. Default: all five.")
    ] = None,
) -> None:
    """Are projections as spread out as real box scores? Test period, active models.

    Calibration slope of outcomes on expected values (1 is right; above 1 means
    projections are too bunched), outcomes by decile of projection, and the
    15+/20+ nights the probabilities implied against those that happened.
    """
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.inference import load_active_models
    from euroleague_ai.ml.report import DEFAULT_TARGETS
    from euroleague_ai.ml.spread import target_spread
    from euroleague_ai.models.enums import PredictionTarget

    targets: tuple[PredictionTarget, ...] = DEFAULT_TARGETS
    if target:
        try:
            targets = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            raise typer.BadParameter(f"unknown target {target!r}") from exc

    async def job() -> object:
        async with session_scope() as session:
            models = await load_active_models(session, targets)
            return [
                await target_spread(session, t, models[t], feature_set_version=FEATURE_SET_VERSION)
                for t in targets
                if t in models
            ]

    reports: Any = _run(job)
    for r in reports:
        typer.echo("")
        colour = typer.colors.GREEN if r.verdict == "about right" else typer.colors.YELLOW
        typer.secho(
            f"  {r.target} - {r.rows:,} test rows, expected values from "
            f"{r.expected_source or 'median'}",
            bold=True,
        )
        typer.secho(
            f"    calibration slope {r.slope:.3f} (intercept {r.intercept:+.2f}): {r.verdict}"
            f"    median slope {r.median_slope:.3f}",
            fg=colour,
        )
        q = r.quantiles
        typer.echo(
            f"    {'':<10}"
            + "".join(f"{'p' + str(int(float(k) * 100)):>8}" for k in q["actual"])
            + f"{'std':>9}"
        )
        for name in ("median", "expected", "actual"):
            typer.echo(
                f"    {name:<10}"
                + "".join(f"{v:>8.2f}" for v in q[name].values())
                + f"{r.std[name]:>9.3f}"
            )
        typer.echo("    by decile of expected value (projected -> actual):")
        typer.echo(
            "     " + "  ".join(f"{d['projected']:.1f}->{d['actual']:.1f}" for d in r.deciles)
        )
        if r.big_nights:
            typer.echo(
                "    big nights, implied by P(X+) vs happened: "
                + ", ".join(
                    f"{b['bar']}+ {b['implied']:,.0f} vs {b['realised']:,}"
                    + (f" ({b['ratio']:.2f}x)" if b["ratio"] is not None else "")
                    for b in r.big_nights
                )
            )


@ml_app.command("ablate")
def cmd_ml_ablate(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="One target, e.g. POINTS or PLAYS. Default: all six."),
    ] = None,
) -> None:
    """Which v6 feature families help? Retrain without each and compare.

    Needs the v6 backfill (`features backfill`). For each target: the full v6
    model, a v5-only model, and one model per v6 family removed, on the same
    rows and split. A family helps a target when removing it makes test MAE
    (log loss for P(plays)) worse by more than twice the retraining noise,
    measured by training the full model three more times with other seeds.
    Nothing is registered. Slow: nine trainings per target.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.ablation import ablate, save_report, verdict
    from euroleague_ai.ml.availability import load_availability_rows
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.pipeline import MIN_ROWS
    from euroleague_ai.ml.report import SERVED_TARGETS
    from euroleague_ai.ml.tuning import load_params
    from euroleague_ai.models.enums import PredictionTarget

    targets: tuple[PredictionTarget, ...] = SERVED_TARGETS
    if target:
        try:
            targets = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            raise typer.BadParameter(f"unknown target {target!r}") from exc

    async def job() -> object:
        reports = []
        async with session_scope() as session:
            for t in targets:
                rows: list[Any]
                if t is PredictionTarget.PLAYS:
                    rows, _ = await load_availability_rows(
                        session, feature_set_version=FEATURE_SET_VERSION
                    )
                    params = None
                else:
                    rows = await load_training_rows(
                        session, target=t, feature_set_version=FEATURE_SET_VERSION
                    )
                    params = load_params(t)
                if len(rows) < MIN_ROWS:
                    typer.secho(
                        f"  {t.name}: only {len(rows):,} {FEATURE_SET_VERSION} rows. Run "
                        "`euroleague-ai features backfill` first.",
                        fg=typer.colors.RED,
                    )
                    continue
                typer.echo(f"  {t.name}: {len(rows):,} rows, 9 trainings...")
                reports.append(ablate(t, rows, params=params))
        path = save_report(reports, now=SystemClock().now()) if reports else None
        return reports, path

    out: Any = _run(job)
    reports, path = out
    for report in reports:
        typer.echo("")
        metric = "log loss" if report["metric"] == "log_loss" else "test MAE"
        typer.secho(
            f"  {report['target']} - {metric}, change vs all v6 "
            f"(retraining noise {report['noise_pct']:.2f}%, helps above "
            f"{report['helps_above_pct']:.2f}%)",
            bold=True,
        )
        for row in report["variants"]:
            mark = "  helps" if row["helps"] else ""
            typer.secho(
                f"    {row['variant']:<30}{row['score']:>10.4f}{row['change_pct']:>+9.2f}%{mark}",
                fg=typer.colors.GREEN if row["helps"] else None,
            )
    if reports:
        typer.echo("")
        typer.secho("  Families", bold=True)
        for family, helped in verdict(reports).items():
            typer.secho(
                f"    {family:<20}"
                + (
                    f"helps {', '.join(helped)}"
                    if helped
                    else "helps nothing: consider dropping it"
                ),
                fg=typer.colors.GREEN if helped else typer.colors.YELLOW,
            )
        typer.echo(f"  saved: {path}")


@ml_app.command("walkforward")
def cmd_ml_walkforward(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Single stat target. Default: all five."),
    ] = None,
    seasons: Annotated[
        int, typer.Option("--seasons", "-n", help="How many recent seasons to test.")
    ] = 4,
    expected: Annotated[
        bool,
        typer.Option(
            "--expected/--no-expected", help="Also fit the expected-value head (twice as slow)."
        ),
    ] = False,
) -> None:
    """Train on every season before N, test on season N, for recent seasons.

    One model per season per target, trained exactly like `ml train` and judged
    against the best baseline on that season's games. Nothing is registered or
    served. The report is saved, so a later change can be compared season by
    season against this one. Slow: one full training per fold.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.report import DEFAULT_TARGETS
    from euroleague_ai.ml.tuning import load_params
    from euroleague_ai.ml.walkforward import evaluate_target, save_report
    from euroleague_ai.models.enums import PredictionTarget

    if target:
        try:
            targets: tuple[PredictionTarget, ...] = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            raise typer.BadParameter(f"unknown target {target!r}") from exc
        if targets[0] not in DEFAULT_TARGETS:
            raise typer.BadParameter(
                f"{target} is not a stat target; walk-forward covers {DEFAULT_TARGETS}"
            )
    else:
        targets = DEFAULT_TARGETS
    current = _season_default()

    async def job() -> object:
        results = []
        async with session_scope() as session:
            for t in targets:
                rows = await load_training_rows(
                    session, target=t, feature_set_version=FEATURE_SET_VERSION
                )
                typer.echo(f"  {t.name}: {len(rows):,} rows, training {seasons} fold(s)...")
                results.append(
                    evaluate_target(
                        rows,
                        t,
                        seasons=seasons,
                        current_season=current,
                        fit_expected=expected,
                        params=load_params(t),
                    )
                )
        path = save_report(
            results, feature_set_version=FEATURE_SET_VERSION, now=SystemClock().now()
        )
        return results, path

    out: Any = _run(job)
    results, path = out
    for result in results:
        typer.echo("")
        summary = result.summary()
        typer.secho(
            f"  {result.target.name}: beats the baseline in "
            f"{summary['beats_baseline']}/{summary['folds']} "
            f"seasons, median {summary['median_improvement_pct']:+.2f}% "
            f"(worst {summary['worst_improvement_pct']:+.2f}%, best "
            f"{summary['best_improvement_pct']:+.2f}%)"
            if summary["folds"]
            else f"  {result.target.name}: no season had enough rows",
            bold=True,
            fg=typer.colors.GREEN
            if summary["folds"] and summary["beats_baseline"] == summary["folds"]
            else typer.colors.YELLOW,
        )
        header = (
            "    "
            f"{'season':<9}{'test rows':>10}{'model MAE':>11}{'baseline':>10}"
            f"{'gain':>9}{'bias':>8}{'80% cov':>9}"
        )
        if expected:
            header += f"{'RMSE med':>10}{'RMSE exp':>10}"
        typer.echo(header + "  baseline used")
        for fold in result.folds:
            line = (
                f"    {fold.season + ('*' if fold.partial else ''):<9}{fold.test_rows:>10,}"
                f"{fold.model_mae:>11.4f}{fold.baseline_mae:>10.4f}{fold.improvement * 100:>+8.2f}%"
                f"{fold.bias:>+8.3f}"
                + (f"{fold.covered_80:>9.1%}" if fold.covered_80 is not None else f"{'-':>9}")
            )
            if expected:
                nan = float("nan")
                median_rmse = fold.median_rmse if fold.median_rmse is not None else nan
                expected_rmse = fold.expected_rmse if fold.expected_rmse is not None else nan
                line += f"{median_rmse:>10.3f}{expected_rmse:>10.3f}"
            typer.secho(
                line + f"  {fold.baseline}", fg=None if fold.beats_baseline else typer.colors.RED
            )
    typer.echo("")
    typer.echo("  * season still being played")
    typer.echo(f"  saved: {path}")


def _print_season_gate(gate: Any) -> None:
    """One line per recent season, then the verdict."""
    if gate is None:
        typer.echo("    per season   skipped (--no-walkforward): single-split gate only")
        return
    for fold in gate.get("seasons") or []:
        mark = "*" if fold.get("partial") else " "
        typer.secho(
            f"      {fold['season']}{mark}  {fold['improvement_pct']:+6.2f}%  "
            f"(MAE {fold['model_mae']:.4f} vs {fold['baseline_mae']:.4f}, "
            f"{fold['test_rows']:,} rows)",
            fg=None if fold.get("beats_baseline") else typer.colors.RED,
        )
    if not gate.get("judged"):
        typer.secho(f"    per season   not judged: {gate['reason']}", fg=typer.colors.YELLOW)
        return
    ok = bool(gate.get("passes"))
    typer.secho(
        f"    per season   {gate['reason']}   {'PASS' if ok else 'FAIL'}",
        fg=typer.colors.GREEN if ok else typer.colors.RED,
        bold=True,
    )


@ml_app.command("train")
def cmd_ml_train(
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            "-t",
            help="Single target, e.g. POINTS or PLAYS. Default: all five stat targets, then PLAYS.",
        ),
    ] = None,
    start: Annotated[int | None, typer.Option("--from", help="First season start year.")] = None,
    end: Annotated[int | None, typer.Option("--to", help="Last season start year.")] = None,
    activate: Annotated[
        bool | None,
        typer.Option(
            "--activate/--no-activate",
            help="Override the gate. Default: activate only if it beats the baseline.",
        ),
    ] = None,
    walkforward: Annotated[
        bool,
        typer.Option(
            "--walkforward/--no-walkforward",
            help="Also require the gain to hold season by season (slow). Default on.",
        ),
    ] = True,
) -> None:
    """Train LightGBM models and register them.

    Each model is scored on a chronological held-out test split and compared to
    the best baseline on exactly those rows. A model that does not beat its
    baseline is registered but never activated -- a model worse than arithmetic
    is a liability, not an improvement.

    PLAYS, the probability that a player takes the floor at all, is judged on
    log loss against appearance-rate baselines rather than on MAE.
    """
    from euroleague_ai.ml.pipeline import train_and_register
    from euroleague_ai.ml.report import SERVED_TARGETS
    from euroleague_ai.models.enums import PredictionTarget

    if target:
        try:
            targets: tuple[PredictionTarget, ...] = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in PredictionTarget)
            raise typer.BadParameter(f"unknown target {target!r}. Valid: {valid}") from exc
    else:
        targets = SERVED_TARGETS

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            return await train_and_register(
                session,
                targets=targets,
                season_codes=codes,
                activate=activate,
                walkforward=walkforward,
                current_season=_season_default(),
            )

    report = _run(job)

    typer.echo("")
    typer.secho(f"Training - feature set {report['feature_set_version']}", bold=True)
    passed = failed = 0

    for name, block in report["targets"].items():
        typer.echo("")
        if "error" in block:
            typer.secho(f"  {name}: {block['error']}", fg=typer.colors.RED)
            failed += 1
            continue

        if name == str(PredictionTarget.PLAYS):
            ok = _print_availability_block(block)
            passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
            continue

        s = block["summary"]
        ok = s["beats_baseline"]
        active = bool((block.get("model") or {}).get("active"))
        passed, failed = (passed + 1, failed) if active else (passed, failed + 1)

        typer.secho(f"  {name}", fg=typer.colors.CYAN, bold=True)
        typer.echo(
            f"    model MAE    {s['model_mae']:.4f}   (RMSE {s['rmse']:.3f}, "
            f"R2 {s['r2']:.3f}, bias {s['bias']:+.3f})"
        )
        typer.echo(f"    baseline     {s['baseline_mae']:.4f}   ({s['baseline']})")
        typer.secho(
            f"    improvement  {s['improvement_pct']:+.2f}%   {'PASS' if ok else 'FAIL'}",
            fg=typer.colors.GREEN if ok else typer.colors.RED,
            bold=True,
        )
        _print_season_gate(block.get("season_gate"))
        typer.secho(
            f"    {'ACTIVATED' if active else 'NOT ACTIVATED'}",
            fg=typer.colors.GREEN if active else typer.colors.RED,
            bold=True,
        )

        if s["improvement_pct"] > 25:
            typer.secho(
                "    WARNING: improvement this large is more likely a leak than a\n"
                "    breakthrough. Published systems gain 5-15%. Investigate before\n"
                "    trusting it.",
                fg=typer.colors.YELLOW,
                bold=True,
            )

        coverage = (block.get("summary") or {}).get("coverage")
        if coverage:
            # Coverage is to an interval what MAE is to a point estimate. An 80%
            # interval that holds 62% of outcomes is not an interval.
            covered = float(coverage.get("covered_80", 0.0))
            drift = covered - 0.80
            typer.secho(
                f"    80% interval  {covered:.1%} covered  "
                f"({drift:+.1%} vs nominal, "
                f"{float(coverage.get('below_interval', 0.0)):.1%} below / "
                f"{float(coverage.get('above_interval', 0.0)):.1%} above)",
                fg=typer.colors.GREEN if abs(drift) <= 0.05 else typer.colors.YELLOW,
            )
            _print_spread_detail(coverage)

        occupancy = (block.get("summary") or {}).get("distribution_rows") or {}
        if occupancy:
            debuts = occupancy.get("history_0", 0)
            typer.secho(
                f"    distribution   {occupancy.get('total', 0):,} residuals, "
                f"{debuts:,} from players with no prior games",
                fg=typer.colors.YELLOW if debuts < 50 else None,
            )
            if debuts < 50:
                # Below the marginal threshold the unknown-player bucket falls
                # back to the pooled sample, and confidence stops discriminating.
                typer.secho(
                    "      too few to characterise unknown players; their "
                    "confidence will match everyone else's",
                    fg=typer.colors.YELLOW,
                )

        _print_expectation(s.get("expectation"))

        typer.echo("    top features: " + ", ".join(n for n, _ in block["top_features"][:5]))
        typer.echo(f"    artifact:     {block['model']['artifact']}")

    if report["targets"]:
        typer.echo("")
        typer.secho(
            "Models changed: run `euroleague-ai ml reconcile` so squad minutes and "
            "stat lines are scaled for the new versions.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.secho(
        f"{passed} target(s) passed the gate, {failed} failed.",
        fg=typer.colors.GREEN if failed == 0 else typer.colors.YELLOW,
        bold=True,
    )


@ml_app.command("reconcile")
def cmd_ml_reconcile(
    save: Annotated[
        bool,
        typer.Option("--save/--dry-run", help="Write the fit so predictions use it."),
    ] = True,
) -> None:
    """Fit how hard to pull squad minutes and stat lines toward what real squads produce.

    Minutes: scaled toward what real squads play. Points, assists, rebounds and
    threes: scaled toward a team-level anchor (expected team points, pace,
    rebounding rates) fitted on past games. Uses the active models on games none
    of them has seen. Each strength is the strongest whose validation player
    error is no worse than unscaled, and is judged on the test period: it
    switches on only if squad totals get better there and player error is not
    significantly worse (paired, clustered by squad). Run it again after every
    `ml train`.
    """
    from euroleague_ai.core.clock import SystemClock
    from euroleague_ai.ml.inference import load_active_models
    from euroleague_ai.ml.reconcile import fit_minutes_reconciliation, reconciliation_path
    from euroleague_ai.ml.report import SERVED_TARGETS
    from euroleague_ai.ml.squad_stats import fit_stats_reconciliation, stats_reconciliation_path

    async def job() -> object:
        async with session_scope() as session:
            models = await load_active_models(session, SERVED_TARGETS)
            now = SystemClock().now()
            fit = await fit_minutes_reconciliation(session, models=models, save=save, now=now)
            try:
                stats: object = (
                    await fit_stats_reconciliation(session, models=models, save=save, now=now)
                ).to_dict()
            except ValueError as exc:
                stats = str(exc)
            return fit.to_dict(), stats

    try:
        both: Any = _run(job)
    except ValueError as exc:
        typer.secho(f"  {exc}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1) from exc

    report, stats_report = both
    r: dict[str, Any] = report
    typer.echo("")
    typer.secho(
        f"Squad minutes reconciliation - {r['minutes_model']} + {r['plays_model']}", bold=True
    )
    typer.echo(
        f"  target: squads' candidates log {r['target_minutes']:.1f} min per team-game "
        f"(validation from {r['periods']['fit_start'][:10]}, test from "
        f"{r['periods']['test_start'][:10]})"
    )
    typer.echo("")
    typer.echo(
        f"  {'alpha':>5}  {'player MAE':>10}  {'player bias':>11}  {'squad error':>11}  "
        f"{'squad bias':>10}"
    )
    for row in r["validation"]:
        marker = "  <- chosen" if row["alpha"] == r["alpha"] else ""
        typer.echo(
            f"  {row['alpha']:>5.1f}  {row['player_mae']:>10.3f}  {row['player_bias']:>+11.3f}  "
            f"{row['squad_mae']:>11.2f}  {row['squad_bias']:>+10.2f}{marker}"
        )
    base, scaled = r["test"]["unscaled"], r["test"]["scaled"]
    typer.echo("")
    typer.echo("  test period (unseen by both models):")
    typer.echo(
        f"    unscaled      player MAE {base['player_mae']:.3f}   squad error "
        f"{base['squad_mae']:.2f} "
        f"(bias {base['squad_bias']:+.2f})"
    )
    typer.echo(
        f"    alpha {r['alpha']:.1f}     player MAE {scaled['player_mae']:.3f}   squad error "
        f"{scaled['squad_mae']:.2f} (bias {scaled['squad_bias']:+.2f})"
    )
    delta = r["test"].get("player_delta") or {}
    if delta.get("se") is not None:
        worse = r["test"].get("players_significantly_worse")
        typer.echo(
            f"    player MAE change {delta['mean']:+.4f} min (standard error {delta['se']:.4f}, "
            f"{delta['squads']:,} squads): "
            + ("significantly worse" if worse else "not significantly worse")
        )
    ok = bool(r["enabled"])
    typer.secho(
        f"  {'ON - predictions will use it' if ok else 'OFF - did not pass, predictions unchanged'}"
        + ("" if save else "  (dry run, nothing saved)"),
        fg=typer.colors.GREEN if ok else typer.colors.YELLOW,
        bold=True,
    )
    if save:
        typer.echo(f"  saved: {reconciliation_path()}")

    _print_stats_reconciliation(
        stats_report, saved_to=stats_reconciliation_path() if save else None
    )
    if save:
        typer.echo("")
        typer.echo("  next:  euroleague-ai ml predict --days 7")


def _print_stats_reconciliation(report: Any, *, saved_to: Any) -> None:
    typer.echo("")
    if isinstance(report, str):
        typer.secho(
            f"Squad stat reconciliation - not fitted: {report}", fg=typer.colors.YELLOW, bold=True
        )
        return
    models = report["models"]
    typer.secho("Squad stat reconciliation - toward each side's expected team line", bold=True)
    typer.echo(
        "  models: "
        + ", ".join(
            models[name] for name in ("POINTS", "ASSISTS", "REBOUNDS", "THREE_POINTERS_MADE")
        )
        + f" + {models['PLAYS']}"
    )
    typer.echo(
        f"  validation from {report['periods']['fit_start'][:10]}, "
        f"test from {report['periods']['test_start'][:10]} (unseen by every model)"
    )
    typer.echo("")
    typer.echo(
        f"  {'stat':<9}{'alpha':>6}  {'squad error':>17}  {'squad bias':>17}  "
        f"{'player RMSE':>17}  {'change in sq. err (SE)':>23}  result"
    )
    labels = {
        "POINTS": "points",
        "ASSISTS": "assists",
        "REBOUNDS": "rebounds",
        "THREE_POINTERS_MADE": "threes",
    }
    for name, fit in report["targets"].items():
        base, new, delta = (
            fit["test"]["unscaled"],
            fit["test"]["scaled"],
            fit["test"]["player_delta"],
        )
        ok = bool(fit["enabled"])
        typer.secho(
            f"  {labels.get(name, name):<9}{fit['alpha']:>6.1f}  "
            f"{base['squad_mae']:>7.2f} -> {new['squad_mae']:<7.2f}  "
            f"{base['squad_bias']:>+7.2f} -> {new['squad_bias']:<+7.2f}  "
            f"{base['player_rmse']:>7.3f} -> {new['player_rmse']:<7.3f}  "
            f"{delta['mean']:>+11.4f} ({delta['se']:.4f})  "
            + ("ON" if ok else "OFF" + (f": {fit['note']}" if fit.get("note") else "")),
            fg=typer.colors.GREEN if ok else typer.colors.YELLOW,
        )
    for name, why in (report.get("skipped") or {}).items():
        typer.secho(f"  {labels.get(name, name):<9}skipped: {why}", fg=typer.colors.YELLOW)
    if saved_to is not None:
        typer.echo(f"  saved: {saved_to}")


@ml_app.command("models")
def cmd_ml_models() -> None:
    """List registered models, newest first."""
    from euroleague_ai.ml.registry import list_models

    async def job() -> list[dict[str, Any]]:
        async with session_scope() as session:
            rows = await list_models(session)
            return [
                {
                    "name": m.name,
                    "version": m.version,
                    "target": str(m.target),
                    "active": m.is_active,
                    "features": m.feature_set_version,
                    "rows": ((m.metrics or {}).get("rows") or {}).get("train"),
                    "mae": (m.metrics or {}).get("model_mae"),
                    "baseline_mae": (m.metrics or {}).get("baseline_mae"),
                    "improvement": (m.metrics or {}).get("improvement_pct"),
                    "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                }
                for m in rows
            ]

    models = _run(job)
    if not models:
        typer.echo("no models registered yet - run `euroleague-ai ml train`")
        return

    typer.echo("")
    typer.echo(f"  {'name':<27}{'ver':<8}{'feat':<6}{'MAE':>9}{'base':>9}{'impr%':>8}  active")
    for m in models:
        mae = m["mae"] if m["mae"] is not None else float("nan")
        base = m["baseline_mae"] if m["baseline_mae"] is not None else float("nan")
        impr = m["improvement"] if m["improvement"] is not None else float("nan")
        typer.secho(
            f"  {m['name']:<27}{m['version']:<8}{m['features']:<6}"
            f"{mae:>9.4f}{base:>9.4f}{impr:>8.2f}"
            f"  {'yes' if m['active'] else 'no'}",
            fg=typer.colors.GREEN if m["active"] else None,
        )


@ml_app.command("predict")
def cmd_ml_predict(
    season: SeasonOption = "",
    days: Annotated[
        int, typer.Option("--days", "-d", help="Predict fixtures within this many days.")
    ] = 7,
    game_code: Annotated[
        list[int] | None,
        typer.Option("--game-code", "-g", help="Specific games. Repeatable."),
    ] = None,
) -> None:
    """Predict upcoming fixtures with the active models.

    Features are built at the same cutoff training used
    (tip-off minus PREDICTION_LEAD_MINUTES), so the model is served exactly the
    information set it learned on.
    """
    from euroleague_ai.ml.inference import predict_upcoming

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            return await predict_upcoming(
                session,
                season_code=season or None,
                within_days=days,
                game_codes=list(game_code) if game_code else None,
            )

    result = _run(job)
    typer.echo("")
    if "error" in result:
        typer.secho(f"  {result['error']}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

    typer.secho(
        f"  {result['predictions']:,} predictions across {result['games']} game(s)",
        fg=typer.colors.GREEN,
        bold=True,
    )
    for target, model in (result.get("models") or {}).items():
        typer.echo(f"    {target:<24} {model}")
    if "minutes_reconciliation" in result:
        typer.echo(f"    {'squad minutes':<24} {result['minutes_reconciliation']}")
    if "stats_reconciliation" in result:
        typer.echo(f"    {'squad stats':<24} {result['stats_reconciliation']}")
    if result["games"] == 0:
        typer.secho(
            "  No scheduled games in the window. Ingest a season with upcoming\n"
            "  fixtures first:  euroleague-ai ingest all --season E2026",
            fg=typer.colors.YELLOW,
        )


@ml_app.command("show")
def cmd_ml_show(
    game_code: Annotated[int, typer.Option("--game-code", "-g")],
    season: SeasonOption = "",
    median: Annotated[
        bool,
        typer.Option(
            "--median",
            help="Show medians (what he usually does) instead of expected values "
            "(what he averages).",
        ),
    ] = False,
) -> None:
    """Print the projected lines for one game.

    Stat columns are expected values given he plays, where the model publishes
    them, otherwise medians. ``play`` is the chance he takes the floor.
    """
    from sqlalchemy import select

    from euroleague_ai.models.competition import Season
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.people import Player
    from euroleague_ai.models.prediction import ModelVersion, Prediction

    async def job() -> object:
        async with session_scope() as session:
            season_code = season or _season_default()
            game = (
                await session.execute(
                    select(Game)
                    .join(Season, Season.id == Game.season_id)
                    .where(Game.game_code == game_code, Season.code == season_code)
                )
            ).scalar_one_or_none()
            if game is None:
                raise typer.BadParameter(f"game {game_code} not found in season {season_code}")

            # Only the current view. Predictions are append-only by design --
            # a projection made before a transfer is a legitimate historical
            # record and must not be deleted -- so the filtering belongs here,
            # at the point of display, not in the writer.
            #
            # Without this, `show` unioned every model version ever run against
            # the game, and a player who has since been reconciled off the
            # roster kept appearing with the numbers some superseded model gave
            # him. Two such ghosts (a transferred guard and a released big) sat
            # in this output looking exactly like a live projection.
            rows = (
                await session.execute(
                    select(
                        Player.full_name,
                        Prediction.target,
                        Prediction.predicted_value,
                        Prediction.quantiles,
                        Prediction.threshold_probabilities,
                        Prediction.confidence,
                        Prediction.expected_value,
                        Prediction.explanation,
                    )
                    .join(Player, Player.id == Prediction.player_id)
                    .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
                    .where(
                        Prediction.game_id == game.id,
                        ModelVersion.is_active.is_(True),
                    )
                )
            ).all()

            by_player: dict[str, dict[str, Any]] = {}
            for (
                full_name,
                target,
                value,
                quantiles,
                thresholds,
                confidence,
                expected,
                explanation,
            ) in rows:
                entry = by_player.setdefault(full_name, {})
                availability = (explanation or {}).get("availability")
                if availability and "status" not in entry:
                    entry["status"] = availability.get("status")
                entry[str(target)] = float(value)
                if expected is not None:
                    entry[f"{target!s}_e"] = float(expected)
                # str() explicitly: PredictionTarget is a StrEnum today, but
                # relying on f-string formatting of an enum is the kind of thing
                # that silently changes key names if the base class ever does.
                entry[f"{target!s}_q"] = quantiles
                entry[f"{target!s}_p"] = thresholds
                # One confidence per player, not per target. Minutes drives
                # every other line, so its spread is the honest summary.
                if confidence is not None and (
                    str(target) == "MINUTES" or "confidence" not in entry
                ):
                    entry["confidence"] = float(confidence)
            return {"tip_off": game.tip_off_utc.isoformat(), "players": by_player}

    data = _run(job)
    players = data["players"]  # type: ignore[index]
    if not players:
        typer.echo("no predictions for this game - run `euroleague-ai ml predict`")
        return

    order = ["MINUTES", "POINTS", "REBOUNDS", "ASSISTS", "THREE_POINTERS_MADE"]
    typer.echo("")
    typer.secho(f"  tip-off {data['tip_off']}", bold=True)  # type: ignore[index]
    use_expected = not median and any(f"{t}_e" in v for v in players.values() for t in order)
    typer.echo(
        "  stat columns: "
        + ("expected if he plays (--median for medians)" if use_expected else "medians if he plays")
    )
    typer.echo(
        f"  {'player':<28}{'play':>6}{'MIN':>7}{'PTS':>7}{'REB':>7}{'AST':>7}{'3PM':>7}"
        f"{'conf':>7}   pts 10/15/20/25"
    )
    for name, values in sorted(players.items(), key=lambda kv: -kv[1].get("MINUTES", 0.0)):

        def cell(target: str, values: dict[str, Any] = values) -> float:
            if use_expected and f"{target}_e" in values:
                return float(values[f"{target}_e"])
            return float(values.get(target, float("nan")))

        cells = "".join(f"{cell(t):>7.1f}" for t in order)
        # P(plays) first: every number to its right is conditional on it.
        plays = values.get("PLAYS")
        play_cell = "     -" if plays is None else f"{plays:>6.0%}"

        confidence = values.get("confidence")
        conf_cell = "      -" if confidence is None else f"{confidence:>7.2f}"

        # Threshold probabilities are the point of the whole exercise: "will he
        # go for twenty" is the question, not "what is his mean".
        probabilities = values.get("POINTS_p") or {}
        odds = (
            "  ".join(
                f"{round(float(probabilities[key]) * 100):>2}%"
                for key in ("10", "15", "20", "25")
                if key in probabilities
            )
            or "-"
        )
        status = values.get("status")
        marker = f"  [{status}]" if status else ""
        typer.echo(f"  {name:<28}{play_cell}{cells}{conf_cell}   {odds}{marker}")

    if not any(v.get("POINTS_q") for v in players.values()):
        typer.secho(
            "\n  No distributions stored. These predictions came from a model\n"
            "  trained before feature set v3 + item 06. Retrain and re-predict:\n"
            "    euroleague-ai ml train && euroleague-ai ml predict --days 60",
            fg=typer.colors.YELLOW,
        )


@ml_app.command("score")
def cmd_ml_score(
    season: SeasonOption = "",
    game_code: Annotated[
        list[int] | None,
        typer.Option("--game-code", "-g", help="Specific games. Repeatable."),
    ] = None,
    rescore: Annotated[
        bool,
        typer.Option("--rescore", help="Recompute outcomes that already exist."),
    ] = False,
) -> None:
    """Join realised box scores onto published predictions.

    Idempotent, so it is safe to run on a schedule: predictions that already
    carry an outcome are skipped unless --rescore is passed.

    Run it after ingesting a round's stats. Until this runs, the system has no
    idea whether anything it published was any good.
    """
    from euroleague_ai.ml.outcomes import score_predictions

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            result = await score_predictions(
                session,
                season_code=season or None,
                game_codes=list(game_code) if game_code else None,
                rescore=rescore,
            )
            return result.as_dict()

    result = _run(job)
    typer.echo("")
    typer.secho(
        f"  {result['predictions_scored']:,} scored across {result['games']} finished game(s)",
        fg=typer.colors.GREEN,
        bold=True,
    )
    if result["already_scored"]:
        typer.echo(f"    {result['already_scored']:,} already had outcomes")
    if result["did_not_appear"]:
        typer.secho(
            f"    {result['did_not_appear']:,} for players who never took the floor "
            "(scored zero, excluded from MAE by default)",
            fg=typer.colors.YELLOW,
        )
    for target, count in (result.get("targets") or {}).items():
        typer.echo(f"    {target:<24}{count:>8,}")


@ml_app.command("accuracy")
def cmd_ml_accuracy(
    season: SeasonOption = "",
    include_absent: Annotated[
        bool,
        typer.Option(
            "--include-absent",
            help="Fold non-appearances into the error figures.",
        ),
    ] = False,
) -> None:
    """Realised accuracy and calibration on games that have been played.

    This is the only measurement that is not computed from the same pipeline as
    the model, which is what makes it worth more than the training report.
    """
    from euroleague_ai.ml.outcomes import accuracy_report, unscored_games

    async def job() -> dict[str, Any]:
        async with session_scope() as session:
            report = await accuracy_report(
                session,
                season_code=season or None,
                include_absent=include_absent,
            )
            report["unscored_games"] = await unscored_games(session, season_code=season or None)
            return report

    report = _run(job)
    targets = report.get("targets") or {}
    if not targets:
        typer.secho(
            "  Nothing scored yet. Ingest a played round, then:\n    euroleague-ai ml score",
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo("")
    typer.secho(f"  Realised accuracy - season {report['season']}", bold=True)
    typer.echo(f"  {'target':<24}{'n':>7}{'MAE':>9}{'bias':>9}{'80% int':>10}")
    for target, block in targets.items():
        interval = block.get("interval") or {}
        covered = interval.get("covered_80")
        cell = "        -" if covered is None else f"{covered:>9.1%}"
        typer.echo(
            f"  {target:<24}{block['scored']:>7,}{block['mae']:>9.4f}{block['bias']:>+9.3f}{cell}"
        )

    for key, title in (
        ("thresholds", "Threshold calibration - P(X+) if he plays"),
        (
            "thresholds_overall",
            "Threshold calibration - overall P(X+), sitting out counts as a miss",
        ),
    ):
        bands = report.get(key) or []
        if not bands:
            continue
        typer.echo("")
        typer.secho(f"  {title}", bold=True)
        typer.echo(f"  {'stated':<12}{'n':>8}{'realised':>11}{'drift':>9}")
        for band in bands:
            realised = band["realised"]
            drift = realised - band["expected"]
            typer.secho(
                f"  {band['band']:<12}{band['predictions']:>8,}{realised:>11.1%}{drift:>+9.1%}",
                fg=typer.colors.YELLOW if abs(drift) > 0.1 else None,
            )

    outstanding = report.get("unscored_games") or 0
    if outstanding:
        typer.secho(
            f"\n  {outstanding} finished game(s) still unscored - run `euroleague-ai ml score`",
            fg=typer.colors.YELLOW,
        )


@ml_app.command("why")
def cmd_ml_why(
    game_code: Annotated[int, typer.Option("--game-code", "-g")],
    player: Annotated[str, typer.Option("--player", "-p", help="Name, or any part of one.")],
    season: SeasonOption = "",
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Also print the largest per-feature SHAP values."),
    ] = False,
) -> None:
    """Explain one player's projected line.

    Prints the signed drivers behind each target, from exact TreeSHAP
    contributions grouped into themes. These explain the *model* -- what moved
    this prediction, given these inputs -- not basketball.
    """
    from sqlalchemy import select

    from euroleague_ai.models.competition import Season
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.people import Player
    from euroleague_ai.models.prediction import ModelVersion, Prediction

    async def job() -> list[dict[str, Any]]:
        async with session_scope() as session:
            season_code = season or _season_default()
            game = (
                await session.execute(
                    select(Game)
                    .join(Season, Season.id == Game.season_id)
                    .where(Game.game_code == game_code, Season.code == season_code)
                )
            ).scalar_one_or_none()
            if game is None:
                raise typer.BadParameter(f"game {game_code} not found in season {season_code}")

            rows = (
                await session.execute(
                    select(
                        Player.full_name,
                        Prediction.target,
                        Prediction.predicted_value,
                        Prediction.explanation,
                    )
                    .join(Player, Player.id == Prediction.player_id)
                    .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
                    .where(
                        Prediction.game_id == game.id,
                        ModelVersion.is_active.is_(True),
                        Player.full_name.ilike(f"%{player}%"),
                    )
                )
            ).all()
            return [
                {
                    "player": name,
                    "target": str(target),
                    "value": float(value),
                    "explanation": explanation,
                }
                for name, target, value, explanation in rows
            ]

    results = _run(job)
    if not results:
        typer.echo(f"no predictions matching '{player}' in game {game_code}")
        return

    order = {
        t: i
        for i, t in enumerate(["MINUTES", "POINTS", "REBOUNDS", "ASSISTS", "THREE_POINTERS_MADE"])
    }
    results = sorted(
        results,
        key=lambda r: (r["player"], order.get(r["target"], 99)),
    )

    current = None
    for row in results:
        if row["player"] != current:
            current = row["player"]
            typer.echo("")
            typer.secho(f"  {current}", bold=True)

        typer.secho(f"    {row['target']:<22}{row['value']:>6.1f}", fg=typer.colors.CYAN)

        explanation = row["explanation"]
        if not explanation:
            typer.echo("      (no explanation stored - predict again with a current model)")
            continue

        for driver in explanation.get("drivers", []):
            colour = typer.colors.GREEN if driver["direction"] == "+" else typer.colors.RED
            typer.secho(
                f"      {driver['direction']} {driver['label']:<38}{driver['impact']:>+7.2f}",
                fg=colour,
            )

        if raw:
            for name, value in explanation.get("top_features", [])[:6]:
                typer.echo(f"        {name:<40}{value:>+8.3f}")


@ml_app.command("roster")
def cmd_ml_roster(
    game_code: Annotated[int, typer.Option("--game-code", "-g")],
    season: SeasonOption = "",
) -> None:
    """Show the roster inference resolves for a game, and why.

    Use this before trusting projections after a transfer window. It prints the
    club each player is resolved to at the prediction cutoff, together with the
    registration date that decided it -- which is what makes a wrong club
    obvious rather than mysterious.
    """
    from sqlalchemy import select

    from euroleague_ai.features.snapshot import default_prediction_timestamp
    from euroleague_ai.models.competition import Season, Team
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.people import Player, PlayerTeamStint
    from euroleague_ai.repositories.rosters import candidate_players, current_club

    async def job() -> object:
        settings = get_settings()
        async with session_scope() as session:
            season_code = season or _season_default()
            game = (
                await session.execute(
                    select(Game)
                    .join(Season, Season.id == Game.season_id)
                    .where(Game.game_code == game_code, Season.code == season_code)
                )
            ).scalar_one_or_none()
            if game is None:
                raise typer.BadParameter(f"game {game_code} not found in season {season_code}")

            cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
            club_by_player = await current_club(session, as_of=cutoff)

            sides = []
            for team_id, label in (
                (game.home_team_id, "HOME"),
                (game.away_team_id, "AWAY"),
            ):
                team = (await session.execute(select(Team).where(Team.id == team_id))).scalar_one()
                players = await candidate_players(
                    session,
                    team_id=team_id,
                    season_id=game.season_id,
                    as_of=cutoff,
                    club_by_player=club_by_player,
                )
                detail = []
                for player_id in players:
                    row = (
                        await session.execute(
                            select(Player.full_name, PlayerTeamStint.effective_from)
                            .join(
                                PlayerTeamStint,
                                PlayerTeamStint.player_id == Player.id,
                            )
                            .where(
                                Player.id == player_id,
                                PlayerTeamStint.team_id == team_id,
                                PlayerTeamStint.effective_from <= cutoff,
                            )
                            .order_by(PlayerTeamStint.effective_from.desc())
                            .limit(1)
                        )
                    ).first()
                    if row:
                        detail.append((row[0], row[1].date().isoformat()))
                sides.append((label, team.name, detail))

            return {"cutoff": cutoff.isoformat(), "sides": sides}

    data = _run(job)
    typer.echo("")
    typer.secho(f"  cutoff {data['cutoff']}", bold=True)  # type: ignore[index]
    for label, team_name, players in data["sides"]:  # type: ignore[index]
        typer.echo("")
        typer.secho(
            f"  {label}  {team_name}  ({len(players)} players)", fg=typer.colors.CYAN, bold=True
        )
        for name, registered in players:
            typer.echo(f"    {name:<32} registered {registered}")
        if not players:
            typer.secho(
                "    none resolved - roster data for this season may be missing",
                fg=typer.colors.YELLOW,
            )

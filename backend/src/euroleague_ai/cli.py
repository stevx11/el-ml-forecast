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
from collections.abc import Awaitable, Callable
from typing import Annotated, TypeVar

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

logger = get_logger(__name__)

app = typer.Typer(help="EuroLeague AI operations", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingestion jobs", no_args_is_help=True)
features_app = typer.Typer(help="Feature computation", no_args_is_help=True)
news_app = typer.Typer(help="News collection", no_args_is_help=True)
quality_app = typer.Typer(help="Data quality", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(news_app, name="news")
app.add_typer(quality_app, name="quality")

T = TypeVar("T")

SeasonOption = Annotated[
    str, typer.Option("--season", "-s", help="Season code, e.g. E2025.")
]


def _season_default() -> str:
    return get_settings().euroleague_default_season


def _run(factory: Callable[[], Awaitable[T]]) -> T:
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
    wait: Annotated[
        bool, typer.Option("--wait", help="Sit through rate-limit cooldowns.")
    ] = True,
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
                        result = await ingest_all(
                            session, provider, season_ref(code, settings)
                        )
                    summary[code] = f"{result.rows_written} rows"
                except Exception as exc:  # noqa: BLE001
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
            help="Skip rounds below this. Early-season form windows are empty, so "
            "those rows teach the model about missingness rather than basketball.",
        ),
    ] = 4,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Stop after N games. Use for a smoke run first."),
    ] = None,
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
            )

    _report("snapshots", _run(job))


@features_app.command("names")
def cmd_feature_names() -> None:
    """Print the feature vector's column names and count."""
    from euroleague_ai.features.registry import FEATURE_SET_VERSION, feature_names

    async def job() -> object:
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
    from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats  # noqa: F401

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
    """Poll every configured news feed.

    No feeds ship configured. Set ``NEWS_FEED_URLS`` to sources you are
    permitted to poll; see the legal note in docs/architecture.md.
    """
    from euroleague_ai.news.rss import RssNewsCollector

    async def job() -> object:
        settings = get_settings()
        if not settings.news_feed_urls:
            typer.echo("No feeds configured. Set NEWS_FEED_URLS in your environment.")
            return {"articles": 0}

        collector = RssNewsCollector(settings)
        try:
            total = 0
            for url in settings.news_feed_urls:
                articles = await collector.collect(url)
                typer.echo(f"  {url}: {len(articles)} articles")
                total += len(articles)
            return {"articles": total}
        finally:
            await collector.aclose()

    _report("news", _run(job))


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

    async def job() -> object:
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

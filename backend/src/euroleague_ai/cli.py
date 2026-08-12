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
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(news_app, name="news")

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
) -> None:
    """Fetch and store box scores for finished games.

    Idempotent -- safe to re-run after an upstream stat correction.
    """

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings)
        try:
            async with session_scope() as session:
                return await ingest_game_stats(
                    session,
                    provider,
                    season_ref(season or _season_default(), settings),
                    game_codes=list(game_code) if game_code else None,
                )
        finally:
            await provider.aclose()

    _report("stats", _run(job))


@ingest_app.command("all")
def cmd_ingest_all(season: SeasonOption = "") -> None:
    """Run every ingestion job in dependency order.

    This is the "make it work" command: teams, then players, then games, then
    box scores.
    """

    async def job() -> object:
        settings = get_settings()
        provider = build_provider(settings)
        try:
            async with session_scope() as session:
                return await ingest_all(
                    session, provider, season_ref(season or _season_default(), settings)
                )
        finally:
            await provider.aclose()

    _report("all", _run(job))


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

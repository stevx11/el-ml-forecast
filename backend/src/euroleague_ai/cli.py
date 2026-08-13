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
ml_app = typer.Typer(help="Datasets, baselines, evaluation", no_args_is_help=True)
maintenance_app = typer.Typer(help="Data repair", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(news_app, name="news")
app.add_typer(quality_app, name="quality")
app.add_typer(ml_app, name="ml")
app.add_typer(maintenance_app, name="maintenance")

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
    from euroleague_ai.models.enums import PredictionTarget
    from euroleague_ai.ml.report import DEFAULT_TARGETS, run_baseline_report

    if target:
        try:
            targets = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in PredictionTarget)
            raise typer.BadParameter(f"unknown target {target!r}. Valid: {valid}") from exc
    else:
        targets = DEFAULT_TARGETS

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> object:
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
            typer.secho(
                f"    bar to beat: {best} at MAE {bar:.3f}", fg=typer.colors.YELLOW
            )

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
    from euroleague_ai.models.enums import PredictionTarget
    from euroleague_ai.ml.dataset import describe_rows, load_training_rows

    async def job() -> object:
        async with session_scope() as session:
            rows = await load_training_rows(
                session,
                target=PredictionTarget[target.upper()],
                feature_set_version=FEATURE_SET_VERSION,
            )
            return describe_rows(rows)

    summary = _run(job)
    typer.echo("")
    for key, value in summary.items():  # type: ignore[union-attr]
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

    async def job() -> object:
        async with session_scope() as session:
            return (await repair_observed_at(session, dry_run=dry_run)).as_dict()

    result = _run(job)
    typer.echo("")
    for key, value in result.items():  # type: ignore[union-attr]
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
        typer.confirm(
            f"Delete feature snapshots ({version or 'all versions'})?", abort=True
        )

    async def job() -> object:
        async with session_scope() as session:
            return await purge_feature_snapshots(session, feature_set_version=version)

    typer.echo(f"deleted {_run(job):,} snapshots")


@ml_app.command("train")
def cmd_ml_train(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Single target. Default: all five."),
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
) -> None:
    """Train LightGBM models and register them.

    Each model is scored on a chronological held-out test split and compared to
    the best baseline on exactly those rows. A model that does not beat its
    baseline is registered but never activated -- a model worse than arithmetic
    is a liability, not an improvement.
    """
    from euroleague_ai.models.enums import PredictionTarget
    from euroleague_ai.ml.pipeline import train_and_register
    from euroleague_ai.ml.report import DEFAULT_TARGETS

    if target:
        try:
            targets = (PredictionTarget[target.upper()],)
        except KeyError as exc:
            valid = ", ".join(t.name for t in PredictionTarget)
            raise typer.BadParameter(f"unknown target {target!r}. Valid: {valid}") from exc
    else:
        targets = DEFAULT_TARGETS

    codes: list[str] | None = None
    if start is not None or end is not None:
        codes = [f"E{y}" for y in range((start or 2000), (end or 2025) + 1)]

    async def job() -> object:
        async with session_scope() as session:
            return await train_and_register(
                session, targets=targets, season_codes=codes, activate=activate
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

        s = block["summary"]
        ok = s["beats_baseline"]
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

        typer.secho(f"  {name}", fg=typer.colors.CYAN, bold=True)
        typer.echo(
            f"    model MAE    {s['model_mae']:.4f}   (RMSE {s['rmse']:.3f}, "
            f"R2 {s['r2']:.3f}, bias {s['bias']:+.3f})"
        )
        typer.echo(f"    baseline     {s['baseline_mae']:.4f}   ({s['baseline']})")
        typer.secho(
            f"    improvement  {s['improvement_pct']:+.2f}%   "
            f"{'PASS' if ok else 'FAIL - not activated'}",
            fg=typer.colors.GREEN if ok else typer.colors.RED,
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

        typer.echo(f"    top features: " + ", ".join(n for n, _ in block["top_features"][:5]))
        typer.echo(f"    artifact:     {block['model']['artifact']}")

    typer.echo("")
    typer.secho(
        f"{passed} target(s) passed the gate, {failed} failed.",
        fg=typer.colors.GREEN if failed == 0 else typer.colors.YELLOW,
        bold=True,
    )


@ml_app.command("models")
def cmd_ml_models() -> None:
    """List registered models, newest first."""
    from euroleague_ai.ml.registry import list_models

    async def job() -> object:
        async with session_scope() as session:
            rows = await list_models(session)
            return [
                {
                    "name": m.name,
                    "version": m.version,
                    "target": str(m.target),
                    "active": m.is_active,
                    "mae": (m.metrics or {}).get("model_mae"),
                    "baseline_mae": (m.metrics or {}).get("baseline_mae"),
                    "improvement": (m.metrics or {}).get("improvement_pct"),
                    "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                }
                for m in rows
            ]

    models = _run(job)
    if not models:  # type: ignore[truthy-bool]
        typer.echo("no models registered yet - run `euroleague-ai ml train`")
        return

    typer.echo("")
    typer.echo(f"  {'name':<22}{'ver':<8}{'MAE':>9}{'base':>9}{'impr%':>8}  active")
    for m in models:  # type: ignore[union-attr]
        mae = m["mae"] if m["mae"] is not None else float("nan")
        base = m["baseline_mae"] if m["baseline_mae"] is not None else float("nan")
        impr = m["improvement"] if m["improvement"] is not None else float("nan")
        typer.secho(
            f"  {m['name']:<22}{m['version']:<8}{mae:>9.4f}{base:>9.4f}{impr:>8.2f}"
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

    async def job() -> object:
        async with session_scope() as session:
            return await predict_upcoming(
                session,
                season_code=season or None,
                within_days=days,
                game_codes=list(game_code) if game_code else None,
            )

    result = _run(job)
    typer.echo("")
    if "error" in result:  # type: ignore[operator]
        typer.secho(f"  {result['error']}", fg=typer.colors.RED, bold=True)  # type: ignore[index]
        raise typer.Exit(code=1)

    typer.secho(
        f"  {result['predictions']:,} predictions across {result['games']} game(s)",  # type: ignore[index]
        fg=typer.colors.GREEN,
        bold=True,
    )
    for target, model in (result.get("models") or {}).items():  # type: ignore[union-attr]
        typer.echo(f"    {target:<24} {model}")
    if result["games"] == 0:  # type: ignore[index]
        typer.secho(
            "  No scheduled games in the window. Ingest a season with upcoming\n"
            "  fixtures first:  euroleague-ai ingest all --season E2026",
            fg=typer.colors.YELLOW,
        )


@ml_app.command("show")
def cmd_ml_show(
    game_code: Annotated[int, typer.Option("--game-code", "-g")],
    season: SeasonOption = "",
) -> None:
    """Print the projected lines for one game."""
    from sqlalchemy import select

    from euroleague_ai.models.competition import Season
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.people import Player
    from euroleague_ai.models.prediction import Prediction

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
                raise typer.BadParameter(
                    f"game {game_code} not found in season {season_code}"
                )

            rows = (
                await session.execute(
                    select(Player.full_name, Prediction.target, Prediction.predicted_value)
                    .join(Player, Player.id == Prediction.player_id)
                    .where(Prediction.game_id == game.id)
                )
            ).all()

            by_player: dict[str, dict[str, float]] = {}
            for full_name, target, value in rows:
                by_player.setdefault(full_name, {})[str(target)] = float(value)
            return {"tip_off": game.tip_off_utc.isoformat(), "players": by_player}

    data = _run(job)
    players = data["players"]  # type: ignore[index]
    if not players:
        typer.echo("no predictions for this game - run `euroleague-ai ml predict`")
        return

    order = ["MINUTES", "POINTS", "REBOUNDS", "ASSISTS", "THREE_POINTERS_MADE"]
    typer.echo("")
    typer.secho(f"  tip-off {data['tip_off']}", bold=True)  # type: ignore[index]
    typer.echo(f"  {'player':<28}{'MIN':>7}{'PTS':>7}{'REB':>7}{'AST':>7}{'3PM':>7}")
    for name, values in sorted(
        players.items(), key=lambda kv: -kv[1].get("MINUTES", 0.0)
    ):
        cells = "".join(f"{values.get(t, float('nan')):>7.1f}" for t in order)
        typer.echo(f"  {name:<28}{cells}")


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
    from euroleague_ai.ml.inference import candidate_players, current_club

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
                raise typer.BadParameter(
                    f"game {game_code} not found in season {season_code}"
                )

            cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
            club_by_player = await current_club(session, as_of=cutoff)

            sides = []
            for team_id, label in (
                (game.home_team_id, "HOME"),
                (game.away_team_id, "AWAY"),
            ):
                team = (
                    await session.execute(select(Team).where(Team.id == team_id))
                ).scalar_one()
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
        typer.secho(f"  {label}  {team_name}  ({len(players)} players)",
                    fg=typer.colors.CYAN, bold=True)
        for name, registered in players:
            typer.echo(f"    {name:<32} registered {registered}")
        if not players:
            typer.secho(
                "    none resolved - roster data for this season may be missing",
                fg=typer.colors.YELLOW,
            )

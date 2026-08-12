"""Ingestion jobs.

Plain async functions taking a session and a provider. They are invoked today by
the CLI; when retries and scheduling are actually needed, a Celery task becomes a
thin wrapper around each of these rather than a rewrite. That is the whole reason
they take their dependencies as arguments instead of reaching for globals.

Every run is recorded in ``ingestion_runs`` so a failed overnight backfill is
diagnosable from a table rather than from log archaeology.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.logging import get_logger, set_correlation_id
from euroleague_ai.models.enums import IngestionStatus
from euroleague_ai.models.ingestion import IngestionRun
from euroleague_ai.providers.base import BasketballDataProvider, SeasonRef
from euroleague_ai.ingestion.identity import IdentityResolver
from euroleague_ai.ingestion.pipeline import IngestionPipeline, IngestionResult

logger = get_logger(__name__)


@asynccontextmanager
async def _run_record(
    session: AsyncSession,
    *,
    job_name: str,
    provider_key: str,
    season_code: str,
    clock: Clock,
) -> AsyncIterator[IngestionRun]:
    """Record an ingestion run, marking it failed on exception.

    The run row is committed independently of the ingestion data so that a
    failure leaves an audit trail rather than rolling away the evidence.
    """
    set_correlation_id()
    run = IngestionRun(
        id=uuid.uuid4(),
        job_name=job_name,
        provider_key=provider_key,
        season_code=season_code,
        status=IngestionStatus.RUNNING,
        started_at=clock.now(),
    )
    session.add(run)
    await session.flush()

    try:
        yield run
    except Exception as exc:
        run.status = IngestionStatus.FAILED
        run.finished_at = clock.now()
        run.error_message = f"{type(exc).__name__}: {exc}"
        await session.flush()
        logger.exception("ingestion job failed", extra={"job": job_name})
        raise
    else:
        run.status = IngestionStatus.SUCCEEDED
        run.finished_at = clock.now()
        await session.flush()
        logger.info(
            "ingestion job finished",
            extra={
                "job": job_name,
                "rows_read": run.rows_read,
                "rows_written": run.rows_written,
            },
        )


def _apply(run: IngestionRun, result: IngestionResult) -> None:
    run.rows_read += result.rows_read
    run.rows_written += result.rows_written
    if result.skipped:
        run.details = {**run.details, "skipped": run.details.get("skipped", 0) + result.skipped}


async def _prepare(
    session: AsyncSession, provider: BasketballDataProvider, season_ref: SeasonRef, clock: Clock
) -> tuple[IngestionPipeline, object]:
    resolver = IdentityResolver(session, provider.provider_key, clock=clock)
    pipeline = IngestionPipeline(session, resolver, clock=clock)
    league = await pipeline.ensure_league()
    season = await pipeline.ensure_season(
        league, season_ref.season_code, season_ref.start_year
    )
    return pipeline, season


async def ingest_teams(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    clock: Clock | None = None,
) -> IngestionResult:
    """Fetch and store every club registered in the season."""
    clock = clock or SystemClock()
    async with _run_record(
        session,
        job_name="teams",
        provider_key=provider.provider_key,
        season_code=season_ref.season_code,
        clock=clock,
    ) as run:
        pipeline, season = await _prepare(session, provider, season_ref, clock)
        teams = await provider.fetch_teams(season_ref)
        result = await pipeline.upsert_teams(teams, season)  # type: ignore[arg-type]
        _apply(run, result)
        return result


async def ingest_players(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    clock: Clock | None = None,
) -> IngestionResult:
    """Fetch and store player biography plus roster intervals.

    Teams must already be ingested: a registration referencing an unknown club
    is skipped rather than inventing a team row.
    """
    clock = clock or SystemClock()
    async with _run_record(
        session,
        job_name="players",
        provider_key=provider.provider_key,
        season_code=season_ref.season_code,
        clock=clock,
    ) as run:
        pipeline, season = await _prepare(session, provider, season_ref, clock)
        stints = await provider.fetch_players(season_ref)
        result = await pipeline.upsert_player_stints(stints, season)  # type: ignore[arg-type]
        _apply(run, result)
        return result


async def ingest_games(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    round_number: int | None = None,
    clock: Clock | None = None,
) -> IngestionResult:
    """Fetch and store the schedule."""
    clock = clock or SystemClock()
    async with _run_record(
        session,
        job_name="games",
        provider_key=provider.provider_key,
        season_code=season_ref.season_code,
        clock=clock,
    ) as run:
        pipeline, season = await _prepare(session, provider, season_ref, clock)
        games = await provider.fetch_games(season_ref, round_number=round_number)
        result = await pipeline.upsert_games(games, season)  # type: ignore[arg-type]
        _apply(run, result)
        return result


async def ingest_game_stats(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    game_codes: list[int] | None = None,
    clock: Clock | None = None,
    progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> IngestionResult:
    """Fetch and store box scores for finished games.

    When ``game_codes`` is omitted, every finished game in the season is
    refreshed. That is idempotent -- box scores upsert on
    ``(game_team, player)`` -- so re-running after a stat correction is safe.
    """
    clock = clock or SystemClock()
    async with _run_record(
        session,
        job_name="game_stats",
        provider_key=provider.provider_key,
        season_code=season_ref.season_code,
        clock=clock,
    ) as run:
        pipeline, season = await _prepare(session, provider, season_ref, clock)
        codes = game_codes if game_codes is not None else await pipeline.finished_game_codes(
            season  # type: ignore[arg-type]
        )

        aggregate = IngestionResult()
        for index, code in enumerate(codes, start=1):
            stats = await provider.fetch_game_stats(season_ref, code)
            if stats is None:
                aggregate.skipped += 1
                continue
            aggregate.merge(await pipeline.upsert_game_stats(stats, season))  # type: ignore[arg-type]
            if progress is not None:
                await progress(index, len(codes))

        _apply(run, aggregate)
        return aggregate


async def ingest_all(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    clock: Clock | None = None,
) -> IngestionResult:
    """Run every job in dependency order.

    Order matters and is not arbitrary: players reference teams, games reference
    teams, and box scores reference games. Running them out of order does not
    corrupt anything -- unresolved references are skipped and counted -- but it
    does silently lose rows.
    """
    clock = clock or SystemClock()
    aggregate = IngestionResult()
    aggregate.merge(await ingest_teams(session, provider, season_ref, clock=clock))
    aggregate.merge(await ingest_players(session, provider, season_ref, clock=clock))
    aggregate.merge(await ingest_games(session, provider, season_ref, clock=clock))
    aggregate.merge(await ingest_game_stats(session, provider, season_ref, clock=clock))
    return aggregate

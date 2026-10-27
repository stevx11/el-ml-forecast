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
from euroleague_ai.core.errors import ProviderError
from euroleague_ai.core.logging import get_logger, set_correlation_id
from euroleague_ai.ingestion.identity import IdentityResolver
from euroleague_ai.ingestion.pipeline import IngestionPipeline, IngestionResult
from euroleague_ai.models.enums import IngestionStatus
from euroleague_ai.models.ingestion import IngestionRun
from euroleague_ai.providers.base import BasketballDataProvider, SeasonRef
from euroleague_ai.providers.euroleague import RateLimitCooldown

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
    season = await pipeline.ensure_season(league, season_ref.season_code, season_ref.start_year)
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


#: A backfill covers hundreds of games. Committing periodically means an
#: interruption costs the current chunk, not the entire run.
COMMIT_EVERY = 10

#: Give up only when the upstream API is clearly unavailable, rather than on the
#: first transient failure. Isolated 404s and malformed payloads are normal.
MAX_CONSECUTIVE_FAILURES = 8


async def ingest_game_stats(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    game_codes: list[int] | None = None,
    refresh_existing: bool = False,
    clock: Clock | None = None,
    progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> IngestionResult:
    """Fetch and store box scores for finished games.

    Resumable and fault-tolerant, because a season backfill is hundreds of
    requests against a rate-limited public API and *something* will fail:

    * **Per-game isolation.** A failure on one game is counted and skipped, not
      propagated. Losing 400 games' work because game 21 returned a 429 is not
      an acceptable failure mode.
    * **Incremental commits.** Progress is committed every
      :data:`COMMIT_EVERY` games, so an interruption costs the current chunk
      rather than everything.
    * **Resume by default.** Games that already have box scores are skipped, so
      re-running continues where the last run stopped. Pass
      ``refresh_existing=True`` to re-fetch everything, which is what you want
      after an upstream stat correction.
    * **Circuit breaker.** After :data:`MAX_CONSECUTIVE_FAILURES` failures in a
      row the API is presumed down and the job stops rather than grinding
      through hundreds of doomed requests.

    Args:
        game_codes: Specific games. Defaults to every finished game in the season.
        refresh_existing: Re-fetch games that already have box scores.
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

        if game_codes is not None:
            codes = list(game_codes)
        elif refresh_existing:
            codes = await pipeline.finished_game_codes(season)  # type: ignore[arg-type]
        else:
            codes = await pipeline.game_codes_missing_stats(season)  # type: ignore[arg-type]

        total = len(codes)
        logger.info(
            "ingesting box scores",
            extra={"games": total, "refresh_existing": refresh_existing},
        )

        aggregate = IngestionResult()
        failures: list[int] = []
        consecutive = 0

        for index, code in enumerate(codes, start=1):
            try:
                stats = await provider.fetch_game_stats(season_ref, code)
            except RateLimitCooldown as exc:
                # Not a per-game failure: the upstream has closed the door for a
                # while. Stop cleanly with progress committed rather than burning
                # the remaining games against a limiter that will refuse them all.
                remaining = total - index + 1
                logger.warning(
                    "provider imposed a cooldown; stopping with progress saved",
                    extra={
                        "retry_after_seconds": round(exc.retry_after),
                        "completed": index - 1,
                        "remaining": remaining,
                        "resume_hint": "re-run the same command to continue",
                    },
                )
                run.details = {
                    **run.details,
                    "stopped_by": "rate_limit_cooldown",
                    "retry_after_seconds": round(exc.retry_after),
                    "remaining": remaining,
                }
                break
            except ProviderError as exc:
                consecutive += 1
                failures.append(code)
                logger.warning(
                    "box score fetch failed; continuing",
                    extra={
                        "game_code": code,
                        "error": str(exc),
                        "consecutive_failures": consecutive,
                    },
                )
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "too many consecutive failures; stopping early. "
                        "Re-run to resume from here.",
                        extra={"game_code": code, "completed": index - 1, "total": total},
                    )
                    break
                continue

            consecutive = 0

            if stats is None:
                aggregate.skipped += 1
                continue

            aggregate.merge(await pipeline.upsert_game_stats(stats, season))  # type: ignore[arg-type]

            if index % COMMIT_EVERY == 0:
                await session.commit()
                logger.info("box score progress", extra={"done": index, "total": total})

            if progress is not None:
                await progress(index, total)

        await session.commit()

        if failures:
            aggregate.details["failed"] = len(failures)
            run.details = {
                **run.details,
                "failed_game_codes": failures[:50],
                "failed_count": len(failures),
            }
            logger.warning(
                "some box scores could not be fetched; re-run to retry them",
                extra={"failed": len(failures), "total": total},
            )

        _apply(run, aggregate)
        return aggregate


async def ingest_all(
    session: AsyncSession,
    provider: BasketballDataProvider,
    season_ref: SeasonRef,
    *,
    clock: Clock | None = None,
) -> IngestionResult:
    """Run every job in dependency order, committing after each.

    Order matters and is not arbitrary: players reference teams, games reference
    teams, and box scores reference games. Running them out of order does not
    corrupt anything -- unresolved references are skipped and counted -- but it
    does silently lose rows.

    **Each job commits before the next begins.** Without that, all four jobs
    share one transaction and a failure in the last one discards the first
    three: a box score error would throw away the teams, players, and schedule
    that had already been fetched successfully, so the retry has to start from
    nothing. Each job is independently meaningful and is therefore durable on
    its own.
    """
    clock = clock or SystemClock()
    aggregate = IngestionResult()

    for name, job in (
        ("teams", ingest_teams),
        ("players", ingest_players),
        ("games", ingest_games),
        ("game_stats", ingest_game_stats),
    ):
        aggregate.merge(await job(session, provider, season_ref, clock=clock))
        await session.commit()
        logger.info("stage committed", extra={"stage": name})

    return aggregate

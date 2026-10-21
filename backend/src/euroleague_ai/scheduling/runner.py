"""The daily pipeline, and the loop that runs it unattended.

One command that does what a human would otherwise do every morning: pull the
latest fixtures and box scores, score the predictions whose games have now been
played, and refresh projections for what is coming.

**No Celery, no broker, no worker pool.** The work is one composed job on a
timer against a single database. A task queue would add a broker, a result
backend, a worker process and a whole class of failure modes, to run one job
twice a day. If the workload ever justifies it the job functions here are
already the right shape to hand to a worker.

**Every step is independently recoverable.** A rate-limited ingest must not stop
predictions being refreshed from the data already held, and a scoring failure
must not block ingestion tomorrow. Each step is therefore caught, recorded and
reported separately; the run is marked failed if any step failed, but the later
steps still get their chance.

**Training is deliberately not in here.** Retraining on a timer would let a model
promote itself to production overnight on the strength of a gate that this
project has already watched pass while a defect degraded every target. Training
stays a decision a person makes, having read the numbers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.enums import IngestionStatus
from euroleague_ai.models.ingestion import IngestionRun

logger = get_logger(__name__)

#: Advisory lock for the pipeline. Distinct from the backfill's key so a long
#: snapshot rebuild and a scheduled run do not lock each other out -- they touch
#: different tables and there is no reason for them to queue.
ADVISORY_LOCK_KEY = 815_243_002

#: Recorded as ``job_name`` on every run, so `SELECT ... WHERE job_name` gives
#: the pipeline's history without disturbing the ingestion audit trail.
JOB_NAME = "pipeline.daily"

#: Resolved here, at import, rather than inline at the end of a run.
#:
#: The first version wrote ``IngestionStatus.SUCCESS``, which does not exist --
#: the member is ``SUCCEEDED``. Because the reference sat on the last line of a
#: long job, the whole pipeline ingested, scored and predicted successfully and
#: *then* died with an AttributeError while recording that it had succeeded.
#: Binding the members up here means a typo is an ImportError that every test
#: run catches, instead of a crash that only appears after several minutes of
#: real work.
STATUS_RUNNING = IngestionStatus.RUNNING
STATUS_SUCCEEDED = IngestionStatus.SUCCEEDED
STATUS_FAILED = IngestionStatus.FAILED


@dataclass(slots=True)
class StepResult:
    """One step of the pipeline."""

    name: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "error": self.error}


@dataclass(slots=True)
class PipelineResult:
    """A whole run."""

    steps: list[StepResult] = field(default_factory=list)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "steps": [step.as_dict() for step in self.steps],
        }


async def _try_lock(session: AsyncSession) -> bool:
    """Session-scoped advisory lock, released when the connection closes.

    A crashed run therefore cannot leave the schedule permanently blocked, which
    a table-based lock would.
    """
    return bool(
        (
            await session.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": ADVISORY_LOCK_KEY}
            )
        ).scalar_one()
    )


async def run_daily(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    within_days: int = 7,
    clock: Clock | None = None,
    settings: Settings | None = None,
    skip_ingest: bool = False,
) -> PipelineResult:
    """Ingest, score, then predict -- in that order, for a reason.

    Ingest first so the box scores exist to score against. Score before
    predicting so a run that dies partway through has still recorded how
    yesterday went, which is the part that cannot be recovered later from
    upstream.

    Returns a result rather than raising: an unattended job that raises produces
    a stack trace nobody reads, while one that records a failed step produces a
    row somebody can query.
    """
    clock = clock or SystemClock()
    settings = settings or get_settings()
    season = season_code or settings.euroleague_default_season
    started = clock.now()

    result = PipelineResult()

    if not await _try_lock(session):
        logger.warning("pipeline already running; skipping this slot")
        result.skipped = True
        return result

    run = IngestionRun(
        id=uuid.uuid4(),
        job_name=JOB_NAME,
        provider_key="pipeline",
        season_code=season,
        status=STATUS_RUNNING,
        started_at=started,
    )
    session.add(run)
    await session.commit()

    if not skip_ingest:
        result.steps.append(await _step_ingest(session, season=season, clock=clock))
    result.steps.append(await _step_score(session, season=season, clock=clock))
    if settings.injury_report_url:
        result.steps.append(await _step_injury_report(session, settings=settings, clock=clock))
    if settings.news_feed_urls:
        # Before predicting, so this morning's injury news reaches this
        # afternoon's projections.
        result.steps.append(await _step_news(session, settings=settings, clock=clock))
    result.steps.append(
        await _step_predict(session, season=season, within_days=within_days, clock=clock)
    )

    run.finished_at = clock.now()
    run.status = STATUS_SUCCEEDED if result.ok else STATUS_FAILED
    run.details = result.as_dict()
    failures = [s for s in result.steps if not s.ok]
    if failures:
        run.error_message = "; ".join(f"{s.name}: {s.error}" for s in failures)[:2000]
    await session.commit()

    logger.info("pipeline run finished", extra={"ok": result.ok, "season": season})
    return result


async def _step_ingest(
    session: AsyncSession, *, season: str, clock: Clock
) -> StepResult:
    """Pull fixtures, rosters and any new box scores for the current season."""
    from euroleague_ai.ingestion.jobs import ingest_all
    from euroleague_ai.providers.factory import build_provider, season_ref

    settings = get_settings()
    # wait_out_cooldowns stays off. Unattended does not mean patient: a run that
    # blocks for a five-minute provider cooldown holds the advisory lock and can
    # collide with the next scheduled slot. Ingestion is resumable, so giving up
    # and retrying at the next slot is strictly better than waiting.
    provider = build_provider(settings, wait_out_cooldowns=False)
    try:
        summary = await ingest_all(
            session, provider, season_ref(season, settings), clock=clock
        )
        await session.commit()
        return StepResult("ingest", True, detail={"summary": str(summary)})
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        await session.rollback()
        logger.warning("pipeline ingest failed", extra={"error": str(exc)})
        # Deliberately not re-raised. A provider cooldown is a routine event and
        # must not stop the run from scoring and predicting on data already held.
        return StepResult("ingest", False, error=str(exc))
    finally:
        await provider.aclose()


async def _step_score(session: AsyncSession, *, season: str, clock: Clock) -> StepResult:
    """Attach realised box scores to predictions whose games have been played."""
    from euroleague_ai.ml.outcomes import score_predictions

    try:
        scored = await score_predictions(session, season_code=season, clock=clock)
        await session.commit()
        return StepResult("score", True, detail=scored.as_dict())
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        await session.rollback()
        logger.warning("pipeline scoring failed", extra={"error": str(exc)})
        return StepResult("score", False, error=str(exc))


async def _step_injury_report(
    session: AsyncSession, *, settings: Settings, clock: Clock
) -> StepResult:
    """Read the daily injury table and record what changed."""
    from euroleague_ai.news.injury_report import collect_injury_report

    try:
        imported = await collect_injury_report(session, settings=settings, now=clock.now())
        await session.commit()
        return StepResult("injury_report", True, detail=imported.as_dict())
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        await session.rollback()
        logger.warning("pipeline injury report failed", extra={"error": str(exc)})
        # Not fatal: predictions still run on the reports already stored.
        return StepResult("injury_report", False, error=str(exc))


async def _step_news(session: AsyncSession, *, settings: Settings, clock: Clock) -> StepResult:
    """Collect configured feeds and read injury statuses out of new articles."""
    from euroleague_ai.news.pipeline import collect_news

    try:
        collected = await collect_news(session, settings=settings, now=clock.now())
        await session.commit()
        return StepResult("news", True, detail=collected.as_dict())
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        await session.rollback()
        logger.warning("pipeline news collection failed", extra={"error": str(exc)})
        # Not fatal: predictions still run on the reports already stored.
        return StepResult("news", False, error=str(exc))


async def _step_predict(
    session: AsyncSession, *, season: str, within_days: int, clock: Clock
) -> StepResult:
    """Refresh projections for fixtures inside the window."""
    from euroleague_ai.ml.inference import predict_upcoming

    try:
        predicted = await predict_upcoming(
            session, season_code=season, within_days=within_days, clock=clock
        )
        await session.commit()
        if "error" in predicted:
            # No active models is a configuration problem, not a transient one,
            # and should show up as a failed step rather than a quiet success.
            return StepResult("predict", False, error=str(predicted["error"]))
        return StepResult("predict", True, detail=predicted)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        await session.rollback()
        logger.warning("pipeline prediction failed", extra={"error": str(exc)})
        return StepResult("predict", False, error=str(exc))


async def recent_runs(session: AsyncSession, *, limit: int = 10) -> list[IngestionRun]:
    """The pipeline's own history, newest first.

    The point of an unattended job is that nobody watches it, which is exactly
    why it has to leave a trail that can be read afterwards.
    """
    from sqlalchemy import select

    stmt = (
        select(IngestionRun)
        .where(IngestionRun.job_name == JOB_NAME)
        .order_by(IngestionRun.started_at.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())

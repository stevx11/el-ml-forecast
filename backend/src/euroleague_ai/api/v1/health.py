"""Liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from euroleague_ai import __version__
from euroleague_ai.api.deps import SessionDep, SettingsDep
from euroleague_ai.core.logging import get_logger
from euroleague_ai.schemas.health import HealthOut, ReadinessOut

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthOut)
async def liveness(settings: SettingsDep) -> HealthOut:
    """Process liveness. Touches no dependencies by design.

    A liveness probe that checks the database restarts the application whenever
    the database blips, which turns a recoverable outage into a crash loop.
    """
    return HealthOut(
        app=settings.app_name,
        version=__version__,
        environment=str(settings.environment),
    )


@router.get("/health/ready", response_model=ReadinessOut)
async def readiness(
    session: SessionDep, settings: SettingsDep, response: Response
) -> ReadinessOut:
    """Dependency readiness. Poll this before routing traffic."""
    database_ok = True
    detail: str | None = None

    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - report, do not propagate
        database_ok = False
        detail = f"database: {type(exc).__name__}"
        logger.warning("readiness: database unreachable", extra={"error": str(exc)})

    redis_ok = await _check_redis(settings.redis_url)

    # Redis is reported but does not gate readiness -- the cache is optional and
    # the API is fully correct without it.
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessOut(
        status="ready" if database_ok else "degraded",
        database=database_ok,
        redis=redis_ok,
        detail=detail,
    )


async def _check_redis(redis_url: object) -> bool:
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(str(redis_url))
        try:
            await client.ping()
            return True
        finally:
            await client.aclose()
    except Exception:  # noqa: BLE001
        return False

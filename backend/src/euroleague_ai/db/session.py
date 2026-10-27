"""Async engine and session lifecycle.

A single engine is created at application startup and disposed at shutdown.
Sessions are per-request (or per-job) and are never shared across tasks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import ConfigurationError

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings | None = None, *, dsn: str | None = None) -> AsyncEngine:
    """Build a new async engine. Does not touch module state."""
    resolved = settings or get_settings()
    return create_async_engine(
        dsn or resolved.database_dsn,
        echo=resolved.database_echo,
        pool_size=resolved.database_pool_size,
        max_overflow=resolved.database_max_overflow,
        pool_pre_ping=True,  # survives Postgres restarts and idle-connection reaping
        future=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a session factory bound to ``engine``."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # response serialisation happens after commit
        autoflush=False,  # flushes are explicit; ingestion depends on ordering
    )


def init_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create and store the process-wide engine. Called once at startup."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(settings)
        _session_factory = create_session_factory(_engine)
    return _engine


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    return _engine if _engine is not None else init_engine()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    if _session_factory is None:
        init_engine()
    if _session_factory is None:  # pragma: no cover - defensive
        raise ConfigurationError("session factory was not initialised")
    return _session_factory


async def dispose_engine() -> None:
    """Close all pooled connections. Called at shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session.

    The session is rolled back on any exception so a failed request can never
    leave a partial write behind.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session for CLI jobs and scripts.

    Commits on success, rolls back on failure.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()

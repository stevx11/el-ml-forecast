"""Shared pytest fixtures.

Database-backed tests are marked ``integration`` and **skip** rather than fail
when ``TEST_DATABASE_URL`` is unset, so ``pytest`` works on a laptop with no
Docker running. Everything else is pure and always runs.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.db.base import Base
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "euroleague"

# The instant ingestion is pretended to run at. Fixed so that observed_at values
# are deterministic and the leakage tests can reason about them precisely.
INGEST_AT = datetime(2025, 9, 1, 12, 0, tzinfo=UTC)

SEASON = SeasonRef(competition_code="E", season_code="E2025", start_year=2025)

# Native enum types are not dropped by Base.metadata.drop_all, so a re-run would
# hit "type already exists" when the migration tries to create them.
_ENUM_TYPE_NAMES = (
    "entitytype",
    "gamestatus",
    "teamside",
    "playerposition",
    "availabilitystatus",
    "statussourcetype",
    "newseventtype",
    "predictiontarget",
    "ingestionstatus",
)


def _test_database_url() -> str | None:
    """Resolve the test database DSN from the environment or settings."""
    if explicit := os.environ.get("TEST_DATABASE_URL"):
        return explicit
    try:
        configured = get_settings().test_database_url
    except Exception:
        return None
    return str(configured) if configured else None


requires_db = pytest.mark.skipif(
    _test_database_url() is None,
    reason="TEST_DATABASE_URL is not set; database-backed tests skipped",
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def ingest_clock() -> FixedClock:
    """A clock frozen at :data:`INGEST_AT`."""
    return FixedClock(INGEST_AT)


@pytest.fixture
def fixture_provider() -> FixtureProvider:
    """Offline provider reading the committed EuroLeague fixtures."""
    return FixtureProvider(FIXTURE_ROOT)


@pytest.fixture
def season() -> SeasonRef:
    return SEASON


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def alembic_config(dsn: str) -> Config:
    """An Alembic config pointed at ``dsn``.

    Used by both the schema fixture and the migration test, so the suite and the
    drift check cannot disagree about which migration they are running.
    """
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn)
    return config


@pytest_asyncio.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    """Session-scoped engine against the test database.

    The schema is built by **running the real migration**, not by
    ``Base.metadata.create_all``. That costs a second at session start and buys
    something worth far more: every test in the suite runs against the schema
    that will actually be deployed. A model change that never got a migration
    fails here rather than in production.
    """
    dsn = _test_database_url()
    if dsn is None:  # pragma: no cover - guarded by requires_db
        pytest.skip("TEST_DATABASE_URL is not set")

    engine = create_async_engine(dsn, future=True)

    # Clear whatever a previous run left behind, including a partial migration.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        for enum_name in _ENUM_TYPE_NAMES:
            await connection.execute(text(f"DROP TYPE IF EXISTS {enum_name} CASCADE"))

    # Alembic's env.py calls asyncio.run(), which cannot run inside an already
    # running loop. Off-loading to a worker thread gives it a clean thread with
    # no loop attached.
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def connection(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """A connection wrapped in a transaction that is always rolled back.

    Every test therefore starts from the same clean schema without the cost of
    recreating it, and no test can leave state behind for the next one.
    """
    async with engine.connect() as conn:
        transaction = await conn.begin()
        try:
            yield conn
        finally:
            await transaction.rollback()


@pytest_asyncio.fixture
async def session(connection: AsyncConnection) -> AsyncIterator[AsyncSession]:
    """A session bound to the rolled-back connection."""
    async with AsyncSession(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    ) as db_session:
        yield db_session

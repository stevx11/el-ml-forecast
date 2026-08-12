"""API tests: the app starts, routes respond, and the DB dependency resolves."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.db.session import get_session
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.main import create_app
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

CLOCK = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    """An HTTP client wired to the rolled-back test session.

    Overriding ``get_session`` rather than pointing the app at the test database
    keeps every request inside the same transaction, so API tests leave no state
    behind either.
    """
    app = create_app()

    async def override() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def seeded(
    session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
) -> None:
    await ingest_all(session, fixture_provider, season, clock=CLOCK)
    await session.flush()


class TestHealth:
    async def test_liveness_needs_no_dependencies(self, client: AsyncClient) -> None:
        """A liveness probe that checks the database turns an outage into a crash loop."""
        response = await client.get("/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    async def test_readiness_reports_the_database(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["database"] is True
        assert body["status"] == "ready"

    async def test_correlation_id_is_echoed(self, client: AsyncClient) -> None:
        """So a user can quote it in a bug report and the logs can be found."""
        response = await client.get(
            "/health/live", headers={"X-Correlation-ID": "abc123"}
        )
        assert response.headers["X-Correlation-ID"] == "abc123"


class TestOpenAPI:
    async def test_schema_is_served(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        assert "paths" in response.json()


class TestCatalogEndpoints:
    async def test_leagues(self, client: AsyncClient, seeded: None) -> None:
        response = await client.get("/api/v1/leagues")
        assert response.status_code == 200
        assert any(row["code"] == "EUROLEAGUE" for row in response.json())

    async def test_teams_are_paginated(self, client: AsyncClient, seeded: None) -> None:
        response = await client.get("/api/v1/teams", params={"limit": 2})
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert body["meta"]["limit"] == 2

    async def test_player_search(self, client: AsyncClient, seeded: None) -> None:
        response = await client.get("/api/v1/players", params={"search": "PETROVIC"})
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["full_name"].startswith("PETROVIC")

    async def test_player_game_log_renders_minutes(
        self, client: AsyncClient, seeded: None
    ) -> None:
        """Storage keeps seconds; the API presents minutes."""
        listing = await client.get("/api/v1/players", params={"search": "PETROVIC"})
        player_id = listing.json()["items"][0]["id"]

        response = await client.get(f"/api/v1/players/{player_id}/stats")
        assert response.status_code == 200
        rows = response.json()
        assert rows
        assert rows[0]["seconds_played"] == 1800
        assert rows[0]["minutes"] == 30.0

    async def test_games_listing(self, client: AsyncClient, seeded: None) -> None:
        response = await client.get("/api/v1/games", params={"limit": 10})
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 6
        assert items[0]["home_team"]["code"]

    async def test_game_detail_includes_the_box_score(
        self, client: AsyncClient, seeded: None
    ) -> None:
        listing = await client.get("/api/v1/games", params={"status": "FINISHED", "limit": 1})
        game_id = listing.json()["items"][0]["id"]

        response = await client.get(f"/api/v1/games/{game_id}")
        assert response.status_code == 200
        body = response.json()
        assert len(body["box_score"]) > 0
        assert {"HOME", "AWAY"} == {line["side"] for line in body["box_score"]}

    async def test_scheduled_game_has_an_empty_box_score(
        self, client: AsyncClient, seeded: None
    ) -> None:
        listing = await client.get("/api/v1/games", params={"status": "SCHEDULED"})
        game_id = listing.json()["items"][0]["id"]

        response = await client.get(f"/api/v1/games/{game_id}")
        assert response.json()["box_score"] == []


class TestErrors:
    async def test_unknown_uuid_is_404(self, client: AsyncClient, seeded: None) -> None:
        response = await client.get(
            "/api/v1/players/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404

    async def test_malformed_uuid_is_422(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/players/not-a-uuid")).status_code == 422

    async def test_oversized_page_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/teams", params={"limit": 5000})).status_code == 422

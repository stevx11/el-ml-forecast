"""Adapter for the public EuroLeague JSON API.

Talks to ``api-live.euroleague.net/v2`` -- the same unauthenticated endpoints
that serve the official website. No HTML scraping happens anywhere in this
codebase.

The client is a polite citizen by default: it identifies itself with a
configurable User-Agent, rate-limits itself, bounds concurrency, retries only on
transient failures, and caches responses in Redis when one is available. If you
intend to use this commercially, obtain a licensed feed and write an adapter for
it -- the protocol makes that a drop-in change.
"""

from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import ProviderError
from euroleague_ai.core.logging import get_logger
from euroleague_ai.providers.base import (
    GameDTO,
    GameStatsDTO,
    PlayerStintDTO,
    SeasonRef,
    TeamDTO,
)
from euroleague_ai.providers.mapping import (
    map_game,
    map_game_stats,
    map_player_stint,
    map_team,
)

logger = get_logger(__name__)

PROVIDER_KEY = "euroleague"

# The API caps page size; requesting more is silently truncated, which would
# cause pagination to stall rather than fail loudly.
_PAGE_SIZE = 100

# Retrying a 4xx is pointless -- the request is wrong and will stay wrong.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _RetryableResponse(Exception):
    """Internal marker so tenacity retries transient HTTP statuses."""


class _RateLimiter:
    """Minimum-interval rate limiter.

    A full token bucket is unnecessary here: the goal is politeness towards a
    free public API, not precise throughput shaping.
    """

    def __init__(self, rate_per_second: float) -> None:
        self._min_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = loop.time()
            self._next_allowed = now + self._min_interval


class ResponseCache:
    """Redis-backed response cache.

    Optional by design. Backfilling a season re-requests the same club and
    player pages repeatedly, and caching removes that load from the upstream API
    entirely. Any cache failure is logged and ignored -- a broken cache must
    never break ingestion.
    """

    def __init__(self, redis_client: Any, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._redis.get(key)
        except Exception as exc:  # noqa: BLE001 - cache must never be fatal
            logger.warning("provider cache read failed", extra={"error": str(exc)})
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    async def set(self, key: str, value: Any) -> None:
        if self._ttl <= 0:
            return
        try:
            await self._redis.set(key, json.dumps(value), ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider cache write failed", extra={"error": str(exc)})


class EuroLeagueProvider:
    """Live EuroLeague data provider.

    Usable as an async context manager::

        async with EuroLeagueProvider() as provider:
            games = await provider.fetch_games(season)
    """

    provider_key = PROVIDER_KEY

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.euroleague_api_base_url,
            timeout=httpx.Timeout(self._settings.euroleague_request_timeout_seconds),
            headers={
                "Accept": "application/json",
                "User-Agent": self._settings.euroleague_user_agent,
            },
            follow_redirects=True,
        )
        self._cache = cache
        self._limiter = _RateLimiter(self._settings.euroleague_request_rate_per_second)
        self._semaphore = asyncio.Semaphore(self._settings.euroleague_max_concurrency)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> EuroLeagueProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    # -- transport ---------------------------------------------------------

    def _season_path(self, season: SeasonRef, suffix: str = "") -> str:
        return (
            f"/competitions/{season.competition_code}"
            f"/seasons/{season.season_code}{suffix}"
        )

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Fetch and decode JSON, with caching, rate limiting and retries."""
        cache_key = f"provider:{self.provider_key}:{path}:{sorted((params or {}).items())}"

        if self._cache is not None and (cached := await self._cache.get(cache_key)) is not None:
            return cached

        payload = await self._get_uncached(path, params)

        if self._cache is not None:
            await self._cache.set(cache_key, payload)
        return payload

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, _RetryableResponse)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        reraise=True,
    )
    async def _get_uncached(self, path: str, params: dict[str, Any] | None) -> Any:
        async with self._semaphore:
            await self._limiter.acquire()
            try:
                response = await self._client.get(path, params=params)
            except httpx.TransportError:
                logger.warning("provider transport error", extra={"path": path})
                raise

        if response.status_code in _RETRYABLE_STATUS:
            logger.warning(
                "provider returned retryable status",
                extra={"path": path, "status": response.status_code},
            )
            raise _RetryableResponse(f"{response.status_code} for {path}")

        if response.status_code == httpx.codes.NOT_FOUND:
            return None

        if response.is_error:
            raise ProviderError(
                self.provider_key, f"HTTP {response.status_code} for {path}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(self.provider_key, f"invalid JSON from {path}") from exc

    async def _get_paginated(
        self, path: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Collect every page of a ``{"data": [...], "total": n}`` endpoint."""
        collected: list[dict[str, Any]] = []
        offset = 0

        while True:
            page_params = {**(params or {}), "limit": _PAGE_SIZE, "offset": offset}
            payload = await self._get(path, page_params)
            if not isinstance(payload, dict):
                break

            rows = payload.get("data")
            if not isinstance(rows, list) or not rows:
                break

            collected.extend(row for row in rows if isinstance(row, dict))
            offset += len(rows)

            total = payload.get("total")
            if not isinstance(total, int) or offset >= total:
                break

        return collected

    # -- provider protocol -------------------------------------------------

    async def fetch_teams(self, season: SeasonRef) -> list[TeamDTO]:
        rows = await self._get_paginated(self._season_path(season, "/clubs"))
        teams = [team for row in rows if (team := map_team(row)) is not None]
        logger.info(
            "fetched teams",
            extra={"season": season.season_code, "count": len(teams), "raw": len(rows)},
        )
        return teams

    async def fetch_players(self, season: SeasonRef) -> list[PlayerStintDTO]:
        # personType=J selects players; coaches and referees use other codes and
        # are ingested separately where needed.
        rows = await self._get_paginated(
            self._season_path(season, "/people"), {"personType": "J"}
        )
        stints = [stint for row in rows if (stint := map_player_stint(row)) is not None]
        logger.info(
            "fetched player registrations",
            extra={"season": season.season_code, "count": len(stints), "raw": len(rows)},
        )
        return stints

    async def fetch_games(
        self, season: SeasonRef, *, round_number: int | None = None
    ) -> list[GameDTO]:
        params: dict[str, Any] = {}
        if round_number is not None:
            params["gameNumber"] = round_number

        rows = await self._get_paginated(self._season_path(season, "/games"), params)
        games = [game for row in rows if (game := map_game(row)) is not None]

        skipped = len(rows) - len(games)
        if skipped:
            # Loud, because a silently dropped game becomes a silently missing
            # training row and a gap in every schedule-density feature.
            logger.warning(
                "skipped unmappable games",
                extra={"season": season.season_code, "skipped": skipped},
            )
        logger.info(
            "fetched games",
            extra={
                "season": season.season_code,
                "round": round_number,
                "count": len(games),
            },
        )
        return games

    async def fetch_game_stats(
        self, season: SeasonRef, game_code: int
    ) -> GameStatsDTO | None:
        payload = await self._get(self._season_path(season, f"/games/{game_code}/stats"))
        if not isinstance(payload, dict):
            return None
        return map_game_stats(game_code, payload)


def build_response_cache(settings: Settings | None = None) -> ResponseCache | None:
    """Build a Redis-backed cache, or ``None`` if Redis is unavailable.

    Returning ``None`` rather than raising keeps ingestion runnable with Redis
    down; the only cost is extra upstream requests.
    """
    resolved = settings or get_settings()
    if resolved.provider_cache_ttl_seconds <= 0:
        return None
    try:
        from redis.asyncio import Redis  # imported lazily: optional at runtime

        client = Redis.from_url(str(resolved.redis_url), decode_responses=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis unavailable, provider cache disabled", extra={"error": str(exc)})
        return None
    return ResponseCache(client, resolved.provider_cache_ttl_seconds)

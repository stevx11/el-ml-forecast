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
from datetime import date
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
    map_club_roster_entry,
    map_game,
    map_game_stats,
    map_team,
)

logger = get_logger(__name__)

PROVIDER_KEY = "euroleague"

# The API caps page size; requesting more is silently truncated, which would
# cause pagination to stall rather than fail loudly.
_PAGE_SIZE = 100

# Retrying a 4xx is pointless -- the request is wrong and will stay wrong.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _RetryableResponse(Exception):  # noqa: N818 - a signal, not an error to report
    """Internal marker so tenacity retries transient HTTP statuses."""


class _TooManyRequests(_RetryableResponse):
    """HTTP 429. Retryable, but only after backing off properly.

    Kept distinct from other transient statuses because the correct response is
    different: a 503 usually clears on its own, whereas a 429 means *we* are the
    problem and must slow down for the rest of the run.
    """

    def __init__(self, path: str, retry_after: float | None) -> None:
        self.retry_after = retry_after
        super().__init__(f"429 for {path}")


class RateLimitCooldown(ProviderError):  # noqa: N818 - established name in logs and callers
    """The provider has imposed a cooldown longer than we are willing to wait.

    Deliberately **not** a subclass of :class:`_RetryableResponse`, so tenacity
    does not retry it. When the upstream asks for five minutes, retrying inside
    that window is worse than useless: the requests are refused anyway and many
    limiters extend the penalty for traffic received during a cooldown.

    Because box score ingestion is resumable, the right response is to stop the
    run cleanly with progress committed and let the caller come back later --
    not to block a terminal for half an hour.
    """

    def __init__(self, provider: str, retry_after: float) -> None:
        self.retry_after = retry_after
        minutes = retry_after / 60.0
        super().__init__(
            provider,
            f"rate limit cooldown of {retry_after:.0f}s ({minutes:.1f} min) requested. "
            "Progress is saved; re-run to resume, or pass --wait to sit it out.",
        )


class _RateLimiter:
    """Minimum-interval rate limiter that backs off when told to.

    A full token bucket is unnecessary here: the goal is politeness towards a
    free public API, not precise throughput shaping.

    The adaptive part matters. A fixed rate that works for one endpoint can still
    trip a 429 on another, and without :meth:`slow_down` the client would keep
    hammering at the same cadence, exhaust its retries, and fail the run.
    """

    #: Multiplier applied to the interval each time the server pushes back.
    BACKOFF_FACTOR = 2.0
    #: Never throttle below one request every this many seconds.
    MAX_INTERVAL = 5.0

    def __init__(self, rate_per_second: float) -> None:
        self._base_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._min_interval = self._base_interval
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

    def slow_down(self) -> float:
        """Halve the request rate. Returns the new interval, in seconds.

        Deliberately sticky: the slower rate persists for the remainder of the
        run rather than decaying back. Creeping back up to a rate we already know
        is too fast just produces another 429 a few hundred requests later.
        """
        if self._min_interval <= 0:
            self._min_interval = 0.5
        else:
            self._min_interval = min(self._min_interval * self.BACKOFF_FACTOR, self.MAX_INTERVAL)
        return self._min_interval

    @property
    def current_interval(self) -> float:
        return self._min_interval


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header, which may be seconds or an HTTP date."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    from datetime import UTC, datetime

    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    return max(0.0, (target - datetime.now(UTC)).total_seconds())


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
        except Exception as exc:
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
        except Exception as exc:
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
        wait_out_cooldowns: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        # When True, sit through arbitrarily long cooldowns instead of aborting.
        # Right for an unattended overnight backfill, wrong for an interactive
        # run where blocking a terminal for half an hour is not acceptable.
        self._wait_out_cooldowns = wait_out_cooldowns
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
        return f"/competitions/{season.competition_code}/seasons/{season.season_code}{suffix}"

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, *, fresh: bool = False
    ) -> Any:
        """Fetch and decode JSON, with caching, rate limiting and retries.

        Exhausted retries surface as :class:`ProviderError`, never as an internal
        marker exception. Callers get one exception type to handle, and the
        retry mechanism stays an implementation detail of this module.

        ``fresh`` skips the cache both ways: for data that changes by the minute
        before tip-off, a cached copy is the wrong answer.
        """
        cache_key = f"provider:{self.provider_key}:{path}:{sorted((params or {}).items())}"

        if (
            not fresh
            and self._cache is not None
            and (cached := await self._cache.get(cache_key)) is not None
        ):
            return cached

        try:
            payload = await self._get_uncached(path, params)
        except RateLimitCooldown:
            raise
        except _TooManyRequests as exc:
            raise ProviderError(
                self.provider_key,
                f"rate limited after {self._settings.euroleague_max_retries} retries "
                f"on {path}; current throttle is "
                f"{self._limiter.current_interval:.2f}s between requests",
            ) from exc
        except _RetryableResponse as exc:
            raise ProviderError(self.provider_key, f"{exc} (retries exhausted)") from exc
        except httpx.TransportError as exc:
            raise ProviderError(self.provider_key, f"transport failure on {path}: {exc}") from exc

        if self._cache is not None and not fresh:
            await self._cache.set(cache_key, payload)
        return payload

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, _RetryableResponse)),
        # Five attempts with a 2s..60s exponential wait. The previous settings
        # (three attempts, 0.5s..8s) burned every retry inside two seconds, which
        # is nowhere near long enough for a rate limiter to release.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.0, min=2.0, max=60.0),
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

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            # A 429 means we are the problem, so slow down permanently for the
            # rest of this run rather than retrying at the same cadence.
            interval = self._limiter.slow_down()
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            logger.warning(
                "rate limited by provider; throttling",
                extra={
                    "path": path,
                    "retry_after": retry_after,
                    "new_interval_seconds": round(interval, 2),
                },
            )

            max_wait = self._settings.euroleague_max_retry_wait_seconds
            if retry_after is not None and retry_after > max_wait and not self._wait_out_cooldowns:
                # Long cooldown. Retrying inside the window is refused anyway and
                # many limiters extend the penalty for traffic during it, so stop
                # rather than making things worse.
                raise RateLimitCooldown(self.provider_key, retry_after)

            if retry_after:
                await asyncio.sleep(
                    retry_after if self._wait_out_cooldowns else min(retry_after, max_wait)
                )
            raise _TooManyRequests(path, retry_after)

        if response.status_code in _RETRYABLE_STATUS:
            logger.warning(
                "provider returned retryable status",
                extra={"path": path, "status": response.status_code},
            )
            raise _RetryableResponse(f"{response.status_code} for {path}")

        if response.status_code == httpx.codes.NOT_FOUND:
            return None

        if response.is_error:
            raise ProviderError(self.provider_key, f"HTTP {response.status_code} for {path}")

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
        """Fetch every club's roster, club by club.

        **Not the season-wide ``/people`` endpoint.** That returns a partial feed
        of recently-created registrations -- 204 rows against 20 full squads for
        E2026 -- and is not a roster. Building team membership on it produced
        players assigned to clubs they had left. ``/clubs/{code}/people`` is what
        the official site renders and returns the actual squad.

        One request per club, so twenty for a EuroLeague season. That is more
        requests than the single season-wide call, but correctness is not
        negotiable and the rate limiter handles the volume.
        """
        # A season runs 1 July to 30 June. Used to clamp the generic contract
        # windows the club endpoint returns, which otherwise span two seasons.
        season_start = date(season.start_year, 7, 1)
        season_end = date(season.start_year + 1, 6, 30)

        teams = await self.fetch_teams(season)
        stints: list[PlayerStintDTO] = []
        raw_count = 0

        for team in teams:
            rows = await self._get(self._season_path(season, f"/clubs/{team.code}/people"))
            # This endpoint returns a bare array, not the usual {"data": [...]}.
            if isinstance(rows, dict):
                rows = rows.get("data")
            if not isinstance(rows, list):
                logger.warning("no roster returned", extra={"club": team.code})
                continue

            raw_count += len(rows)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                stint = map_club_roster_entry(
                    row,
                    club_code=team.code,
                    season_start=season_start,
                    season_end=season_end,
                )
                if stint is not None:
                    stints.append(stint)

        logger.info(
            "fetched club rosters",
            extra={
                "season": season.season_code,
                "clubs": len(teams),
                "players": len(stints),
                "raw": raw_count,
            },
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

    async def fetch_game_stats_payload(
        self, season: SeasonRef, game_code: int
    ) -> dict[str, Any] | None:
        """The raw, uncached ``/games/{code}/stats`` response, for the lineup probe."""
        payload = await self._get(
            self._season_path(season, f"/games/{game_code}/stats"), fresh=True
        )
        return payload if isinstance(payload, dict) else None

    async def fetch_game_stats(self, season: SeasonRef, game_code: int) -> GameStatsDTO | None:
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
    except Exception as exc:
        logger.warning("redis unavailable, provider cache disabled", extra={"error": str(exc)})
        return None
    return ResponseCache(client, resolved.provider_cache_ttl_seconds)

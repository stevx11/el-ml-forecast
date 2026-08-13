"""Application configuration.

All configuration enters the application here and nowhere else. There are no
module-level reads of ``os.environ`` anywhere in this codebase, which means the
full set of knobs is discoverable by reading this one file and ``.env.example``.

Settings are cached, so :func:`get_settings` is cheap to call from a FastAPI
dependency on every request.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from euroleague_ai.core.errors import ConfigurationError


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Typed application configuration, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application -------------------------------------------------------
    app_name: str = "EuroLeague AI"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # -- Logging -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # -- Database ----------------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://euroleague:euroleague@localhost:5432/euroleague"),
        description="Async SQLAlchemy DSN. Must use the postgresql+asyncpg driver.",
    )
    database_echo: bool = False
    database_pool_size: Annotated[int, Field(ge=1, le=100)] = 5
    database_max_overflow: Annotated[int, Field(ge=0, le=100)] = 10

    test_database_url: PostgresDsn | None = Field(
        default=None,
        description=(
            "Database used by integration tests. When unset, database-backed tests "
            "skip rather than fail, so the unit suite runs on a laptop with no Docker."
        ),
    )

    # -- Redis -------------------------------------------------------------
    redis_url: RedisDsn = Field(default=RedisDsn("redis://localhost:6379/0"))
    provider_cache_ttl_seconds: Annotated[int, Field(ge=0)] = 900

    # -- EuroLeague provider ----------------------------------------------
    euroleague_api_base_url: str = "https://api-live.euroleague.net/v2"
    euroleague_competition_code: str = "E"
    euroleague_default_season: str = "E2025"
    euroleague_request_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0
    # Conservative on purpose. 5 req/s across 4 concurrent workers trips the
    # upstream rate limiter roughly 20 box scores into a season backfill.
    # The client also throttles itself further on any 429, so these are a
    # starting point rather than a hard ceiling.
    euroleague_max_concurrency: Annotated[int, Field(ge=1, le=32)] = 2
    euroleague_request_rate_per_second: Annotated[float, Field(gt=0)] = 2.0
    euroleague_max_retries: Annotated[int, Field(ge=0, le=10)] = 5
    # Longest Retry-After we will sit through before giving up and asking the
    # caller to resume later. Ingestion is resumable, so blocking a terminal
    # for a five-minute cooldown is the wrong trade by default.
    euroleague_max_retry_wait_seconds: Annotated[float, Field(ge=0)] = 60.0
    euroleague_user_agent: str = "euroleague-ai/0.1"

    data_provider: Literal["euroleague", "fixture"] = "euroleague"

    # -- Prediction policy -------------------------------------------------
    prediction_lead_minutes: Annotated[int, Field(ge=0, le=10_080)] = Field(
        default=60,
        description=(
            "Default cutoff, in minutes before tip-off, at which features are frozen "
            "and predictions are made. Training rows use the same offset so that "
            "offline and online information sets match."
        ),
    )

    # -- News --------------------------------------------------------------
    #
    # NoDecode is load-bearing. Without it, pydantic-settings sees a list-typed
    # field, assumes the environment holds JSON, and calls json.loads() on the
    # raw string *before* any validator runs -- so `NEWS_FEED_URLS=` (the
    # default in .env.example) fails with "Expecting value: line 1 column 1".
    # NoDecode hands the raw string to the validator below instead, which is
    # what lets a plain comma-separated list work.
    news_feed_urls: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="RSS/Atom feeds you are permitted to poll. Empty by default.",
    )
    news_fetch_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0

    @field_validator("news_feed_urls", mode="before")
    @classmethod
    def _split_feed_urls(cls, value: object) -> object:
        """Accept a comma-separated string, since env vars cannot hold lists.

        Also tolerates a JSON array, so both of these work::

            NEWS_FEED_URLS=https://a/rss, https://b/rss
            NEWS_FEED_URLS=["https://a/rss", "https://b/rss"]
        """
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    decoded = json.loads(text)
                except ValueError as exc:
                    raise ValueError(
                        "NEWS_FEED_URLS looks like JSON but could not be parsed. "
                        "Use a comma-separated list instead."
                    ) from exc
                return decoded
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("euroleague_api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def _validate_driver(self) -> Settings:
        """Reject a sync DSN early rather than failing at first query."""
        scheme = self.database_url.scheme
        if scheme != "postgresql+asyncpg":
            raise ConfigurationError(
                f"DATABASE_URL must use the postgresql+asyncpg driver, got {scheme!r}. "
                "The application uses async SQLAlchemy throughout."
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def database_dsn(self) -> str:
        """The database DSN as a plain string, for SQLAlchemy."""
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached because FastAPI resolves this dependency on every request and
    re-reading ``.env`` each time would be wasteful. Call
    ``get_settings.cache_clear()`` in tests that need to vary configuration.
    """
    return Settings()

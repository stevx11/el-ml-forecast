"""Configuration validation and time handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from euroleague_ai.core.clock import FixedClock, SystemClock, ensure_utc, utcnow
from euroleague_ai.core.config import Environment, Settings
from euroleague_ai.core.errors import ConfigurationError


class TestSettings:
    def test_rejects_a_sync_driver(self) -> None:
        """A sync DSN must fail at startup, not at the first query."""
        with pytest.raises((ConfigurationError, ValidationError)):
            Settings(
                _env_file=None,
                database_url="postgresql://user:pass@localhost:5432/db",  # type: ignore[arg-type]
            )

    def test_accepts_the_async_driver(self) -> None:
        settings = Settings(
            _env_file=None,
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",  # type: ignore[arg-type]
        )
        assert settings.database_dsn.startswith("postgresql+asyncpg://")

    def test_feed_urls_split_from_a_comma_separated_string(self) -> None:
        """Environment variables cannot hold lists."""
        settings = Settings(_env_file=None, news_feed_urls="https://a/rss, https://b/rss")  # type: ignore[arg-type]
        assert settings.news_feed_urls == ["https://a/rss", "https://b/rss"]

    def test_empty_feed_urls_is_an_empty_list(self) -> None:
        assert Settings(_env_file=None, news_feed_urls="").news_feed_urls == []  # type: ignore[arg-type]

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        settings = Settings(_env_file=None, euroleague_api_base_url="https://api/v2/")
        assert settings.euroleague_api_base_url == "https://api/v2"

    def test_prediction_lead_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, prediction_lead_minutes=-1)

    def test_is_production(self) -> None:
        assert Settings(_env_file=None, environment=Environment.PRODUCTION).is_production


class TestClock:
    def test_utcnow_is_aware(self) -> None:
        assert utcnow().tzinfo is not None

    def test_system_clock_is_aware(self) -> None:
        assert SystemClock().now().tzinfo is not None

    def test_fixed_clock_does_not_move(self) -> None:
        instant = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)
        clock = FixedClock(instant)
        assert clock.now() == clock.now() == instant

    def test_ensure_utc_rejects_naive(self) -> None:
        """Naive timestamps are rejected, never assumed.

        Guessing a timezone in a system whose correctness depends on time
        ordering is how a leak gets introduced silently.
        """
        with pytest.raises(ValueError, match="timezone-aware"):
            ensure_utc(datetime(2025, 1, 1))

    def test_ensure_utc_converts_other_offsets(self) -> None:
        belgrade = datetime(2025, 1, 1, 13, 0, tzinfo=timezone(timedelta(hours=1)))
        assert ensure_utc(belgrade) == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)

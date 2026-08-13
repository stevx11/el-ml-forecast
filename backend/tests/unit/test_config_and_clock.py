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

    def test_json_array_is_also_accepted(self) -> None:
        settings = Settings(_env_file=None, news_feed_urls='["https://a/rss"]')  # type: ignore[arg-type]
        assert settings.news_feed_urls == ["https://a/rss"]

    @pytest.mark.parametrize("raw", ["", "   ", "https://a/rss", "https://a/rss,https://b/rss"])
    def test_feed_urls_load_from_the_environment(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an empty NEWS_FEED_URLS must not crash startup.

        pydantic-settings JSON-decodes list-typed fields *before* any validator
        runs, so a bare ``NEWS_FEED_URLS=`` -- which is the default shipped in
        .env.example -- died with "Expecting value: line 1 column 1" and took
        down every entrypoint, including `alembic upgrade`. The NoDecode
        annotation on the field is what prevents that.

        This test goes through the environment rather than constructor kwargs on
        purpose: the bug only manifested in the env source, so a kwargs-based
        test would have passed while the application was broken.
        """
        monkeypatch.setenv("NEWS_FEED_URLS", raw)
        settings = Settings(_env_file=None)
        expected = [item.strip() for item in raw.split(",") if item.strip()]
        assert settings.news_feed_urls == expected

    def test_unset_feed_urls_defaults_to_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEWS_FEED_URLS", raising=False)
        assert Settings(_env_file=None).news_feed_urls == []

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


class TestLoggingExtras:
    """Regression: a colliding ``extra`` key crashed a completed training run.

    ``logger.info("registered model", extra={"name": name})`` raises
    ``KeyError: "Attempt to overwrite 'name' in LogRecord"``. The model had
    already trained and passed its gate; the crash happened while reporting it,
    and rolled the registration back.
    """

    def test_reserved_key_does_not_raise(self) -> None:
        import logging as stdlib_logging

        from euroleague_ai.core.logging import get_logger

        logger = get_logger("test.extras")
        # Would raise KeyError without SafeExtraLogger.
        logger.info("registered model", extra={"name": "minutes_lgbm", "version": "0.1.0"})
        assert isinstance(logger.logger, stdlib_logging.Logger)

    @pytest.mark.parametrize(
        "key", ["name", "module", "filename", "args", "lineno", "message", "asctime"]
    )
    def test_every_reserved_key_is_survivable(self, key: str) -> None:
        from euroleague_ai.core.logging import get_logger

        get_logger("test.extras").warning("reserved", extra={key: "value"})

    def test_colliding_values_are_kept_not_dropped(self) -> None:
        """Suffixed rather than discarded -- the value still reaches the log."""
        import io
        import json
        import logging as stdlib_logging

        from euroleague_ai.core.logging import JsonFormatter, get_logger

        buffer = io.StringIO()
        handler = stdlib_logging.StreamHandler(buffer)
        handler.setFormatter(JsonFormatter())
        root = stdlib_logging.getLogger()
        original = root.handlers[:]
        root.handlers = [handler]
        root.setLevel(stdlib_logging.INFO)
        try:
            get_logger("test.extras").info("m", extra={"name": "minutes_lgbm"})
            payload = json.loads(buffer.getvalue().strip())
        finally:
            root.handlers = original

        assert payload["name_"] == "minutes_lgbm"
        assert payload["logger"] == "test.extras"

"""Cross-cutting concerns: configuration, logging, time, errors.

This package sits at the bottom of the dependency graph. It may be imported by
anything and must import nothing from the rest of the application.
"""

from euroleague_ai.core.clock import Clock, SystemClock, utcnow
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import (
    ConfigurationError,
    EuroLeagueAIError,
    IngestionError,
    LeakageError,
    NotFoundError,
    ProviderError,
)
from euroleague_ai.core.logging import configure_logging, get_logger

__all__ = [
    "Clock",
    "ConfigurationError",
    "EuroLeagueAIError",
    "IngestionError",
    "LeakageError",
    "NotFoundError",
    "ProviderError",
    "Settings",
    "SystemClock",
    "configure_logging",
    "get_logger",
    "get_settings",
    "utcnow",
]

"""Application exception hierarchy.

Every error raised deliberately by this codebase derives from
:class:`EuroLeagueAIError`, so callers can distinguish our failures from
library failures without catching bare ``Exception``.
"""

from __future__ import annotations


class EuroLeagueAIError(Exception):
    """Base class for every error this application raises deliberately."""


class ConfigurationError(EuroLeagueAIError):
    """Configuration is missing, malformed, or internally inconsistent."""


class NotFoundError(EuroLeagueAIError):
    """A requested entity does not exist."""

    def __init__(self, entity: str, identifier: object) -> None:
        self.entity = entity
        self.identifier = identifier
        super().__init__(f"{entity} not found: {identifier!r}")


class ProviderError(EuroLeagueAIError):
    """An external data provider failed or returned something unusable."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class IngestionError(EuroLeagueAIError):
    """An ingestion job could not complete."""


class LeakageError(EuroLeagueAIError):
    """A feature computation attempted to read data from after its cutoff.

    This is always a programming error. It is raised rather than logged because
    a leaked feature silently corrupts every model trained on it, and a loud
    failure during development is far cheaper than a quiet one in production.
    """

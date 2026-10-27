"""Provider selection.

The single place that turns the ``DATA_PROVIDER`` setting into an object. Keeps
``if provider == "fixture"`` out of ingestion, jobs, and tests.
"""

from __future__ import annotations

from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import ConfigurationError
from euroleague_ai.providers.base import BasketballDataProvider, SeasonRef
from euroleague_ai.providers.euroleague import EuroLeagueProvider, build_response_cache
from euroleague_ai.providers.fixture import FixtureProvider


def build_provider(
    settings: Settings | None = None, *, wait_out_cooldowns: bool = False
) -> BasketballDataProvider:
    """Return the configured provider.

    Args:
        wait_out_cooldowns: Sit through long rate-limit cooldowns rather than
            stopping. Only meaningful for the live provider.
    """
    resolved = settings or get_settings()
    match resolved.data_provider:
        case "euroleague":
            return EuroLeagueProvider(
                resolved,
                cache=build_response_cache(resolved),
                wait_out_cooldowns=wait_out_cooldowns,
            )
        case "fixture":
            return FixtureProvider()
        case unknown:  # pragma: no cover - guarded by the Literal type
            raise ConfigurationError(f"unknown DATA_PROVIDER: {unknown!r}")


def season_ref(season_code: str, settings: Settings | None = None) -> SeasonRef:
    """Build a :class:`SeasonRef` from a season code such as ``"E2025"``.

    The trailing four digits are the starting year, which is how the upstream API
    encodes it (``E2025`` is the 2025-26 season).
    """
    resolved = settings or get_settings()
    digits = "".join(ch for ch in season_code if ch.isdigit())
    if len(digits) < 4:
        raise ConfigurationError(f"season code {season_code!r} does not contain a four-digit year")
    return SeasonRef(
        competition_code=resolved.euroleague_competition_code,
        season_code=season_code,
        start_year=int(digits[-4:]),
    )

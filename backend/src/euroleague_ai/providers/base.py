"""Provider protocol and provider-neutral DTOs.

Domain code never sees provider JSON. Adapters translate into the DTOs below,
and everything downstream -- ingestion, features, models -- is written against
these shapes. Swapping EuroLeague for Sportradar, or adding ACB, therefore
touches one file.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from euroleague_ai.models.enums import GameStatus, PlayerPosition


class ProviderModel(BaseModel):
    """Base for provider DTOs. Immutable, so a mapped value cannot be edited in place."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class SeasonRef(ProviderModel):
    """Identifies a season to a provider."""

    competition_code: str = Field(description='Provider competition code, e.g. "E".')
    season_code: str = Field(description='Provider season code, e.g. "E2025".')
    start_year: int


class TeamDTO(ProviderModel):
    external_id: str
    code: str
    name: str
    short_name: str | None = None
    country_code: str | None = None
    city: str | None = None
    venue_name: str | None = None


class PlayerStintDTO(ProviderModel):
    """A player plus the registration interval they were observed in.

    Player biography and team registration arrive together from the upstream
    ``/people`` endpoint, and are stored into two tables -- ``players`` for the
    immutable parts and ``player_team_stints`` for the interval.
    """

    external_id: str
    full_name: str
    last_name: str | None = None
    first_name: str | None = None
    birth_date: date | None = None
    country_code: str | None = None
    birth_country_code: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    position: PlayerPosition = PlayerPosition.UNKNOWN

    team_external_id: str | None = None
    jersey_number: str | None = None
    previous_team_name: str | None = None
    is_active: bool = True
    effective_from: datetime
    effective_until: datetime | None = None


class GameDTO(ProviderModel):
    external_id: str
    game_code: int
    round_number: int | None = None
    phase_code: str | None = None
    phase_name: str | None = None
    group_name: str | None = None

    tip_off_utc: datetime
    tip_off_local: datetime | None = None
    local_timezone: str | None = None

    status: GameStatus = GameStatus.SCHEDULED
    home_team_external_id: str
    away_team_external_id: str
    home_score: int | None = None
    away_score: int | None = None
    home_quarter_scores: dict[str, int] | None = None
    away_quarter_scores: dict[str, int] | None = None

    is_neutral_venue: bool = False
    venue_name: str | None = None
    attendance: int | None = None


class PlayerLineDTO(ProviderModel):
    """One player's box score line."""

    player_external_id: str
    player_name: str
    jersey_number: str | None = None
    seconds_played: int = 0
    is_starter: bool = False
    did_not_play: bool = False

    points: int = 0
    fg2m: int = 0
    fg2a: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    total_rebounds: int = 0
    assists: int = 0
    steals: int = 0
    turnovers: int = 0
    blocks_favour: int = 0
    blocks_against: int = 0
    fouls_committed: int = 0
    fouls_received: int = 0
    valuation: int = 0
    plus_minus: int | None = None


class TeamLineDTO(ProviderModel):
    """Team totals for one side of a game."""

    points: int = 0
    fg2m: int = 0
    fg2a: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    total_rebounds: int = 0
    assists: int = 0
    steals: int = 0
    turnovers: int = 0
    blocks_favour: int = 0
    blocks_against: int = 0
    fouls_committed: int = 0
    fouls_received: int = 0
    valuation: int = 0


class TeamStatsDTO(ProviderModel):
    """Everything known about one side of one game."""

    team_external_id: str | None = None
    coach_external_id: str | None = None
    coach_name: str | None = None
    players: tuple[PlayerLineDTO, ...] = ()
    totals: TeamLineDTO = TeamLineDTO()


class GameStatsDTO(ProviderModel):
    """Box score for a completed game."""

    game_code: int
    home: TeamStatsDTO
    away: TeamStatsDTO


@runtime_checkable
class BasketballDataProvider(Protocol):
    """Source of basketball data.

    Two implementations ship: :class:`~euroleague_ai.providers.euroleague.EuroLeagueProvider`
    against the live public API, and
    :class:`~euroleague_ai.providers.fixture.FixtureProvider` against committed
    JSON. Because both satisfy this protocol, the entire ingestion pipeline is
    tested end to end with no network access.
    """

    provider_key: str

    async def fetch_teams(self, season: SeasonRef) -> list[TeamDTO]:
        """Return every team registered in the season."""
        ...

    async def fetch_players(self, season: SeasonRef) -> list[PlayerStintDTO]:
        """Return every player registration observed in the season."""
        ...

    async def fetch_games(
        self, season: SeasonRef, *, round_number: int | None = None
    ) -> list[GameDTO]:
        """Return games, optionally restricted to one round."""
        ...

    async def fetch_game_stats(self, season: SeasonRef, game_code: int) -> GameStatsDTO | None:
        """Return the box score for a game, or ``None`` if it has not been played."""
        ...

    async def aclose(self) -> None:
        """Release any held resources."""
        ...

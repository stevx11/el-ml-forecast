"""Game response models."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from euroleague_ai.models.enums import GameStatus, TeamSide
from euroleague_ai.schemas.competition import TeamOut


class GameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    season_id: uuid.UUID
    game_code: int
    round_number: int | None = None
    phase_name: str | None = None
    tip_off_utc: datetime
    local_timezone: str | None = None
    status: GameStatus
    home_team: TeamOut
    away_team: TeamOut
    home_score: int | None = None
    away_score: int | None = None
    venue_name: str | None = None
    is_neutral_venue: bool
    attendance: int | None = None


class BoxScoreLineOut(BaseModel):
    """One player's line in a game's box score."""

    player_id: uuid.UUID
    player_name: str
    side: TeamSide
    seconds_played: int
    is_starter: bool
    did_not_play: bool
    points: int
    fg2m: int
    fg2a: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    offensive_rebounds: int
    defensive_rebounds: int
    total_rebounds: int
    assists: int
    steals: int
    turnovers: int
    blocks_favour: int
    fouls_committed: int
    valuation: int
    plus_minus: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minutes(self) -> float:
        return round(self.seconds_played / 60.0, 2)


class GameDetailOut(GameOut):
    """A game plus its box score, when one exists."""

    box_score: list[BoxScoreLineOut] = []

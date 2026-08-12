"""Player response models."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, computed_field

from euroleague_ai.models.enums import PlayerPosition


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    full_name: str
    first_name: str | None = None
    last_name: str | None = None
    birth_date: date | None = None
    country_code: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    position: PlayerPosition


class PlayerGameLogOut(BaseModel):
    """One row of a player's game log."""

    model_config = ConfigDict(from_attributes=True)

    game_id: uuid.UUID
    tip_off_utc: datetime
    seconds_played: int
    did_not_play: bool
    is_starter: bool
    points: int
    total_rebounds: int
    assists: int
    fg3m: int
    steals: int
    blocks_favour: int
    turnovers: int
    valuation: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minutes(self) -> float:
        """Seconds rendered as minutes. Storage keeps seconds; display uses this."""
        return round(self.seconds_played / 60.0, 2)

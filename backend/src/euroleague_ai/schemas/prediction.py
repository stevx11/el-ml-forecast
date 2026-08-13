"""Prediction response models."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from euroleague_ai.models.enums import PredictionTarget


class PredictionOut(BaseModel):
    """One model's projection of one target for one player-game."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    player_id: uuid.UUID
    player_name: str
    game_id: uuid.UUID
    target: PredictionTarget
    predicted_value: float
    prediction_timestamp: datetime
    model: str = Field(description="Model name and version that produced this.")


class PlayerProjectionOut(BaseModel):
    """A player's full projected line, one target per key."""

    model_config = ConfigDict(protected_namespaces=())

    player_id: uuid.UUID
    player_name: str
    prediction_timestamp: datetime
    targets: dict[str, float]
    models: dict[str, str]


class ProjectionsOut(BaseModel):
    """Every projected line for one game."""

    game_id: uuid.UUID
    tip_off_utc: datetime
    players: list[PlayerProjectionOut]

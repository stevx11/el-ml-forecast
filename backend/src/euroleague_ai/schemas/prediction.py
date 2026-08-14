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


class DriverOut(BaseModel):
    """One readable reason behind a projection."""

    theme: str
    direction: str
    label: str
    impact: float


class TargetProjectionOut(BaseModel):
    """One target for one player: the number, its spread, and why.

    Everything a reader needs to judge a projection travels together. A point
    estimate on its own invites more confidence than it has earned.
    """

    value: float
    model: str
    #: Predicted quantiles, e.g. ``{"0.1": 8.0, "0.5": 14.0, "0.9": 21.0}``.
    quantiles: dict[str, float] | None = None
    #: ``{"15": 0.42}`` -- P(actual >= 15).
    threshold_probabilities: dict[str, float] | None = None
    #: Robust standard deviation of the predicted distribution.
    predicted_std: float | None = None
    #: Relative tightness in (0, 1]. Null for models trained before item 06.
    confidence: float | None = None
    drivers: list[DriverOut] = Field(default_factory=list)


class PlayerProjectionOut(BaseModel):
    """A player's full projected line, one target per key."""

    model_config = ConfigDict(protected_namespaces=())

    player_id: uuid.UUID
    player_name: str
    prediction_timestamp: datetime
    #: Team the player was resolved to for this fixture, so the UI can split a
    #: game into two squads without a second round trip.
    team_id: uuid.UUID | None = None
    team_code: str | None = None
    #: Kept flat for convenience and backwards compatibility.
    targets: dict[str, float]
    models: dict[str, str]
    #: The richer per-target payload: spread, probabilities, drivers.
    projections: dict[str, TargetProjectionOut] = Field(default_factory=dict)

    @property
    def minutes(self) -> float:
        return self.targets.get("MINUTES", 0.0)


class ProjectionsOut(BaseModel):
    """Every projected line for one game."""

    game_id: uuid.UUID
    tip_off_utc: datetime
    home_team: str | None = None
    away_team: str | None = None
    home_team_code: str | None = None
    away_team_code: str | None = None
    players: list[PlayerProjectionOut]

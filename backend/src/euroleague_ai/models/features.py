"""Persisted feature snapshots.

A snapshot is the exact input a model saw, frozen at a stated cutoff. It is
written once and never updated.

Persisting rather than recomputing is what lets a prediction made in October
still be explained in April with the inputs it actually consumed -- even if the
underlying box scores have since been corrected upstream.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from euroleague_ai.db.base import Base
from euroleague_ai.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class PlayerFeatureSnapshot(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Feature vector for one player-game, computed as of a stated instant."""

    __tablename__ = "player_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "game_id",
            "feature_set_version",
            "prediction_timestamp",
            name="uq_player_feature_snapshots_key",
        ),
        Index("ix_player_feature_snapshots_game", "game_id"),
        Index("ix_player_feature_snapshots_player_ts", "player_id", "prediction_timestamp"),
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("games.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    opponent_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )

    # The cutoff. Defines precisely what this row was permitted to see: no fact
    # with observed_at greater than this may have contributed to any value below.
    prediction_timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)

    feature_set_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Pins feature names and semantics. Changing definitions means a new version.",
    )

    # JSONB rather than several hundred columns: feature sets churn constantly
    # during model development and a migration per experiment is unworkable.
    # Volume is players x games -- tens of thousands of rows per season -- so the
    # query cost is irrelevant, and feature_set_version pins the schema.
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    computed_at: Mapped[datetime] = mapped_column(nullable=False)


class TeamFeatureSnapshot(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Feature vector for one team-game, computed as of a stated instant."""

    __tablename__ = "team_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "game_id",
            "feature_set_version",
            "prediction_timestamp",
            name="uq_team_feature_snapshots_key",
        ),
        Index("ix_team_feature_snapshots_game", "game_id"),
    )

    team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("games.id"), nullable=False
    )
    opponent_team_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("teams.id"), nullable=False
    )
    prediction_timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(nullable=False)

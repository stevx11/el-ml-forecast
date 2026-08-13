"""Model registry, predictions, and post-game scoring."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ModelVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A trained, versioned model artifact."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        Index("ix_model_versions_target_active", "target", "is_active"),
    )

    name: Mapped[str] = mapped_column(String(96), nullable=False, doc='e.g. "minutes_lgbm".')
    version: Mapped[str] = mapped_column(String(32), nullable=False, doc="Semantic version.")
    target: Mapped[PredictionTarget] = mapped_column(nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)

    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    training_data_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Hash of the training row keys. Makes a training run reproducible.",
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Validation metrics, including the baseline this model was required to beat.",
    )
    trained_at: Mapped[datetime | None] = mapped_column(nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="model_version")


class Prediction(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One model's projection of one target for one player-game.

    Three columns make a prediction reproducible and explainable months later:
    ``model_version_id``, ``prediction_timestamp``, and
    ``player_feature_snapshot_id``.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "model_version_id",
            "player_id",
            "game_id",
            "target",
            "prediction_timestamp",
            name="uq_predictions_key",
        ),
        Index("ix_predictions_game_target", "game_id", "target"),
        Index("ix_predictions_player_ts", "player_id", "prediction_timestamp"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_in_unit_interval",
        ),
    )

    model_version_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False
    )
    player_feature_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        # Named explicitly. The naming convention would generate
        # "fk_predictions_player_feature_snapshot_id_player_feature_snapshots",
        # which is 66 characters and exceeds PostgreSQL's 63-byte identifier
        # limit. SQLAlchemy raises rather than silently truncating, so the
        # migration fails outright.
        ForeignKey(
            "player_feature_snapshots.id",
            name="fk_predictions_player_feature_snapshot_id",
        ),
        nullable=False,
        doc="The exact inputs this prediction consumed.",
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("games.id"), nullable=False
    )

    target: Mapped[PredictionTarget] = mapped_column(nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_std: Mapped[float | None] = mapped_column(Float, nullable=True)

    quantiles: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, doc='e.g. {"0.1": 8.0, "0.5": 18.0, "0.9": 27.0}.'
    )
    threshold_probabilities: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, doc='e.g. {"15": 0.72, "20": 0.43, "25": 0.17}.'
    )
    explanation: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Signed feature contributions plus their human-readable labels.",
    )
    confidence: Mapped[float | None] = mapped_column(
        Numeric(4, 3),
        nullable=True,
        doc=(
            "How tight this prediction is relative to a typical one, in (0, 1]. "
            "Deliberately relative -- there is no absolute scale on which a "
            "basketball projection is '90% confident'. Nullable because models "
            "trained before the distribution work have nothing to report."
        ),
    )

    prediction_timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)

    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    outcome: Mapped[PredictionOutcome | None] = relationship(
        back_populates="prediction", uselist=False
    )


class PredictionOutcome(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """The realised value, joined onto a prediction after the game.

    Written by a post-game job so that model accuracy is a SQL query rather than
    a recomputation. That is what makes continuous drift monitoring cheap enough
    to actually run.
    """

    __tablename__ = "prediction_outcomes"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_prediction_outcomes_prediction"),
    )

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False
    )
    actual_value: Mapped[float] = mapped_column(Float, nullable=False)
    absolute_error: Mapped[float] = mapped_column(Float, nullable=False)
    squared_error: Mapped[float] = mapped_column(Float, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(nullable=False)
    did_not_appear: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        doc=(
            "The player did not take the floor: either absent from the squad "
            "entirely, or dressed and never used. Scored as zero, but flagged, "
            "because it is an availability miss rather than a statistical one. "
            "Folding these into MAE would blame the points model for a coach's "
            "rotation decision; hiding them would flatter a projection we did "
            "publish. Recording both keeps either analysis available."
        ),
    )

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")

"""Audit trail for ingestion jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from euroleague_ai.db.base import Base
from euroleague_ai.models.enums import IngestionStatus
from euroleague_ai.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class IngestionRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One execution of an ingestion job.

    A failed overnight backfill should be diagnosable from a table, not from log
    archaeology across a rotated file.
    """

    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_job_started", "job_name", "started_at"),)

    job_name: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_key: Mapped[str] = mapped_column(String(64), nullable=False)
    season_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[IngestionStatus] = mapped_column(
        default=IngestionStatus.RUNNING, nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    rows_read: Mapped[int] = mapped_column(default=0, nullable=False)
    rows_written: Mapped[int] = mapped_column(default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

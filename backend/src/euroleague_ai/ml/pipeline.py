"""End-to-end training run: load, train, judge, register."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.ml.dataset import describe_rows, load_training_rows
from euroleague_ai.ml.registry import register_model
from euroleague_ai.ml.report import DEFAULT_TARGETS
from euroleague_ai.ml.train import train_target

logger = get_logger(__name__)

#: Below this many rows a chronological three-way split leaves too little to
#: train on, and the resulting metrics are noise rather than measurement.
MIN_ROWS = 2_000


async def train_and_register(
    session: AsyncSession,
    *,
    targets: tuple[PredictionTarget, ...] = DEFAULT_TARGETS,
    season_codes: list[str] | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    artifact_root: Path | None = None,
    params: dict[str, Any] | None = None,
    activate: bool | None = None,
) -> dict[str, Any]:
    """Train each target in order and register the results.

    Targets are trained in the order given -- minutes first by default, because
    every other box score quantity is conditional on playing time.
    """
    report: dict[str, Any] = {"feature_set_version": feature_set_version, "targets": {}}

    for target in targets:
        rows = await load_training_rows(
            session,
            target=target,
            feature_set_version=feature_set_version,
            season_codes=season_codes,
        )
        if len(rows) < MIN_ROWS:
            report["targets"][str(target)] = {
                "error": f"only {len(rows)} rows; need at least {MIN_ROWS}"
            }
            continue

        result = train_target(rows, target, params=params)
        model_version = await register_model(
            session,
            result,
            feature_set_version=feature_set_version,
            artifact_root=artifact_root,
            activate=activate,
        )

        report["targets"][str(target)] = {
            "dataset": describe_rows(rows),
            "summary": result.summary(),
            "by_minutes": result.by_minutes,
            "top_features": result.feature_importance[:15],
            "model": {
                "name": model_version.name,
                "version": model_version.version,
                "active": model_version.is_active,
                "artifact": model_version.artifact_path,
            },
        }

    return report

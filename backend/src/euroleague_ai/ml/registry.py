"""Model registry.

Every trained model gets a row recording what it is, what it was trained on, how
it scored, and where its artifact lives. Three things make a prediction
reproducible months later: ``model_version_id``, ``prediction_timestamp`` and
``feature_snapshot_id``. This module supplies the first.

**Activation is gated.** ``is_active`` is only set when the model beat its
baseline. A model that lost is still registered -- the negative result is worth
keeping, and silently discarding failures is how a team convinces itself
everything works -- but nothing will serve it.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.prediction import ModelVersion
from euroleague_ai.ml.train import TrainingResult, suspiciously_good

logger = get_logger(__name__)

#: Where boosters and their column order are written. Bind-mounted in Compose,
#: so artifacts survive a container rebuild.
DEFAULT_ARTIFACT_ROOT = Path("/app/data/models")


def _artifact_paths(root: Path, name: str, version: str) -> tuple[Path, Path]:
    directory = root / name / version
    return directory / "model.txt", directory / "columns.json"


def distribution_path(model_path: Path) -> Path:
    """Where the error distribution sits, beside its booster.

    Travels with the artifact for the same reason the column order does: it
    describes *this* model's errors, and pairing a new booster with an old
    distribution would produce intervals that quietly no longer hold.
    """
    return model_path.parent / "distribution.json"


def expectation_paths(model_path: Path) -> tuple[Path, Path]:
    """``(expected_model.txt, expectation.json)`` beside a booster.

    The JSON says where the expected value comes from (``booster``,
    ``shifted_median`` or ``none``). The second booster exists only when it won.
    A model directory with neither predates expected values, and inference falls
    back to the shifted median, which needs only the distribution.
    """
    return model_path.parent / "expected_model.txt", model_path.parent / "expectation.json"


def save_artifact(
    result: TrainingResult, *, name: str, version: str, root: Path | None = None
) -> Path:
    """Write the booster and its pinned column order to disk.

    The column order is saved beside the model, not derived at load time. A model
    served with columns in a different order produces plausible nonsense rather
    than an error, so the order has to travel with the artifact.
    """
    root = root or DEFAULT_ARTIFACT_ROOT
    model_path, columns_path = _artifact_paths(root, name, version)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    result.booster.save_model(str(model_path), num_iteration=result.best_iteration)
    columns_path.write_text(
        json.dumps(
            {
                "feature_names": result.feature_names,
                "target": str(result.target),
                "best_iteration": result.best_iteration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    distribution_path(model_path).write_text(
        json.dumps(result.distribution.to_dict()), encoding="utf-8"
    )
    if result.expectation is not None:
        booster_path, report_path = expectation_paths(model_path)
        report = result.expectation.summary()
        if result.expected_booster is not None:
            result.expected_booster.save_model(
                str(booster_path), num_iteration=result.expectation.booster_best_iteration
            )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path


def model_name_for(target: PredictionTarget) -> str:
    """Registry name for a target, e.g. ``minutes_lgbm``."""
    return f"{str(target).lower()}_lgbm"


async def next_version(session: AsyncSession, name: str) -> str:
    """Return the next patch version for ``name``, starting at ``0.1.0``."""
    existing = (
        await session.execute(
            select(ModelVersion.version).where(ModelVersion.name == name)
        )
    ).scalars().all()
    if not existing:
        return "0.1.0"

    def parse(value: str) -> tuple[int, int, int]:
        try:
            major, minor, patch = (int(part) for part in value.split("."))
        except ValueError:
            return (0, 0, 0)
        return (major, minor, patch)

    major, minor, patch = max(parse(value) for value in existing)
    return f"{major}.{minor}.{patch + 1}"


def _json_safe(value: Any) -> Any:
    """NaN and infinity as null: JSONB rejects them, and an empty stratum (no
    test rows in some minutes band) produces NaN metrics."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


async def register_model(
    session: AsyncSession,
    result: TrainingResult,
    *,
    feature_set_version: str,
    artifact_root: Path | None = None,
    clock: Clock | None = None,
    activate: bool | None = None,
    season_gate: dict[str, Any] | None = None,
) -> ModelVersion:
    """Persist a trained model and, if it passed the gate, activate it.

    Args:
        activate: Override the gate. ``None`` means "activate iff it beat the
            baseline", which is what you want; passing ``True`` to publish a
            losing model is possible but deliberate.
        season_gate: The walk-forward verdict from
            :func:`euroleague_ai.ml.walkforward.season_gate`. When it was able
            to judge, it must pass too; either way it is stored with the model.
    """
    clock = clock or SystemClock()
    name = model_name_for(result.target)
    version = await next_version(session, name)

    artifact_path = save_artifact(result, name=name, version=version, root=artifact_root)

    passes = result.beats_baseline
    if season_gate is not None and season_gate.get("judged"):
        passes = passes and bool(season_gate.get("passes"))
    should_activate = passes if activate is None else activate

    metrics: dict[str, Any] = {
        **result.summary(),
        "by_minutes": result.by_minutes,
        # The bar travels with the model. Without it, a future reader has a
        # number and no way to know whether it was any good.
        "baseline_comparison": {
            "name": result.baseline_name,
            "mae": round(result.baseline_metrics.mae, 4),
            "rmse": round(result.baseline_metrics.rmse, 4),
            "r2": round(result.baseline_metrics.r2, 4),
        },
        "top_features": result.feature_importance[:15],
        "seasons": result.seasons,
        "suspicious": suspiciously_good(result),
        "season_gate": season_gate,
    }

    model_version = ModelVersion(
        id=uuid.uuid4(),
        name=name,
        version=version,
        target=result.target,
        algorithm="lightgbm",
        feature_set_version=feature_set_version,
        hyperparameters={**result.params, "best_iteration": result.best_iteration},
        training_data_hash=result.training_data_hash,
        metrics=_json_safe(metrics),
        trained_at=clock.now(),
        artifact_path=str(artifact_path),
        is_active=should_activate,
    )
    session.add(model_version)
    await session.flush()

    if should_activate:
        # Exactly one active model per target, or inference has to guess.
        await session.execute(
            update(ModelVersion)
            .where(
                ModelVersion.target == result.target,
                ModelVersion.id != model_version.id,
            )
            .values(is_active=False)
        )

    logger.info(
        "registered model",
        extra={
            "model_name": name,
            "version": version,
            "active": should_activate,
            "mae": round(result.metrics.mae, 4),
            "baseline_mae": round(result.baseline_metrics.mae, 4),
        },
    )
    return model_version


async def active_model(
    session: AsyncSession, target: PredictionTarget
) -> ModelVersion | None:
    """Return the active model for a target, if one passed the gate."""
    return (
        await session.execute(
            select(ModelVersion).where(
                ModelVersion.target == target, ModelVersion.is_active.is_(True)
            )
        )
    ).scalar_one_or_none()


async def list_models(session: AsyncSession) -> list[ModelVersion]:
    """Every registered model, newest first."""
    return list(
        (
            await session.execute(
                select(ModelVersion).order_by(ModelVersion.trained_at.desc())
            )
        )
        .scalars()
        .all()
    )

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
from euroleague_ai.ml.tuning import load_params

logger = get_logger(__name__)

#: Below this many rows a chronological three-way split leaves too little to
#: train on, and the resulting metrics are noise rather than measurement.
MIN_ROWS = 2_000


def _excluded() -> frozenset[str]:
    """Features of the families listed in EXCLUDED_FEATURE_FAMILIES."""
    from euroleague_ai.core.config import get_settings
    from euroleague_ai.features.context_v6 import excluded_feature_names

    return excluded_feature_names(get_settings().excluded_feature_families)


async def train_and_register(
    session: AsyncSession,
    *,
    targets: tuple[PredictionTarget, ...] = DEFAULT_TARGETS,
    season_codes: list[str] | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    artifact_root: Path | None = None,
    params: dict[str, Any] | None = None,
    activate: bool | None = None,
    walkforward: bool = True,
    walkforward_seasons: int = 4,
    current_season: str | None = None,
) -> dict[str, Any]:
    """Train each target in order and register the results.

    Targets are trained in the order given -- minutes first by default, because
    every other box score quantity is conditional on playing time. ``PLAYS``, if
    requested, is routed to the availability trainer, whose report has a
    different shape (log loss and calibration rather than MAE).
    """
    report: dict[str, Any] = {"feature_set_version": feature_set_version, "targets": {}}

    for target in targets:
        if target is PredictionTarget.PLAYS:
            # A classifier over the roster-candidate population, with its own
            # loader, metrics and gate. See ml.availability.
            from euroleague_ai.ml.availability import train_and_register_availability

            report["targets"][str(target)] = await train_and_register_availability(
                session,
                season_codes=season_codes,
                feature_set_version=feature_set_version,
                artifact_root=artifact_root,
                activate=activate,
            )
            continue

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

        # Tuned parameters, if the search has been run for this target.
        # Explicit params passed by the caller still win, so a one-off
        # experiment does not have to fight the saved file.
        resolved_params = params
        tuned_note = "defaults"
        if resolved_params is None:
            tuned = load_params(target, root=artifact_root)
            if tuned:
                resolved_params = tuned
                tuned_note = "tuned"

        excluded = _excluded()
        if excluded:
            from euroleague_ai.ml.ablation import strip

            rows = strip(rows, excluded)
        result = train_target(rows, target, params=resolved_params)
        gate: dict[str, Any] | None = None
        if walkforward:
            # The same training, once per recent season: activation needs the
            # gain to hold season after season, not just on the one split.
            from euroleague_ai.ml.walkforward import evaluate_target, season_gate

            folds = evaluate_target(
                rows,
                target,
                seasons=walkforward_seasons,
                current_season=current_season,
                params=resolved_params,
            )
            gate = season_gate(folds)
        model_version = await register_model(
            session,
            result,
            feature_set_version=feature_set_version,
            artifact_root=artifact_root,
            activate=activate,
            season_gate=gate,
        )

        report["targets"][str(target)] = {
            "dataset": describe_rows(rows),
            "summary": result.summary(),
            "by_minutes": result.by_minutes,
            "top_features": result.feature_importance[:15],
            "params_source": tuned_note,
            "season_gate": gate,
            "model": {
                "name": model_version.name,
                "version": model_version.version,
                "active": model_version.is_active,
                "artifact": model_version.artifact_path,
            },
        }

    return report

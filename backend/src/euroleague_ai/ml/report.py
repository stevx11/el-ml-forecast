"""Baseline comparison run.

Produces the table Phase 3 has to beat. Publishing it -- and keeping it in the
model registry alongside every trained model -- is what stops the classic
self-deception of comparing a tuned model against a deliberately weak straw man.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.ml.baselines import default_baselines
from euroleague_ai.ml.dataset import describe_rows, load_training_rows
from euroleague_ai.ml.evaluation import backtest, compare

logger = get_logger(__name__)

#: The five targets the roadmap starts with. Minutes first, because every other
#: box score quantity is conditional on playing time.
DEFAULT_TARGETS: tuple[PredictionTarget, ...] = (
    PredictionTarget.MINUTES,
    PredictionTarget.POINTS,
    PredictionTarget.REBOUNDS,
    PredictionTarget.ASSISTS,
    PredictionTarget.THREE_POINTERS_MADE,
)


async def run_baseline_report(
    session: AsyncSession,
    *,
    targets: tuple[PredictionTarget, ...] = DEFAULT_TARGETS,
    season_codes: list[str] | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    min_train_rounds: int = 5,
) -> dict[str, Any]:
    """Backtest every baseline against every target."""
    report: dict[str, Any] = {
        "feature_set_version": feature_set_version,
        "targets": {},
    }

    for target in targets:
        rows = await load_training_rows(
            session,
            target=target,
            feature_set_version=feature_set_version,
            season_codes=season_codes,
        )
        if not rows:
            report["targets"][str(target)] = {"error": "no training rows"}
            continue

        results = {
            baseline.name: backtest(
                rows,
                lambda row, b=baseline, t=target: b.predict(row, t),  # type: ignore[misc]
                min_train_rounds=min_train_rounds,
            )
            for baseline in default_baselines()
        }

        table = compare(results)
        report["targets"][str(target)] = {
            "dataset": describe_rows(rows),
            "ranking": table,
            "best": table[0]["predictor"] if table else None,
            "bar_to_beat": table[0]["mae"] if table else None,
            "detail": results,
        }
        logger.info(
            "baseline backtest complete",
            extra={
                "target": str(target),
                "best": table[0]["predictor"] if table else None,
                "mae": table[0]["mae"] if table else None,
            },
        )

    return report

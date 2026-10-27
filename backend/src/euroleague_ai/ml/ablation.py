"""Which v6 feature families earn their place? Train without each one and compare.

For every target, the same training is repeated on the same rows and split:

* **all v6**: every feature;
* **v5 only**: every v6 family removed, the model as it was;
* **without <family>**: one v6 family removed at a time.

A family helps if removing it makes the model worse: test MAE up for a stat,
test log loss up for P(plays), by more than retraining noise. Both models sample
70% of features and 80% of rows per tree, so any change to the feature list
grows different trees, and a family of pure noise can "move" the score by a few
tenths of a percent. The noise is measured on every run: the full model is
trained again with :data:`RESEEDS` other sets of random seeds, and a family only
counts as helping if removing it costs more than twice the largest of those
differences (and at least :data:`HELPS_THRESHOLD`). One reseed is not enough: on
a synthetic league it measured 0.15% while a pure-noise feature moved the score
0.33%. Whether a gain holds next season is a separate
question, for the walk-forward gate on the retrained models.

Families that help no target are candidates for removal before v6 is shipped:
a feature that adds nothing still costs backfill time and adds noise for the
next change to fight against.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.context_v6 import V6_FAMILIES, V6_FEATURE_NAMES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from datetime import datetime

    from euroleague_ai.models.enums import PredictionTarget

logger = get_logger(__name__)

REPORTS_DIR = Path("reports") / "ablation"

#: The floor for "helps", whatever the measured noise: a family has to be worth
#: at least a tenth of a percent.
HELPS_THRESHOLD = 0.001

#: Extra trainings of the full model, with other seeds, to measure noise.
RESEEDS = 3


def _reseeded(params: dict[str, Any] | None, index: int) -> dict[str, Any]:
    from euroleague_ai.ml.train import TRAINING_SEED

    seed = TRAINING_SEED + 1_000 * (index + 1)
    return {
        **(params or {}),
        "seed": seed,
        "bagging_seed": seed + 1,
        "feature_fraction_seed": seed + 2,
        "data_random_seed": seed + 3,
    }


def variants() -> list[tuple[str, frozenset[str]]]:
    """``(name, features removed)``, the full model first."""
    out = [("all v6", frozenset()), ("v5 only", frozenset(V6_FEATURE_NAMES))]
    out += [(f"without {family}", frozenset(names)) for family, names in V6_FAMILIES.items()]
    return out


def strip(rows: Sequence[Any], removed: frozenset[str]) -> list[Any]:
    """Copies of ``rows`` without the removed features."""
    if not removed:
        return list(rows)
    return [
        dataclasses.replace(
            row, features={k: v for k, v in row.features.items() if k not in removed}
        )
        for row in rows
    ]


def _score(target: PredictionTarget, rows: Sequence[Any], **train_kwargs: Any) -> float:
    """Test MAE for a stat, test log loss for P(plays)."""
    from euroleague_ai.models.enums import PredictionTarget as Target

    if target is Target.PLAYS:
        from euroleague_ai.ml.availability import train_availability

        return float(train_availability(rows, **train_kwargs).metrics.log_loss)
    from euroleague_ai.ml.train import train_target

    return float(train_target(list(rows), target, fit_expected=False, **train_kwargs).metrics.mae)


def ablate(target: PredictionTarget, rows: Sequence[Any], **train_kwargs: Any) -> dict[str, Any]:
    """Every variant for one target, with its change against the full model."""
    scores: list[tuple[str, int, float]] = []
    for name, removed in variants():
        scores.append((name, len(removed), _score(target, strip(rows, removed), **train_kwargs)))
    full = scores[0][2]
    deviations = [
        abs(
            _score(
                target, rows, **{**train_kwargs, "params": _reseeded(train_kwargs.get("params"), i)}
            )
            - full
        )
        for i in range(RESEEDS)
    ]
    noise = max(deviations) / full if full else math.nan
    bar = max(HELPS_THRESHOLD, 2 * noise)

    results: list[dict[str, Any]] = []
    for name, removed_count, score in scores:
        change = (score - full) / full if full else math.nan
        results.append(
            {
                "variant": name,
                "removed": removed_count,
                "score": round(score, 5),
                "change_pct": round(change * 100, 3),
                # Removing it made things worse, beyond noise: the family helps.
                "helps": bool(name.startswith("without ") and change > bar),
            }
        )
        logger.info("ablation variant", extra={"target": target.name, **results[-1]})
    return {
        "target": target.name,
        "metric": "log_loss" if target.name == "PLAYS" else "mae",
        "rows": len(rows),
        "noise_pct": round(noise * 100, 3),
        "helps_above_pct": round(bar * 100, 3),
        "variants": results,
    }


def verdict(reports: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Per family, the targets it helps. An empty list means: consider dropping it."""
    out: dict[str, list[str]] = {family: [] for family in V6_FAMILIES}
    for report in reports:
        for row in report["variants"]:
            if row["helps"]:
                out[row["variant"].removeprefix("without ")].append(report["target"])
    return out


def save_report(
    reports: Sequence[dict[str, Any]], *, now: datetime, root: Path | None = None
) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    directory = (root or DEFAULT_ARTIFACT_ROOT) / REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y%m%dT%H%M%S}.json"
    path.write_text(
        json.dumps(
            {"created_at": now.isoformat(), "targets": list(reports), "verdict": verdict(reports)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

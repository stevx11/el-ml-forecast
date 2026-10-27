"""Baselines.

The bar every model has to clear. A model that does not beat these is not a
model, it is a liability -- it costs training time, inference latency and
explanation complexity while producing worse numbers than three lines of
arithmetic.

**Baselines read the same feature snapshots the model does.** They could be
computed from raw box scores instead, but then they would be using a different
information set and the comparison would be dishonest in whichever direction
happened to favour the model. Sharing the snapshot guarantees both see exactly
what was knowable before tip-off, and nothing else.

The one to beat is :class:`MinutesScaledRate`. Most of the variance in a
box-score line is variance in playing time, so a per-minute rate scaled by
recent minutes is far stronger than it sounds. Comparing a tuned model only
against ``season mean`` is the classic way to convince yourself a model works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from euroleague_ai.ml.dataset import TARGET_FEATURE_STEMS, TrainingRow
from euroleague_ai.models.enums import PredictionTarget


def _feature(row: TrainingRow, name: str) -> float | None:
    """Read a feature, treating missing and null identically."""
    value = row.features.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Baseline(Protocol):
    """A prediction rule requiring no training."""

    @property
    def name(self) -> str: ...

    def predict(self, row: TrainingRow, target: PredictionTarget) -> float | None:
        """Return a prediction, or ``None`` when the inputs are unavailable.

        ``None`` matters: a baseline that silently guesses zero for players
        without history would score better than it deserves on exactly the rows
        it understands least.
        """
        ...


@dataclass(frozen=True, slots=True)
class SeasonMean:
    """The player's season-to-date average. The weakest honest baseline."""

    name: str = "season_mean"

    def predict(self, row: TrainingRow, target: PredictionTarget) -> float | None:
        return _feature(row, f"season_{TARGET_FEATURE_STEMS[target]}_mean")


@dataclass(frozen=True, slots=True)
class LastNMean:
    """Mean over the player's last ``window`` games."""

    window: int = 5

    @property
    def name(self) -> str:
        return f"last{self.window}_mean"

    def predict(self, row: TrainingRow, target: PredictionTarget) -> float | None:
        return _feature(row, f"last{self.window}_{TARGET_FEATURE_STEMS[target]}_mean")


@dataclass(frozen=True, slots=True)
class WeightedBlend:
    """Blend of season-to-date and recent form.

    Recency-weighted because EuroLeague rotations shift sharply mid-season, so a
    pure season average lags every role change by weeks.
    """

    season_weight: float = 0.4
    recent_weight: float = 0.6
    recent_window: int = 5
    name: str = "weighted_blend"

    def predict(self, row: TrainingRow, target: PredictionTarget) -> float | None:
        stem = TARGET_FEATURE_STEMS[target]
        season = _feature(row, f"season_{stem}_mean")
        recent = _feature(row, f"last{self.recent_window}_{stem}_mean")

        # Degrade gracefully: early in a season only one of the two exists, and
        # falling back beats refusing to predict.
        if season is None and recent is None:
            return None
        if season is None:
            return recent
        if recent is None:
            return season
        return self.season_weight * season + self.recent_weight * recent


@dataclass(frozen=True, slots=True)
class MinutesScaledRate:
    """Season per-minute rate multiplied by recent average minutes.

    **The baseline that matters.** It separates the two things that drive a box
    score line -- how much a player plays, and how productive they are per minute
    -- and recombines them using the most recent estimate of each. A model that
    cannot beat this has not learned anything beyond "playing time predicts
    counting stats".

    Applied to MINUTES itself it degenerates to the recent minutes average, which
    is the correct behaviour rather than a special case.
    """

    recent_window: int = 5
    name: str = "minutes_scaled_rate"

    def predict(self, row: TrainingRow, target: PredictionTarget) -> float | None:
        recent_minutes = _feature(row, f"last{self.recent_window}_minutes_mean")
        if recent_minutes is None:
            recent_minutes = _feature(row, "season_minutes_mean")
        if recent_minutes is None:
            return None

        if target is PredictionTarget.MINUTES:
            return recent_minutes

        stem = TARGET_FEATURE_STEMS[target]
        season_value = _feature(row, f"season_{stem}_mean")
        season_minutes = _feature(row, "season_minutes_mean")

        if season_value is None or not season_minutes:
            return None

        per_minute = season_value / season_minutes
        return per_minute * recent_minutes


def default_baselines() -> list[Baseline]:
    """The five baselines every target is measured against."""
    return [
        SeasonMean(),
        LastNMean(window=5),
        LastNMean(window=10),
        WeightedBlend(),
        MinutesScaledRate(),
    ]


def predict_all(
    rows: list[TrainingRow], target: PredictionTarget
) -> dict[str, list[tuple[float, float]]]:
    """Run every baseline over ``rows``.

    Returns:
        Baseline name to a list of ``(predicted, actual)`` pairs, omitting rows
        the baseline could not predict.
    """
    results: dict[str, list[tuple[float, float]]] = {}
    for baseline in default_baselines():
        pairs: list[tuple[float, float]] = []
        for row in rows:
            prediction = baseline.predict(row, target)
            if prediction is not None:
                pairs.append((prediction, row.target))
        results[baseline.name] = pairs
    return results


def baseline_coverage(rows: list[TrainingRow], target: PredictionTarget) -> dict[str, Any]:
    """Fraction of rows each baseline can actually predict.

    Reported alongside accuracy because a baseline that only predicts the easy
    half of the data will look artificially good. Comparing MAE across baselines
    with different coverage is comparing different problems.
    """
    total = len(rows)
    if total == 0:
        return {}
    return {
        baseline.name: round(
            sum(1 for row in rows if baseline.predict(row, target) is not None) / total, 4
        )
        for baseline in default_baselines()
    }

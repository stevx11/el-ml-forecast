"""Predicted distributions, threshold probabilities and confidence.

A point estimate is the wrong shape for this product. "Nine points" answers a
question nobody asked; the questions people actually have are *will he clear
fifteen* and *how sure are you*. Both need a distribution.

**Why empirical residuals rather than quantile regression.** The obvious
alternative is to train a LightGBM model per quantile per target -- twenty extra
boosters, each fitted independently, with no guarantee the 0.25 model stays below
the 0.75 model for a given player. This module instead measures how wrong the
existing median model actually was on held-out data, and reuses that error
distribution. It adds no training, cannot produce crossing quantiles, and its
correctness is directly checkable: if the 80% interval really contains 80% of
outcomes, it works. :func:`coverage_report` measures exactly that.

**What the residuals are conditioned on.** Two things, because a single pooled
error distribution would be wrong in opposite directions for different players:

1. **Predicted level.** A player projected for 20 points misses by more, in
   absolute terms, than one projected for 3. Variance scales with volume.
2. **How much history we have.** This is the one that matters for honesty. A
   player with no prior EuroLeague box scores gets the model's fallback constant,
   and presenting that with the same interval as a well-known starter is a lie.
   Conditioning on history depth makes his interval wide on its own, without any
   special-casing at the call site.

Cells with too few observations fall back to the history-marginal distribution
and then to the pooled one, so a sparse corner degrades to a broader estimate
rather than to a confidently wrong narrow one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Quantiles stored for every prediction. Deliberately including the tails: the
#: interesting question about a streaky shooter is what his good night looks
#: like, and that is the 0.9, not the median.
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

#: Central interval used for the width that drives confidence.
CONFIDENCE_INTERVAL: tuple[float, float] = (0.1, 0.9)

#: Upper edges for the "how many prior games do we have" buckets, read from the
#: ``games_played_career_window`` feature. The first bucket is the one that
#: matters: zero prior games means every form feature was NaN and the model
#: returned its fallback.
HISTORY_EDGES: tuple[float, ...] = (0.0, 4.0, 19.0)

#: Number of predicted-level buckets. Edges are quantiles of the validation
#: predictions, so cells hold roughly equal counts rather than equal widths --
#: predicted minutes are not uniformly distributed and equal widths would leave
#: the top bucket nearly empty.
LEVEL_BUCKETS = 5

#: Below this many observations a two-dimensional cell is not trusted on its own.
MIN_CELL_SIZE = 200

#: The same idea for the history-only marginal, but far lower, and deliberately.
#:
#: The first version used one threshold for both, and it silently defeated the
#: whole point. Players with no prior EuroLeague games are rare in any validation
#: slice -- a debut happens once per career -- so the zero-history marginal came
#: in under 200, fell through to the pooled sample, and every player in the
#: league got the same confidence of 0.50. The one signal the design existed to
#: capture was the one the fallback ate.
#:
#: Fifty residuals estimate an interquartile range perfectly adequately. Being
#: approximately right about an unknown player beats being precisely wrong.
MIN_MARGINAL_SIZE = 50


#: Thresholds published per target. Chosen to bracket the decisions people
#: actually make: the brief's example is P(15+), P(20+), P(25+) for points, and
#: the others follow the same logic -- a double-digit rebound night, a
#: five-assist floor game, a multi-three shooting night.
THRESHOLDS: dict[str, tuple[float, ...]] = {
    "MINUTES": (20.0, 25.0, 30.0),
    "POINTS": (10.0, 15.0, 20.0, 25.0),
    "REBOUNDS": (5.0, 8.0, 10.0),
    "ASSISTS": (3.0, 5.0, 7.0),
    "THREE_POINTERS_MADE": (1.0, 2.0, 3.0, 4.0),
}


def thresholds_for(target: str) -> tuple[float, ...]:
    """Published thresholds for a target, empty if it has none."""
    return THRESHOLDS.get(str(target).upper(), ())


#: Targets measured on a continuous scale. Everything else in a box score is a
#: whole number.
CONTINUOUS_TARGETS: frozenset[str] = frozenset({"MINUTES"})


def threshold_cutoff(target: str, threshold: float) -> float:
    """Where to evaluate P(actual >= threshold) on the residual distribution.

    **Continuity correction for whole-number stats.** The residual distribution
    is continuous: projection plus a residual measured on someone else. A player
    who made exactly one three, projected above this player, has a residual
    that lands *below* 1 when it is applied here, so outcomes sitting exactly on
    a bar are split across both sides of it and "at least k" is undercounted.
    For a whole-number stat, "at least k" is "more than k - 1", so the event is
    evaluated at k - 0.5, the midpoint.

    Measured, not assumed: on the 2025 test period P(X+) implied only 0.69-0.79
    of the threes nights that happened, 0.74-0.83 of assists, 0.76-0.83 of
    rebounds and 0.87-0.96 of points, while continuous minutes were exactly
    1.00. A synthetic league with the same fitting reproduces those ratios
    without the correction and lands within 2% of reality with it.
    """
    if str(target).upper() in CONTINUOUS_TARGETS:
        return float(threshold)
    return float(threshold) - 0.5


def history_bucket(games: float | None) -> int:
    """Bucket index for a player's prior-game count.

    ``None`` is treated as zero. A missing history count and a history count of
    zero mean the same thing here: we know nothing about this player.
    """
    value = 0.0 if games is None else float(games)
    for index, edge in enumerate(HISTORY_EDGES):
        if value <= edge:
            return index
    return len(HISTORY_EDGES)


def _quantile(sorted_values: list[float], level: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = level * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _fraction_at_least(sorted_values: list[float], cutoff: float) -> float:
    """Fraction of ``sorted_values`` greater than or equal to ``cutoff``.

    Interpolated rather than a raw count, so the probability moves smoothly as
    the threshold moves. A step function here would make P(15+) and P(15.1+)
    differ by a visible jump for no real reason.
    """
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if cutoff <= sorted_values[0]:
        return 1.0
    if cutoff > sorted_values[-1]:
        return 0.0

    low, high = 0, count - 1
    while low < high:
        middle = (low + high) // 2
        if sorted_values[middle] < cutoff:
            low = middle + 1
        else:
            high = middle
    index = low

    # Interpolate between the bracketing observations.
    previous = sorted_values[index - 1] if index > 0 else sorted_values[0]
    span = sorted_values[index] - previous
    within = 0.0 if span <= 0 else (cutoff - previous) / span
    exceeded = count - index + (1.0 - within)
    return max(0.0, min(1.0, exceeded / count))


@dataclass(slots=True)
class ResidualDistribution:
    """Held-out error distribution, conditioned on predicted level and history.

    Residuals are ``actual - predicted``. Adding them back to a fresh prediction
    turns a point estimate into a distribution.
    """

    #: Predicted-value cut points, ascending. ``len == LEVEL_BUCKETS - 1``.
    level_edges: list[float]
    #: ``"{level}:{history}"`` to that cell's sorted residuals.
    cells: dict[str, list[float]]
    #: History bucket to its residuals, used when a cell is too small.
    by_history: dict[str, list[float]]
    #: Every residual, the last resort.
    pooled: list[float]
    #: Pooled interquartile range, the reference for confidence.
    pooled_iqr: float = 0.0
    sample_size: int = 0
    #: Cells that fell back, for the training report.
    thin_cells: list[str] = field(default_factory=list)

    def level_bucket(self, predicted: float) -> int:
        for index, edge in enumerate(self.level_edges):
            if predicted <= edge:
                return index
        return len(self.level_edges)

    def residuals_for(self, predicted: float, games_played: float | None) -> list[float]:
        """The most specific residual sample that is large enough to trust."""
        history = history_bucket(games_played)
        key = f"{self.level_bucket(predicted)}:{history}"

        cell = self.cells.get(key)
        if cell is not None and len(cell) >= MIN_CELL_SIZE:
            return cell

        marginal = self.by_history.get(str(history))
        if marginal is not None and len(marginal) >= MIN_MARGINAL_SIZE:
            return marginal

        return self.pooled

    def quantiles(self, predicted: float, games_played: float | None) -> dict[str, float]:
        """Predicted quantiles, clipped at zero and forced non-decreasing.

        Clipping can flatten the lower tail (a bench player's 0.05 and 0.1 both
        land on zero), which is correct -- you cannot score negative points.
        """
        residuals = self.residuals_for(predicted, games_played)
        out: dict[str, float] = {}
        previous = 0.0
        for level in QUANTILE_LEVELS:
            value = max(0.0, predicted + _quantile(residuals, level))
            previous = max(previous, value)
            out[str(level)] = round(previous, 2)
        return out

    def probability_at_least(
        self, predicted: float, games_played: float | None, threshold: float
    ) -> float:
        """P(actual >= threshold), from the held-out error distribution."""
        residuals = self.residuals_for(predicted, games_played)
        return round(_fraction_at_least(residuals, threshold - predicted), 4)

    def mean_residual(self, predicted: float, games_played: float | None) -> float:
        """Mean of ``actual - predicted`` in this prediction's error cell.

        Added to the median, it gives the cell's *expected* outcome: the
        correction from "what he usually does" to "what he averages". Uses
        the same cell-then-marginal-then-pooled fallback as the quantiles, so a
        thin cell borrows from a broader one instead of producing a noisy mean.
        """
        residuals = self.residuals_for(predicted, games_played)
        return sum(residuals) / len(residuals) if residuals else 0.0

    def spread(self, predicted: float, games_played: float | None) -> float:
        """Robust standard deviation, from the 10-90 range.

        ``(q90 - q10) / 2.563`` is the normal-consistent estimator. Preferred
        over the raw standard deviation because a single 40-point night should
        not widen every interval we publish.
        """
        residuals = self.residuals_for(predicted, games_played)
        low, high = CONFIDENCE_INTERVAL
        width = _quantile(residuals, high) - _quantile(residuals, low)
        return round(max(0.0, width / 2.563), 4)

    def confidence(self, predicted: float, games_played: float | None) -> float:
        """How tight this prediction is relative to a typical one, in (0, 1].

        ``1 / (1 + cell_iqr / pooled_iqr)``: exactly average spread scores 0.5, a
        tighter-than-usual case scores above it, an unknown player well below.

        This is deliberately *relative*. There is no absolute scale on which a
        basketball projection is "90% confident", and inventing one would be
        false precision. It answers "how much more or less sure are we here than
        we usually are", which is the question a reader actually has.
        """
        if self.pooled_iqr <= 0:
            return 0.5
        residuals = self.residuals_for(predicted, games_played)
        iqr = _quantile(residuals, 0.75) - _quantile(residuals, 0.25)
        return round(1.0 / (1.0 + iqr / self.pooled_iqr), 4)

    def occupancy(self) -> dict[str, int]:
        """How many residuals sit in each history bucket, plus the total.

        Surfaced in the training report because a distribution that has quietly
        fallen back to the pooled sample looks identical, from the outside, to
        one that is working -- it just returns the same confidence for everyone.
        """
        counts = {f"history_{k}": len(v) for k, v in sorted(self.by_history.items())}
        counts["total"] = self.sample_size
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_edges": self.level_edges,
            "cells": self.cells,
            "by_history": self.by_history,
            "pooled": self.pooled,
            "pooled_iqr": self.pooled_iqr,
            "sample_size": self.sample_size,
            "thin_cells": self.thin_cells,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualDistribution:
        return cls(
            level_edges=[float(v) for v in payload["level_edges"]],
            cells={k: [float(x) for x in v] for k, v in payload["cells"].items()},
            by_history={
                k: [float(x) for x in v] for k, v in payload["by_history"].items()
            },
            pooled=[float(v) for v in payload["pooled"]],
            pooled_iqr=float(payload.get("pooled_iqr", 0.0)),
            sample_size=int(payload.get("sample_size", 0)),
            thin_cells=list(payload.get("thin_cells", [])),
        )


def fit_residual_distribution(
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
) -> ResidualDistribution:
    """Fit on held-out predictions -- validation, never test.

    Fitting the error distribution on the same rows used to report accuracy
    would make the intervals look better calibrated than they are, for the same
    reason the point-estimate offset is fitted on validation.
    """
    if not (len(predictions) == len(actuals) == len(games_played)):
        raise ValueError("predictions, actuals and games_played must align")

    ordered_predictions = sorted(predictions)
    level_edges = [
        round(_quantile(ordered_predictions, (index + 1) / LEVEL_BUCKETS), 4)
        for index in range(LEVEL_BUCKETS - 1)
    ]
    # Duplicate edges collapse buckets, which is fine and happens when a target
    # is mostly zero (three-pointers). Deduplicate so bucket indices stay sane.
    level_edges = sorted(set(level_edges))

    distribution = ResidualDistribution(
        level_edges=level_edges, cells={}, by_history={}, pooled=[]
    )

    raw_cells: dict[str, list[float]] = {}
    raw_history: dict[str, list[float]] = {}
    pooled: list[float] = []

    for predicted, actual, games in zip(predictions, actuals, games_played, strict=True):
        residual = float(actual) - float(predicted)
        history = history_bucket(games)
        key = f"{distribution.level_bucket(float(predicted))}:{history}"
        raw_cells.setdefault(key, []).append(residual)
        raw_history.setdefault(str(history), []).append(residual)
        pooled.append(residual)

    distribution.cells = {k: sorted(v) for k, v in raw_cells.items()}
    distribution.by_history = {k: sorted(v) for k, v in raw_history.items()}
    distribution.pooled = sorted(pooled)
    distribution.sample_size = len(pooled)
    distribution.pooled_iqr = _quantile(distribution.pooled, 0.75) - _quantile(
        distribution.pooled, 0.25
    )
    distribution.thin_cells = sorted(
        key for key, values in distribution.cells.items() if len(values) < MIN_CELL_SIZE
    )
    return distribution


def coverage_report(
    distribution: ResidualDistribution,
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
) -> dict[str, float]:
    """Measure whether the intervals mean what they claim, on the test split.

    **This is the gate for a distribution.** A point estimate is judged by MAE; an
    interval is judged by coverage. If the nominal 80% interval contains 62% of
    outcomes it is not a useful interval, however good the median underneath it
    is. Reported as ``covered_80`` against a target of 0.80.
    """
    low_level, high_level = CONFIDENCE_INTERVAL
    inside_80 = 0
    inside_50 = 0
    below = 0
    above = 0

    for predicted, actual, games in zip(predictions, actuals, games_played, strict=True):
        residuals = distribution.residuals_for(float(predicted), games)
        low = max(0.0, float(predicted) + _quantile(residuals, low_level))
        high = max(0.0, float(predicted) + _quantile(residuals, high_level))
        q25 = max(0.0, float(predicted) + _quantile(residuals, 0.25))
        q75 = max(0.0, float(predicted) + _quantile(residuals, 0.75))

        value = float(actual)
        if low <= value <= high:
            inside_80 += 1
        elif value < low:
            below += 1
        else:
            above += 1
        if q25 <= value <= q75:
            inside_50 += 1

    total = max(1, len(predictions))
    return {
        "covered_80": round(inside_80 / total, 4),
        "covered_50": round(inside_50 / total, 4),
        # Asymmetric misses say *which way* the interval is wrong, which a single
        # coverage number hides.
        "below_interval": round(below / total, 4),
        "above_interval": round(above / total, 4),
        "rows": float(total),
    }

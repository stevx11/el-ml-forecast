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

**Each player's own spread.** Level and history alone give a streaky shooter
and a steady big man projected for the same 12 points identical ranges. The
residuals can be scaled by how much the player's own recent numbers swing
(``std10_<stat>``, or minutes volatility), relative to the typical swing at his
projected level::

    k = clip((volatility / median volatility at this level) ** power, 0.5, 2.0)

The cells then hold residuals divided by ``k``, and a prediction gets its cell's
quantiles multiplied by its own ``k``. ``power = 0`` is the unscaled distribution,
exactly as before. The source and the power are chosen on validation alone
(:func:`choose_volatility_scaling`): fitted on its earlier part, scored by
quantile loss on its later part, kept only if they beat the unscaled spread.
"""

from __future__ import annotations

import math
import statistics
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


#: Bounds on the per-player scale. A player whose last ten games were identical
#: is not certain to repeat them, and one with a single wild night is not twice
#: as unpredictable as everyone else.
SCALE_BOUNDS: tuple[float, float] = (0.5, 2.0)

#: Powers tried for the volatility scale. 0 (unscaled) is always the incumbent.
#: Fine at the low end, where real data lands: on the first v6 retrain minutes
#: and threes both chose 0.25, and for threes that overshot (steady shooters'
#: coverage fell from 84% to 76.5%), so the step below it matters.
SCALE_POWERS: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)

#: The later share of validation that scores the candidates. Validation is
#: chronological, so this is "fit on the past, judge on what came next".
SCALING_HOLDOUT = 0.4

#: A scaled spread has to beat the unscaled one by this much quantile loss to
#: be chosen. Below that the gain is noise, and simpler wins.
SCALING_MIN_GAIN = 0.002


def volatility_features(stem: str) -> tuple[str, ...]:
    """Candidate volatility sources for a target: its own swing, then minutes'."""
    own = f"std10_{stem}"
    return (own,) if stem == "minutes" else (own, "std10_minutes")


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


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    #: The feature whose value scales a player's spread, ``None`` for unscaled.
    volatility_feature: str | None = None
    #: Exponent of the scale. 0 means every player gets his cell's spread.
    scale_power: float = 0.0
    #: Level bucket to the median volatility in it, the reference for ``k``.
    volatility_reference: dict[str, float] = field(default_factory=dict)

    def level_bucket(self, predicted: float) -> int:
        for index, edge in enumerate(self.level_edges):
            if predicted <= edge:
                return index
        return len(self.level_edges)

    @property
    def scaled(self) -> bool:
        return bool(self.volatility_feature) and self.scale_power > 0

    def volatility_of(self, features: dict[str, Any]) -> float | None:
        """This distribution's volatility input, read from a feature vector."""
        if not self.volatility_feature:
            return None
        return _finite(features.get(self.volatility_feature))

    def scale(self, predicted: float, volatility: float | None) -> float:
        """The player's spread relative to his cell's: 1.0 when unknown or unscaled."""
        if not self.scaled:
            return 1.0
        value = _finite(volatility)
        reference = self.volatility_reference.get(str(self.level_bucket(predicted)))
        if value is None or not reference or reference <= 0:
            return 1.0
        low, high = SCALE_BOUNDS
        if value <= 0:
            return low
        return float(min(high, max(low, (value / reference) ** self.scale_power)))

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

    def quantiles(
        self, predicted: float, games_played: float | None, volatility: float | None = None
    ) -> dict[str, float]:
        """Predicted quantiles, clipped at zero and forced non-decreasing.

        Clipping can flatten the lower tail (a bench player's 0.05 and 0.1 both
        land on zero), which is correct -- you cannot score negative points.
        """
        residuals = self.residuals_for(predicted, games_played)
        k = self.scale(predicted, volatility)
        out: dict[str, float] = {}
        previous = 0.0
        for level in QUANTILE_LEVELS:
            value = max(0.0, predicted + k * _quantile(residuals, level))
            previous = max(previous, value)
            out[str(level)] = round(previous, 2)
        return out

    def probability_at_least(
        self,
        predicted: float,
        games_played: float | None,
        threshold: float,
        volatility: float | None = None,
    ) -> float:
        """P(actual >= threshold), from the held-out error distribution."""
        residuals = self.residuals_for(predicted, games_played)
        k = self.scale(predicted, volatility)
        return round(_fraction_at_least(residuals, (threshold - predicted) / k), 4)

    def mean_residual(
        self, predicted: float, games_played: float | None, volatility: float | None = None
    ) -> float:
        """Mean of ``actual - predicted`` in this prediction's error cell.

        Added to the median, it gives the cell's *expected* outcome: the
        correction from "what he usually does" to "what he averages". Uses
        the same cell-then-marginal-then-pooled fallback as the quantiles, so a
        thin cell borrows from a broader one instead of producing a noisy mean.
        """
        residuals = self.residuals_for(predicted, games_played)
        if not residuals:
            return 0.0
        return self.scale(predicted, volatility) * sum(residuals) / len(residuals)

    def spread(
        self, predicted: float, games_played: float | None, volatility: float | None = None
    ) -> float:
        """Robust standard deviation, from the 10-90 range.

        ``(q90 - q10) / 2.563`` is the normal-consistent estimator. Preferred
        over the raw standard deviation because a single 40-point night should
        not widen every interval we publish.
        """
        residuals = self.residuals_for(predicted, games_played)
        low, high = CONFIDENCE_INTERVAL
        width = _quantile(residuals, high) - _quantile(residuals, low)
        width *= self.scale(predicted, volatility)
        return round(max(0.0, width / 2.563), 4)

    def confidence(
        self, predicted: float, games_played: float | None, volatility: float | None = None
    ) -> float:
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
        iqr *= self.scale(predicted, volatility)
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
            "volatility_feature": self.volatility_feature,
            "scale_power": self.scale_power,
            "volatility_reference": self.volatility_reference,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualDistribution:
        return cls(
            level_edges=[float(v) for v in payload["level_edges"]],
            cells={k: [float(x) for x in v] for k, v in payload["cells"].items()},
            by_history={k: [float(x) for x in v] for k, v in payload["by_history"].items()},
            pooled=[float(v) for v in payload["pooled"]],
            pooled_iqr=float(payload.get("pooled_iqr", 0.0)),
            sample_size=int(payload.get("sample_size", 0)),
            thin_cells=list(payload.get("thin_cells", [])),
            # Absent in distributions saved before per-player spreads: unscaled.
            volatility_feature=payload.get("volatility_feature"),
            scale_power=float(payload.get("scale_power") or 0.0),
            volatility_reference={
                k: float(v) for k, v in (payload.get("volatility_reference") or {}).items()
            },
        )


def fit_residual_distribution(
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
    volatilities: list[float | None] | None = None,
    *,
    volatility_feature: str | None = None,
    scale_power: float = 0.0,
) -> ResidualDistribution:
    """Fit on held-out predictions -- validation, never test.

    Fitting the error distribution on the same rows used to report accuracy
    would make the intervals look better calibrated than they are, for the same
    reason the point-estimate offset is fitted on validation.

    With ``volatility_feature`` and a positive ``scale_power``, ``volatilities``
    (that feature's value per row) scale each residual by the player's own
    swing; see the module docstring. Without them the fit is unscaled.
    """
    if not (len(predictions) == len(actuals) == len(games_played)):
        raise ValueError("predictions, actuals and games_played must align")
    if volatilities is not None and len(volatilities) != len(predictions):
        raise ValueError("volatilities must align with predictions")

    ordered_predictions = sorted(predictions)
    level_edges = [
        round(_quantile(ordered_predictions, (index + 1) / LEVEL_BUCKETS), 4)
        for index in range(LEVEL_BUCKETS - 1)
    ]
    # Duplicate edges collapse buckets, which is fine and happens when a target
    # is mostly zero (three-pointers). Deduplicate so bucket indices stay sane.
    level_edges = sorted(set(level_edges))

    distribution = ResidualDistribution(level_edges=level_edges, cells={}, by_history={}, pooled=[])

    vols: list[float | None] = [None] * len(predictions)
    if volatilities is not None:
        vols = [_finite(v) for v in volatilities]
    if volatility_feature and scale_power > 0 and volatilities is not None:
        by_level: dict[str, list[float]] = {}
        for predicted, vol in zip(predictions, vols, strict=True):
            if vol is not None:
                key = str(distribution.level_bucket(float(predicted)))
                by_level.setdefault(key, []).append(vol)
        distribution.volatility_feature = volatility_feature
        distribution.scale_power = float(scale_power)
        distribution.volatility_reference = {
            key: round(statistics.median(values), 4) for key, values in by_level.items()
        }

    raw_cells: dict[str, list[float]] = {}
    raw_history: dict[str, list[float]] = {}
    pooled: list[float] = []

    for predicted, actual, games, vol in zip(predictions, actuals, games_played, vols, strict=True):
        k = distribution.scale(float(predicted), vol)
        residual = (float(actual) - float(predicted)) / k
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


def quantile_loss(
    distribution: ResidualDistribution,
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
    volatilities: list[float | None] | None = None,
) -> float:
    """Mean pinball loss over :data:`QUANTILE_LEVELS`: lower is a better spread.

    A proper score for quantiles: it rewards intervals that are narrow where the
    outcome really is predictable and wide where it is not, and cannot be gamed
    by making every interval wide (or narrow) at once.
    """
    if not predictions:
        return float("nan")
    vols: list[float | None] = (
        list(volatilities) if volatilities is not None else [None] * len(predictions)
    )
    total = 0.0
    for predicted, actual, games, vol in zip(predictions, actuals, games_played, vols, strict=True):
        quantiles = distribution.quantiles(float(predicted), games, vol)
        for level in QUANTILE_LEVELS:
            miss = float(actual) - quantiles[str(level)]
            total += max(level * miss, (level - 1.0) * miss)
    return total / (len(predictions) * len(QUANTILE_LEVELS))


@dataclass(slots=True)
class ScalingChoice:
    """Which volatility scale, if any, the spread uses, and why."""

    volatility_feature: str | None
    scale_power: float
    #: Holdout quantile loss per candidate, ``"unscaled"`` included.
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "volatility_feature": self.volatility_feature,
            "scale_power": self.scale_power,
            "holdout_quantile_loss": self.scores,
        }


def choose_volatility_scaling(
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
    candidates: dict[str, list[float | None]],
    *,
    powers: tuple[float, ...] = SCALE_POWERS,
    holdout: float = SCALING_HOLDOUT,
    min_gain: float = SCALING_MIN_GAIN,
) -> ScalingChoice:
    """Pick a volatility source and power on validation alone.

    Rows must be in time order. Every candidate (and the unscaled incumbent) is
    fitted on the earlier part and scored on the later part with
    :func:`quantile_loss`. A candidate wins only if it beats the unscaled loss
    by ``min_gain`` (relative); otherwise the spread stays unscaled.
    """
    n = len(predictions)
    cut = n - int(n * holdout)
    if cut < MIN_CELL_SIZE or n - cut < MIN_CELL_SIZE:
        return ScalingChoice(None, 0.0, {})

    def part(values: list[Any], first: bool) -> list[Any]:
        return values[:cut] if first else values[cut:]

    fit_args = (part(predictions, True), part(actuals, True), part(games_played, True))
    score_args = (part(predictions, False), part(actuals, False), part(games_played, False))

    unscaled = fit_residual_distribution(*fit_args)
    scores = {"unscaled": round(quantile_loss(unscaled, *score_args), 5)}
    best: tuple[float, str | None, float] = (scores["unscaled"], None, 0.0)
    for feature, volatilities in candidates.items():
        for power in powers:
            fitted = fit_residual_distribution(
                *fit_args,
                part(volatilities, True),
                volatility_feature=feature,
                scale_power=power,
            )
            loss = quantile_loss(fitted, *score_args, part(volatilities, False))
            scores[f"{feature}^{power}"] = round(loss, 5)
            if loss < best[0]:
                best = (loss, feature, power)

    loss, chosen, power = best
    if chosen is None or loss > scores["unscaled"] * (1.0 - min_gain):
        return ScalingChoice(None, 0.0, scores)
    return ScalingChoice(chosen, power, scores)


#: Groups for coverage by volatility: below the 1/3 and above the 2/3 quantile
#: of the player's swing relative to his level.
VOLATILITY_GROUPS: tuple[str, ...] = ("low", "mid", "high")


def _relative_volatility(
    predictions: list[float], volatilities: list[float | None], level_of: Any
) -> list[float | None]:
    """Each row's volatility over the median volatility at its level."""
    by_level: dict[int, list[float]] = {}
    for predicted, vol in zip(predictions, volatilities, strict=True):
        if vol is not None:
            by_level.setdefault(level_of(float(predicted)), []).append(vol)
    medians = {k: statistics.median(v) for k, v in by_level.items()}
    out: list[float | None] = []
    for predicted, vol in zip(predictions, volatilities, strict=True):
        reference = medians.get(level_of(float(predicted)))
        out.append(vol / reference if vol is not None and reference else None)
    return out


def _interval_hits(
    distribution: ResidualDistribution,
    predicted: float,
    actual: float,
    games: float | None,
    volatility: float | None,
) -> tuple[bool, bool, bool]:
    """(inside 80%, below it, inside 50%) for one outcome."""
    quantiles = distribution.quantiles(predicted, games, volatility)
    low_level, high_level = CONFIDENCE_INTERVAL
    low = quantiles[str(low_level)]
    high = quantiles[str(high_level)]
    inside_80 = low <= actual <= high
    return inside_80, actual < low, quantiles["0.25"] <= actual <= quantiles["0.75"]


def threshold_brier(
    distribution: ResidualDistribution,
    target: str,
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
    volatilities: list[float | None] | None = None,
) -> float | None:
    """Mean Brier score of the published P(X+) chances, every threshold pooled."""
    bars = thresholds_for(target)
    if not bars or not predictions:
        return None
    vols: list[float | None] = (
        list(volatilities) if volatilities is not None else [None] * len(predictions)
    )
    total = 0.0
    count = 0
    for predicted, actual, games, vol in zip(predictions, actuals, games_played, vols, strict=True):
        for bar in bars:
            p = distribution.probability_at_least(
                float(predicted), games, threshold_cutoff(target, bar), vol
            )
            total += (p - (1.0 if float(actual) >= bar else 0.0)) ** 2
            count += 1
    return round(total / count, 5) if count else None


def coverage_report(
    distribution: ResidualDistribution,
    predictions: list[float],
    actuals: list[float],
    games_played: list[float | None],
    *,
    volatilities: list[float | None] | None = None,
    group_volatilities: list[float | None] | None = None,
    target: str | None = None,
    reference: ResidualDistribution | None = None,
) -> dict[str, Any]:
    """Measure whether the intervals mean what they claim, on the test split.

    **This is the gate for a distribution.** A point estimate is judged by MAE; an
    interval is judged by coverage. If the nominal 80% interval contains 62% of
    outcomes it is not a useful interval, however good the median underneath it
    is. Reported as ``covered_80`` against a target of 0.80.

    ``volatilities`` are the distribution's own scaling inputs per row.
    ``group_volatilities`` (the target's own recent swing) split the rows into
    low, mid and high volatility thirds, and coverage is reported per third: a
    spread that is right on average can still be too narrow for streaky
    players and too wide for steady ones. With ``target``, the Brier score of
    the published P(X+) chances is added. With ``reference`` (the unscaled
    distribution), the same numbers are reported for it, to compare.
    """
    vols: list[float | None] = (
        list(volatilities) if volatilities is not None else [None] * len(predictions)
    )
    inside_80 = 0
    inside_50 = 0
    below = 0
    above = 0
    hits: list[bool] = []
    reference_hits: list[bool] = []

    for predicted, actual, games, vol in zip(predictions, actuals, games_played, vols, strict=True):
        value = float(actual)
        in_80, is_below, in_50 = _interval_hits(distribution, float(predicted), value, games, vol)
        hits.append(in_80)
        if in_80:
            inside_80 += 1
        elif is_below:
            below += 1
        else:
            above += 1
        if in_50:
            inside_50 += 1
        if reference is not None:
            reference_hits.append(
                _interval_hits(reference, float(predicted), value, games, None)[0]
            )

    total = max(1, len(predictions))
    out: dict[str, Any] = {
        "covered_80": round(inside_80 / total, 4),
        "covered_50": round(inside_50 / total, 4),
        # Asymmetric misses say *which way* the interval is wrong, which a single
        # coverage number hides.
        "below_interval": round(below / total, 4),
        "above_interval": round(above / total, 4),
        "rows": float(total),
    }

    if group_volatilities is not None and predictions:
        relative = _relative_volatility(
            [float(p) for p in predictions],
            [_finite(v) for v in group_volatilities],
            distribution.level_bucket,
        )
        known = sorted(v for v in relative if v is not None)
        if len(known) >= len(VOLATILITY_GROUPS):
            edges = (_quantile(known, 1 / 3), _quantile(known, 2 / 3))

            def group(value: float | None) -> str | None:
                if value is None:
                    return None
                return "low" if value <= edges[0] else "mid" if value <= edges[1] else "high"

            groups = [group(v) for v in relative]

            def by_group(flags: list[bool]) -> dict[str, dict[str, float]]:
                table: dict[str, dict[str, float]] = {}
                for name in VOLATILITY_GROUPS:
                    chosen = [f for f, g in zip(flags, groups, strict=True) if g == name]
                    if chosen:
                        table[name] = {
                            "covered_80": round(sum(chosen) / len(chosen), 4),
                            "rows": float(len(chosen)),
                        }
                return table

            out["by_volatility"] = by_group(hits)
            if reference is not None:
                out["reference_by_volatility"] = by_group(reference_hits)

    if reference is not None:
        out["reference_covered_80"] = round(sum(reference_hits) / total, 4)
    if target is not None:
        brier = threshold_brier(distribution, target, predictions, actuals, games_played, vols)
        if brier is not None:
            out["threshold_brier"] = brier
            if reference is not None:
                out["reference_threshold_brier"] = threshold_brier(
                    reference, target, predictions, actuals, games_played
                )
    return out

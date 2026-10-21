"""Are the projections as spread out as real box scores?

Error metrics can look fine while every projection huddles near the middle:
nobody is projected for 25, yet real games have plenty of 25-point nights. Three
views on the test period, on rows none of the models has seen:

* **Calibration slope.** Regress what happened on the expected value. For an
  honest conditional mean the slope is 1: a player projected two points above
  another really does score about two more. Above 1, projections are too
  bunched (the high ones should be higher, the low ones lower); below 1, too
  spread out. The intercept and slope together say where the shrinkage is.
* **Deciles.** Rows sorted by expected value into ten groups: the mean
  projection against the mean outcome in each. Bunching shows as the top decile
  scoring more than projected and the bottom less.
* **Big nights.** For each published bar (15+, 20+), the number the threshold
  probabilities implied (their sum) against the number that happened. Too few
  implied means the tails are too thin.

The medians' own spread is reported but not judged: a median of a skewed stat
should sit below the mean, so its quantiles are expected to be lower than the
outcomes'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.ml.inference import LoadedModel
    from euroleague_ai.models.enums import PredictionTarget

QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

#: A slope this far from 1 is worth acting on. Inside it, sampling noise and
#: the clip at zero explain the gap.
SLOPE_TOLERANCE = 0.05


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def calibration_line(predicted: Sequence[float], actual: Sequence[float]) -> tuple[float, float]:
    """``(intercept, slope)`` of actual on predicted, by least squares."""
    n = len(predicted)
    if n < 2:
        return float("nan"), float("nan")
    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n
    var_p = sum((p - mean_p) ** 2 for p in predicted)
    if var_p == 0:
        return float("nan"), float("nan")
    cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual, strict=True))
    slope = cov / var_p
    return mean_a - slope * mean_p, slope


def deciles(
    predicted: Sequence[float], actual: Sequence[float], groups: int = 10
) -> list[dict[str, float]]:
    order = sorted(range(len(predicted)), key=lambda i: predicted[i])
    out = []
    for g in range(groups):
        chunk = order[g * len(order) // groups : (g + 1) * len(order) // groups]
        if not chunk:
            continue
        out.append(
            {
                "decile": g + 1,
                "rows": len(chunk),
                "projected": round(sum(predicted[i] for i in chunk) / len(chunk), 3),
                "actual": round(sum(actual[i] for i in chunk) / len(chunk), 3),
            }
        )
    return out


def big_nights(
    thresholds: Sequence[dict[str, Any] | None], actual: Sequence[float]
) -> list[dict[str, Any]]:
    """Implied (sum of P) against realised counts, per published bar."""
    implied: dict[str, float] = {}
    realised: dict[str, int] = {}
    rows: dict[str, int] = {}
    for probabilities, outcome in zip(thresholds, actual, strict=True):
        for bar, probability in (probabilities or {}).items():
            implied[bar] = implied.get(bar, 0.0) + float(probability)
            realised[bar] = realised.get(bar, 0) + int(outcome >= float(bar))
            rows[bar] = rows.get(bar, 0) + 1
    return [
        {
            "bar": bar,
            "rows": rows[bar],
            "implied": round(implied[bar], 1),
            "realised": realised[bar],
            "ratio": round(implied[bar] / realised[bar], 3) if realised[bar] else None,
        }
        for bar in sorted(implied, key=float)
    ]


@dataclass(slots=True)
class SpreadReport:
    target: str
    rows: int
    expected_source: str
    quantiles: dict[str, dict[str, float]]
    std: dict[str, float]
    intercept: float
    slope: float
    median_slope: float
    deciles: list[dict[str, float]]
    big_nights: list[dict[str, Any]]

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.slope):
            return "unknown"
        if self.slope > 1 + SLOPE_TOLERANCE:
            return "too bunched"
        if self.slope < 1 - SLOPE_TOLERANCE:
            return "too spread out"
        return "about right"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "rows": self.rows,
            "expected_source": self.expected_source,
            "quantiles": self.quantiles,
            "std": self.std,
            "intercept": round(self.intercept, 4),
            "slope": round(self.slope, 4),
            "median_slope": round(self.median_slope, 4),
            "verdict": self.verdict,
            "deciles": self.deciles,
            "big_nights": self.big_nights,
        }


def spread_report(
    target: str,
    *,
    medians: Sequence[float],
    expected: Sequence[float | None],
    actual: Sequence[float],
    thresholds: Sequence[dict[str, Any] | None],
    expected_source: str = "",
) -> SpreadReport:
    """Every view, from aligned per-row sequences."""
    # Where no expected value is published, the median stands in, as on the board.
    means = [
        float(e) if e is not None else float(m) for e, m in zip(expected, medians, strict=True)
    ]

    def describe(values: Sequence[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {f"{q:g}": round(_quantile(ordered, q), 3) for q in QUANTILES}

    def std(values: Sequence[float]) -> float:
        mean = sum(values) / len(values)
        return round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 4)

    intercept, slope = calibration_line(means, actual)
    _, median_slope = calibration_line(list(medians), actual)
    return SpreadReport(
        target=target,
        rows=len(actual),
        expected_source=expected_source,
        quantiles={
            "median": describe(medians),
            "expected": describe(means),
            "actual": describe(actual),
        },
        std={"median": std(medians), "expected": std(means), "actual": std(actual)},
        intercept=intercept,
        slope=slope,
        median_slope=median_slope,
        deciles=deciles(means, actual),
        big_nights=big_nights(thresholds, actual),
    )


async def target_spread(
    session: AsyncSession, target: PredictionTarget, model: LoadedModel, *, feature_set_version: str
) -> SpreadReport:
    """The active model's spread on its test period."""
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.train import chronological_split

    rows = await load_training_rows(session, target=target, feature_set_version=feature_set_version)
    _, _, test = chronological_split(rows)
    if not test:
        raise ValueError(f"no test rows for {target.name}")
    features = [row.features for row in test]
    medians = model.predict_batch(features)
    expected = model.expected_batch(features, medians)
    thresholds = [
        model.distribution_for(value, row_features).get("threshold_probabilities")
        for value, row_features in zip(medians, features, strict=True)
    ]
    return spread_report(
        target.name,
        medians=medians,
        expected=expected,
        actual=[float(row.target) for row in test],
        thresholds=thresholds,
        expected_source=str(model.expected_source or ""),
    )

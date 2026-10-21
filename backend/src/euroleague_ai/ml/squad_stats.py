"""Squad stat lines that add up to a plausible team: points, assists, rebounds, threes.

**The problem.** Each player's expected points, rebounds and assists are
predicted on their own. Summed over a squad they are unbiased on average, but a
squad projected for a fast game against a weak defence comes out the same as
one heading into a grind: nothing ties the sum to what this team, in this game,
is likely to produce. Assists drift away from scoring, and rebounds ignore how
many shots both sides will miss.

**The anchor.** For each side and each stat, a small linear model fitted on past
games says what the squad total should be::

    A = b0 + b1 * S_own (+ b2 * S_opp) + c . context

``S`` is the squad's expected total from the player models (each player's
expected value weighted by his chance of playing), ``context`` is team-level
information every candidate row already carries:

* points: the game's expected team points (ratings and pace) and pace;
* assists: the squad's expected points as well as its expected assists, so a
  side projected to score more is projected to assist more;
* rebounds: both squads' expected rebounds, both teams' offensive and defensive
  rebounds per game, pace and the expected total (more scoring, fewer misses);
* threes: threes attempted per game and pace.

Every regressor is known at the cutoff. Fitted by least squares on the
validation period only.

**The pull.** Each player on the side is scaled by one factor,
``(A / S) ** alpha``, so the correction lands mostly on the players who carry
the line, as it does in real games. ``alpha`` runs from 0 (off) to 1 (squad sums
exactly to ``A``) and is the strongest whose validation player error is no worse
than unscaled, the rule the minutes reconciliation uses.

**The gate, per stat.** On the test period, unseen by every model involved, the
squad-total error must fall and player error must not be *significantly* worse:
a paired comparison of squared error (these are expected values, so squared
error is the loss they target), with a standard error clustered by squad. A
stat that fails is saved switched off and predictions keep the model's values.

**What it changes.** Only expected values. Medians, intervals and threshold
probabilities stay as the models produced them, because scoring and calibration
are built on those. Injury reports apply first: a player ruled out has
P(plays) = 0, drops out of ``S``, and the squad's share is spread over the
players who will play.

The fit records every model version it used; after a retrain it no longer
applies until ``ml reconcile`` is run again.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TARGET_COLUMNS
from euroleague_ai.ml.reconcile import ALPHA_GRID
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.game import GameTeam, PlayerGameStats

if TYPE_CHECKING:  # pragma: no cover - typing only
    import uuid
    from collections.abc import Hashable, Mapping, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.ml.availability import AvailabilityRow
    from euroleague_ai.ml.inference import LoadedModel

logger = get_logger(__name__)

POINTS = PredictionTarget.POINTS
ASSISTS = PredictionTarget.ASSISTS
REBOUNDS = PredictionTarget.REBOUNDS
THREES = PredictionTarget.THREE_POINTERS_MADE

#: Reconciled stats, in reporting order.
STAT_TARGETS: tuple[PredictionTarget, ...] = (POINTS, ASSISTS, REBOUNDS, THREES)


@dataclass(frozen=True, slots=True)
class AnchorSpec:
    """What a stat's squad anchor is built from."""

    #: Squad sums used as regressors: ``("own" | "opp", target)``.
    squads: tuple[tuple[str, PredictionTarget], ...]
    #: Team-context features, read from any candidate row on the side.
    context: tuple[str, ...]

    def names(self) -> list[str]:
        return [f"{side}_{target.name.lower()}" for side, target in self.squads] + list(
            self.context
        )


SPECS: dict[PredictionTarget, AnchorSpec] = {
    POINTS: AnchorSpec(
        squads=(("own", POINTS),),
        context=("game_expected_team_points", "matchup_expected_pace"),
    ),
    ASSISTS: AnchorSpec(
        squads=(("own", ASSISTS), ("own", POINTS)),
        context=("game_expected_team_points", "matchup_expected_pace"),
    ),
    REBOUNDS: AnchorSpec(
        squads=(("own", REBOUNDS), ("opp", REBOUNDS)),
        context=(
            "team_oreb_per_game",
            "team_dreb_per_game",
            "opp_oreb_per_game",
            "opp_dreb_per_game",
            "matchup_expected_pace",
            "game_expected_total",
        ),
    ),
    THREES: AnchorSpec(
        squads=(("own", THREES),),
        context=("team_fg3a_per_game", "matchup_expected_pace"),
    ),
}

#: A factor outside this range means the anchor and the players disagree about
#: the game so badly that one of them is broken. Held, not trusted.
FACTOR_BOUNDS = (0.6, 1.6)

RECONCILIATION_FILE = Path("reconciliation") / "stats.json"


def stats_reconciliation_path(root: Path | None = None) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    return (root or DEFAULT_ARTIFACT_ROOT) / RECONCILIATION_FILE


# --------------------------------------------------------------------------
# Sides and their inputs
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Sides:
    """Rows grouped into the two squads of each fixture."""

    #: Side label for each row.
    labels: list[Hashable]
    #: Row indices per side.
    members: dict[Hashable, list[int]]
    #: The other side of the same fixture, if it has candidates.
    opponent: dict[Hashable, Hashable | None]
    #: Team context per side, from its first row.
    context: dict[Hashable, dict[str, Any]]

    @classmethod
    def build(
        cls,
        labels: Sequence[Hashable],
        fixtures: Sequence[Hashable],
        features: Sequence[dict[str, Any]],
    ) -> Sides:
        members: dict[Hashable, list[int]] = defaultdict(list)
        by_fixture: dict[Hashable, list[Hashable]] = defaultdict(list)
        context: dict[Hashable, dict[str, Any]] = {}
        for index, (label, fixture) in enumerate(zip(labels, fixtures, strict=True)):
            if label not in members:
                by_fixture[fixture].append(label)
                context[label] = features[index]
            members[label].append(index)
        opponent: dict[Hashable, Hashable | None] = {}
        for pair in by_fixture.values():
            for label in pair:
                others = [other for other in pair if other != label]
                opponent[label] = others[0] if others else None
        return cls(list(labels), dict(members), opponent, context)


def squad_sums(
    values: Sequence[float], plays: Sequence[float], sides: Sides
) -> dict[Hashable, float]:
    """Each side's expected total: values weighted by P(plays)."""
    return {
        label: sum(float(plays[i]) * float(values[i]) for i in indices)
        for label, indices in sides.members.items()
    }


def design_rows(
    spec: AnchorSpec,
    sums: dict[PredictionTarget, dict[Hashable, float]],
    sides: Sides,
) -> dict[Hashable, list[float] | None]:
    """Regressors per side, intercept first. ``None`` where any input is missing."""
    out: dict[Hashable, list[float] | None] = {}
    for label in sides.members:
        row: list[float] = [1.0]
        complete = True
        for which, target in spec.squads:
            other = label if which == "own" else sides.opponent.get(label)
            value = sums[target].get(other) if other is not None else None
            if value is None:
                complete = False
                break
            row.append(float(value))
        if complete:
            for name in spec.context:
                value = sides.context[label].get(name)
                if value is None or not math.isfinite(float(value)):
                    complete = False
                    break
                row.append(float(value))
        out[label] = row if complete else None
    return out


def fit_anchor(
    design: dict[Hashable, list[float] | None], actual: dict[Hashable, float]
) -> list[float] | None:
    """Least-squares coefficients over sides with every input, or ``None``."""
    import numpy as np

    keys = [k for k, row in design.items() if row is not None and k in actual]
    if not keys:
        return None
    x = np.array([design[k] for k in keys], dtype=float)
    if len(keys) <= x.shape[1] * 5:
        return None  # too few squads to trust the coefficients
    y = np.array([actual[k] for k in keys], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return [float(b) for b in beta]


def side_factors(
    design: dict[Hashable, list[float] | None],
    own: dict[Hashable, float],
    coefficients: Sequence[float] | None,
    *,
    alpha: float,
) -> dict[Hashable, float]:
    """``(A / S) ** alpha`` per side, 1.0 where the anchor cannot be formed."""
    lo, hi = FACTOR_BOUNDS
    out: dict[Hashable, float] = {}
    for label, row in design.items():
        current = own.get(label, 0.0)
        if coefficients is None or row is None or current <= 0 or alpha <= 0:
            out[label] = 1.0
            continue
        anchor = max(0.0, sum(c * x for c, x in zip(coefficients, row, strict=True)))
        out[label] = min(hi, max(lo, (anchor / current) ** alpha)) if anchor > 0 else 1.0
    return out


def scale(values: Sequence[float], sides: Sides, factors: dict[Hashable, float]) -> list[float]:
    return [
        float(v) * factors.get(label, 1.0) for v, label in zip(values, sides.labels, strict=True)
    ]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(
    values: Sequence[float],
    plays: Sequence[float],
    actual: Sequence[float | None],
    sides: Sides,
    *,
    alpha: float,
) -> dict[str, float]:
    """Player RMSE over those who played; squad error of the P(plays)-weighted sum."""
    errors = [v - a for v, a in zip(values, actual, strict=True) if a is not None]
    projected = squad_sums(values, plays, sides)
    real = {
        label: sum(actual[i] or 0.0 for i in indices) for label, indices in sides.members.items()
    }
    squad = [projected[k] - real[k] for k in projected]
    nan = float("nan")
    return {
        "alpha": alpha,
        "players": len(errors),
        "player_rmse": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4)
        if errors
        else nan,
        "player_bias": round(sum(errors) / len(errors), 4) if errors else nan,
        "team_games": len(squad),
        "squad_mae": round(sum(abs(e) for e in squad) / len(squad), 3) if squad else nan,
        "squad_bias": round(sum(squad) / len(squad), 3) if squad else nan,
    }


def paired_squared_delta(
    before: Sequence[float],
    after: Sequence[float],
    actual: Sequence[float | None],
    sides: Sides,
) -> dict[str, float]:
    """Mean change in squared error, players paired with themselves, SE clustered by squad."""
    totals: dict[Hashable, float] = defaultdict(float)
    counts: dict[Hashable, int] = defaultdict(int)
    for b, a, y, label in zip(before, after, actual, sides.labels, strict=True):
        if y is None:
            continue
        totals[label] += (a - y) ** 2 - (b - y) ** 2
        counts[label] += 1
    n = sum(counts.values())
    clusters = len(totals)
    if n == 0 or clusters < 2:
        return {"mean": float("nan"), "se": float("nan"), "players": n, "squads": clusters}
    mean = sum(totals.values()) / n
    spread = sum((totals[k] - mean * counts[k]) ** 2 for k in totals)
    se = math.sqrt(spread * clusters / (clusters - 1)) / n
    return {"mean": round(mean, 5), "se": round(se, 5), "players": n, "squads": clusters}


@dataclass(slots=True)
class StatFit:
    target: PredictionTarget
    alpha: float
    coefficients: list[float] | None
    inputs: list[str]
    validation: list[dict[str, float]]
    test_unscaled: dict[str, float]
    test_scaled: dict[str, float]
    player_delta: dict[str, float]
    note: str = ""

    @property
    def players_significantly_worse(self) -> bool:
        mean, se = self.player_delta.get("mean", math.nan), self.player_delta.get("se", math.nan)
        if not (math.isfinite(mean) and math.isfinite(se)):
            return True
        return mean > 2.0 * se

    @property
    def enabled(self) -> bool:
        base, new = self.test_unscaled, self.test_scaled
        if self.alpha <= 0 or self.coefficients is None:
            return False
        if not all(math.isfinite(v) for v in (base["squad_mae"], new["squad_mae"])):
            return False
        return new["squad_mae"] < base["squad_mae"] and not self.players_significantly_worse

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "coefficients": self.coefficients,
            "inputs": self.inputs,
            "validation": self.validation,
            "test": {
                "unscaled": self.test_unscaled,
                "scaled": self.test_scaled,
                "player_delta": self.player_delta,
                "players_significantly_worse": self.players_significantly_worse,
            },
            "note": self.note,
        }


def fit_stat(
    target: PredictionTarget,
    *,
    values: dict[PredictionTarget, list[float]],
    plays: Sequence[float],
    actual: dict[PredictionTarget, list[float | None]],
    sides: Sides,
    fit_mask: Sequence[bool],
    grid: Sequence[float] = ALPHA_GRID,
) -> StatFit:
    """Fit one stat's anchor and strength on validation rows, judge it on test rows.

    ``fit_mask`` marks validation rows; the rest are test. Sides never straddle
    the boundary, because both squads of a fixture share a tip-off.
    """
    spec = SPECS[target]
    sums = {t: squad_sums(values[t], plays, sides) for t in {t for _, t in spec.squads}}
    design = design_rows(spec, sums, sides)
    realised = {
        label: sum(actual[target][i] or 0.0 for i in indices)
        for label, indices in sides.members.items()
    }
    fit_sides = {label for label, indices in sides.members.items() if fit_mask[indices[0]]}
    coefficients = fit_anchor(
        {k: v for k, v in design.items() if k in fit_sides},
        {k: v for k, v in realised.items() if k in fit_sides},
    )

    subsets: dict[bool, tuple[list[int], Sides]] = {}
    for mask_value in (True, False):
        idx = [i for i, m in enumerate(fit_mask) if bool(m) is mask_value]
        # Evaluation needs only who is in which squad, not who they played.
        subsets[mask_value] = (
            idx,
            Sides.build(
                [sides.labels[i] for i in idx],
                [sides.labels[i] for i in idx],
                [sides.context[sides.labels[i]] for i in idx],
            ),
        )

    def subset(mask_value: bool) -> tuple[list[int], Sides]:
        return subsets[mask_value]

    def run(alpha: float, mask_value: bool) -> tuple[list[float], dict[str, float]]:
        factors = side_factors(design, sums[target], coefficients, alpha=alpha)
        scaled = scale(values[target], sides, factors)
        idx, part = subset(mask_value)
        metrics = evaluate(
            [scaled[i] for i in idx],
            [plays[i] for i in idx],
            [actual[target][i] for i in idx],
            part,
            alpha=alpha,
        )
        return scaled, metrics

    table = [run(a, True)[1] for a in grid]
    reference = table[0]["player_rmse"]
    eligible = [
        row
        for row in table
        if math.isfinite(row["player_rmse"]) and row["player_rmse"] <= reference + 1e-9
    ]
    alpha = float(max(eligible, key=lambda row: row["alpha"])["alpha"]) if eligible else 0.0
    if coefficients is None:
        alpha = 0.0

    unscaled_values, base = run(0.0, False)
    scaled_values, scaled = run(alpha, False)
    idx, part = subset(False)
    delta = paired_squared_delta(
        [unscaled_values[i] for i in idx],
        [scaled_values[i] for i in idx],
        [actual[target][i] for i in idx],
        part,
    )
    return StatFit(
        target=target,
        alpha=alpha,
        coefficients=coefficients,
        inputs=spec.names(),
        validation=table,
        test_unscaled=base,
        test_scaled=scaled,
        player_delta=delta,
        note="" if coefficients is not None else "not enough complete squads to fit the anchor",
    )


# --------------------------------------------------------------------------
# Fitting on history
# --------------------------------------------------------------------------


async def _actual_stats(
    session: AsyncSession, game_ids: Sequence[uuid.UUID], targets: Sequence[PredictionTarget]
) -> dict[tuple[uuid.UUID, uuid.UUID], dict[PredictionTarget, float]]:
    columns = [getattr(PlayerGameStats, TARGET_COLUMNS[t]) for t in targets]
    out: dict[tuple[uuid.UUID, uuid.UUID], dict[PredictionTarget, float]] = {}
    ids = list(game_ids)
    for start in range(0, len(ids), 1000):
        chunk = ids[start : start + 1000]
        for game_id, player_id, *stats in (
            await session.execute(
                select(GameTeam.game_id, PlayerGameStats.player_id, *columns)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                .where(GameTeam.game_id.in_(chunk), PlayerGameStats.did_not_play.is_(False))
            )
        ).all():
            out[(game_id, player_id)] = {
                t: float(v or 0) for t, v in zip(targets, stats, strict=True)
            }
    return out


async def stat_periods(
    session: AsyncSession, rows: Sequence[AvailabilityRow], *, feature_set_version: str
) -> tuple[datetime, datetime]:
    """``(fit_start, test_start)``, out-of-sample for every stat model and P(plays)."""
    from euroleague_ai.ml.availability import chronological_split as plays_split
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.train import chronological_split as stat_split

    _, p_val, p_test = plays_split(rows)
    if not (p_val and p_test):
        raise ValueError("not enough history to form validation and test periods")
    fit_start, test_start = p_val[0].tip_off_utc, p_test[0].tip_off_utc
    for target in STAT_TARGETS:
        training = await load_training_rows(
            session, target=target, feature_set_version=feature_set_version
        )
        _, val, test = stat_split(training)
        if not (val and test):
            raise ValueError(f"not enough {target.name} history for validation and test")
        fit_start = max(fit_start, val[0].tip_off_utc)
        test_start = max(test_start, test[0].tip_off_utc)
    return fit_start, test_start


@dataclass(slots=True)
class StatsReconciliationFit:
    fits: dict[PredictionTarget, StatFit]
    model_ids: dict[str, str]
    model_labels: dict[str, str]
    periods: dict[str, str]
    skipped: dict[str, str] = field(default_factory=dict)
    fitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ids": self.model_ids,
            "models": self.model_labels,
            "periods": self.periods,
            "factor_bounds": list(FACTOR_BOUNDS),
            "targets": {t.name: f.to_dict() for t, f in self.fits.items()},
            "skipped": self.skipped,
            "fitted_at": self.fitted_at,
        }


async def fit_stats_reconciliation(
    session: AsyncSession,
    *,
    models: dict[PredictionTarget, LoadedModel],
    settings: Settings | None = None,
    fit_start: datetime | None = None,
    test_start: datetime | None = None,
    artifact_root: Path | None = None,
    save: bool = True,
    now: datetime | None = None,
) -> StatsReconciliationFit:
    """Fit, gate and (by default) save squad reconciliation for every stat."""
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.availability import load_availability_rows

    plays_model = models.get(PredictionTarget.PLAYS)
    if plays_model is None:
        raise ValueError("needs an active PLAYS model; run `ml train` first")
    missing = [t.name for t in STAT_TARGETS if t not in models]
    if missing:
        raise ValueError(f"needs active models for {', '.join(missing)}; run `ml train` first")

    rows, _ = await load_availability_rows(
        session, feature_set_version=FEATURE_SET_VERSION, settings=settings
    )
    if not rows:
        raise ValueError("no candidate rows; run `features backfill` first")
    if fit_start is None or test_start is None:
        fit_start, test_start = await stat_periods(
            session, rows, feature_set_version=FEATURE_SET_VERSION
        )
    in_scope = [row for row in rows if row.tip_off_utc >= fit_start]
    if not in_scope:
        raise ValueError("validation and test periods are empty")
    features = [row.features for row in in_scope]
    fit_mask = [row.tip_off_utc < test_start for row in in_scope]
    if all(fit_mask) or not any(fit_mask):
        raise ValueError("validation or test period is empty")

    actual_by_row = await _actual_stats(
        session, sorted({row.game_id for row in in_scope}), STAT_TARGETS
    )
    plays = plays_model.predict_batch(features)
    sides = Sides.build(
        [(row.game_id, row.team_id) for row in in_scope],
        [row.game_id for row in in_scope],
        features,
    )

    values: dict[PredictionTarget, list[float]] = {}
    actual: dict[PredictionTarget, list[float | None]] = {}
    skipped: dict[str, str] = {}
    for target in STAT_TARGETS:
        model = models[target]
        medians = model.predict_batch(features)
        expected = model.expected_batch(features, medians)
        values[target] = [e if e is not None else m for e, m in zip(expected, medians, strict=True)]
        if any(e is None for e in expected):
            skipped[target.name] = "no published expected values (its expectation gate failed)"
        actual[target] = [
            actual_by_row.get((row.game_id, row.player_id), {}).get(target) if row.played else None
            for row in in_scope
        ]
        # A player who played but has no box-score line is treated as zero.
        actual[target] = [
            (a if a is not None else 0.0) if row.played else None
            for a, row in zip(actual[target], in_scope, strict=True)
        ]

    fits: dict[PredictionTarget, StatFit] = {}
    for target in STAT_TARGETS:
        if target.name in skipped:
            continue
        fits[target] = fit_stat(
            target, values=values, plays=plays, actual=actual, sides=sides, fit_mask=fit_mask
        )

    def label(model: LoadedModel) -> str:
        return f"{model.model_version.name} {model.model_version.version}"

    involved = (*STAT_TARGETS, PredictionTarget.PLAYS)
    result = StatsReconciliationFit(
        fits=fits,
        model_ids={t.name: str(models[t].model_version.id) for t in involved},
        model_labels={t.name: label(models[t]) for t in involved},
        periods={"fit_start": fit_start.isoformat(), "test_start": test_start.isoformat()},
        skipped=skipped,
        fitted_at=(now.isoformat() if now else ""),
    )
    logger.info(
        "fitted squad stat reconciliation",
        extra={t.name: f.enabled for t, f in fits.items()},
    )
    if save:
        path = stats_reconciliation_path(artifact_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatsReconciliation:
    """The enabled part of a saved fit, as inference uses it."""

    #: target -> (alpha, coefficients)
    targets: dict[PredictionTarget, tuple[float, list[float]]]
    model_ids: dict[str, str]

    def applies_to(self, models: dict[PredictionTarget, LoadedModel]) -> bool:
        """Only for the exact model versions it was fitted on."""
        for name, model_id in self.model_ids.items():
            model = models.get(PredictionTarget[name])
            if model is None or str(model.model_version.id) != model_id:
                return False
        return True

    def factors(
        self,
        base: Mapping[PredictionTarget, Sequence[float]],
        plays: Sequence[float],
        sides: Sequence[Hashable],
        features: Sequence[dict[str, Any]],
    ) -> dict[PredictionTarget, list[float]]:
        """Per-row factor for each enabled stat. Every input is the unreconciled value."""
        grouped = Sides.build(list(sides), [0] * len(sides), list(features))
        sums = {t: squad_sums(v, plays, grouped) for t, v in base.items()}
        out: dict[PredictionTarget, list[float]] = {}
        for target, (alpha, coefficients) in self.targets.items():
            spec = SPECS[target]
            if any(t not in sums for _, t in spec.squads):
                continue
            design = design_rows(spec, sums, grouped)
            by_side = side_factors(design, sums[target], coefficients, alpha=alpha)
            out[target] = [by_side.get(label, 1.0) for label in grouped.labels]
        return out


def load_stats_reconciliation(root: Path | None = None) -> StatsReconciliation | None:
    """The saved fit's enabled stats, or ``None`` if there are none."""
    path = stats_reconciliation_path(root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = {
        PredictionTarget[name]: (float(entry["alpha"]), [float(c) for c in entry["coefficients"]])
        for name, entry in payload.get("targets", {}).items()
        if entry.get("enabled") and entry.get("coefficients")
    }
    if not targets:
        return None
    return StatsReconciliation(targets=targets, model_ids=dict(payload["model_ids"]))

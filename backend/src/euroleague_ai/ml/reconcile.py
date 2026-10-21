"""Squad minutes that add up: post-game-style reconciliation, before the game.

**The problem.** Every player's minutes are predicted on their own. A squad's
projected minutes therefore add up to whatever they add up to, and nothing ties
them to the one hard fact about a basketball game: a team plays 200 minutes in
regulation, a little more with overtime. Model 0.1.10's minutes expectation failed
its gate precisely because squad totals got *worse* (+1.7 to +2.0 per team-game),
and minutes are the input every other line depends on.

**The fix.** For each side of a fixture, scale every candidate's minutes by one
factor so that the squad's expected minutes, each weighted by his chance of
playing, land on the squad total real teams play:

    s     = (T / sum_i P(plays)_i * m_i) ** alpha
    m_i'  = min(cap, m_i * s)

``T`` is measured, not assumed: the average minutes real squads' candidates
actually logged in the fit period (just over 200, because of overtime). ``alpha``
is how hard to pull, from 0 (off) to 1 (sum exactly to T). Multiplicative, so a
30-minute starter absorbs more of a correction than a 6-minute reserve, which is
how coaches redistribute minutes too.

**Fitting honestly.** Both models' held-out periods are recomputed from the data
(the later of the two validation starts, and of the two test starts), so every
row scored here is out-of-sample for *both* the minutes model and P(plays).
``alpha`` is chosen on the validation period and the result is reported on the
test period.

**Which alpha.** The strongest one whose validation player minutes MAE is no
worse than not scaling at all. The first version picked the lowest player MAE
instead, and on real data that was the wrong question: player MAE is almost
flat across strengths (4.574 at best, 4.588 unscaled, 4.597 at full strength
on model 0.1.10) while squad error falls steadily (8.9 to 2.7 minutes). The
minimum sat at 0.4 by a hundredth of a minute and left most of the squad fix
unused. Squads that add up are the point; player accuracy is the constraint.

**The gate.** Enabled only if, on test, player minutes are not *significantly*
worse *and* the squad-total error gets smaller. "Significantly" is a paired
comparison: the same players in the same games, scaled against unscaled, with a
standard error clustered by squad, because players in one game are not
independent. Worse by more than two standard errors fails.

The first version demanded "no worse at all", and real data showed why that is
the wrong test: at alpha 0.8 squad error fell from 9.74 to 4.02 minutes while
player MAE moved from 4.380 to 4.382. Two thousandths of a minute, averaged
over tens of thousands of players, is noise, and a zero-tolerance gate let it
veto a large, real improvement. Otherwise the fit is saved with
``enabled: false`` and predictions are untouched. The fit records which minutes
and P(plays) model versions it was made for; after a retrain it no longer
applies until ``ml reconcile`` is run again.

**What it changes.** Only the minutes *expected value*. The median, its interval
and threshold probabilities stay as the model produced them, because those are
what scoring and calibration are judged on.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.enums import PredictionTarget
from euroleague_ai.models.game import GameTeam, PlayerGameStats

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Hashable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.ml.availability import AvailabilityRow
    from euroleague_ai.ml.inference import LoadedModel

logger = get_logger(__name__)

#: No projection goes above a full regulation game after scaling.
MINUTES_CAP = 40.0

#: Strengths tried, from untouched to fully reconciled.
ALPHA_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))

#: Where the fit is written, relative to the model artifact root.
RECONCILIATION_FILE = Path("reconciliation") / "minutes.json"


def reconciliation_path(root: Path | None = None) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    return (root or DEFAULT_ARTIFACT_ROOT) / RECONCILIATION_FILE


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def scale_sides(
    minutes: Sequence[float],
    plays: Sequence[float],
    sides: Sequence[Hashable],
    *,
    target: float,
    alpha: float,
    cap: float = MINUTES_CAP,
) -> list[float]:
    """Scale each side's minutes toward ``target``, one factor per side.

    ``sides`` labels which squad each row belongs to; rows sharing a label are
    scaled together. Each side aims for a weighted sum of
    ``current * (target / current) ** alpha``. A player pushed past ``cap`` is
    held there and the rest of his side is scaled a little further to make up
    the difference, so a cap never quietly loses minutes. A side whose weighted
    sum is zero is left alone rather than divided by zero.
    """
    out = [float(m) for m in minutes]
    by_side: dict[Hashable, list[int]] = defaultdict(list)
    for index, side in enumerate(sides):
        by_side[side].append(index)

    for indices in by_side.values():
        current = sum(float(plays[i]) * out[i] for i in indices)
        if current <= 0:
            continue
        goal = current * (target / current) ** alpha
        capped: set[int] = set()
        # A few passes always suffice: each one caps at least one more player
        # or finishes. Bounded so a degenerate input cannot spin.
        for _ in range(len(indices) + 1):
            fixed = sum(float(plays[i]) * cap for i in capped)
            free = [i for i in indices if i not in capped]
            free_sum = sum(float(plays[i]) * float(minutes[i]) for i in free)
            if not free or free_sum <= 0:
                break
            factor = max(0.0, goal - fixed) / free_sum
            newly = [i for i in free if float(minutes[i]) * factor > cap]
            for i in free:
                out[i] = min(cap, float(minutes[i]) * factor)
            for i in capped:
                out[i] = cap
            if not newly:
                break
            capped.update(newly)
    return out


@dataclass(frozen=True, slots=True)
class CandidateMinutes:
    """One candidate in one historical squad, with everything the fit needs."""

    side: tuple[Any, Any]
    predicted: float
    plays: float
    #: Minutes he actually played, ``None`` if he did not take the floor.
    actual: float | None


def evaluate(
    candidates: Sequence[CandidateMinutes], *, target: float, alpha: float
) -> dict[str, float]:
    """Player and squad minutes error at one strength.

    Player error is over candidates who played (conditional minutes against what
    they played). Squad error compares the P(plays)-weighted sum with what the
    squad's candidates actually logged.
    """
    scaled = scale_sides(
        [c.predicted for c in candidates],
        [c.plays for c in candidates],
        [c.side for c in candidates],
        target=target,
        alpha=alpha,
    )
    errors = [s - c.actual for s, c in zip(scaled, candidates, strict=True) if c.actual is not None]
    projected: dict[Any, float] = defaultdict(float)
    actual: dict[Any, float] = defaultdict(float)
    for s, c in zip(scaled, candidates, strict=True):
        projected[c.side] += c.plays * s
        actual[c.side] += c.actual or 0.0
    squad = [projected[k] - actual[k] for k in projected]
    nan = float("nan")
    return {
        "alpha": alpha,
        "players": len(errors),
        "player_mae": round(sum(abs(e) for e in errors) / len(errors), 4) if errors else nan,
        "player_bias": round(sum(errors) / len(errors), 4) if errors else nan,
        "team_games": len(squad),
        "squad_mae": round(sum(abs(e) for e in squad) / len(squad), 3) if squad else nan,
        "squad_bias": round(sum(squad) / len(squad), 3) if squad else nan,
    }


def paired_player_delta(
    candidates: Sequence[CandidateMinutes], *, target: float, alpha: float
) -> dict[str, float]:
    """How much scaling changes each player's absolute error, with a fair error bar.

    ``mean`` is scaled MAE minus unscaled MAE over the same players (positive
    means scaling made players worse). ``se`` is its standard error, clustered
    by squad: one squad's players share one scaling factor, so their changes
    move together, and treating them as independent would make the error bar
    far too narrow.
    """
    scaled = scale_sides(
        [c.predicted for c in candidates],
        [c.plays for c in candidates],
        [c.side for c in candidates],
        target=target,
        alpha=alpha,
    )
    totals: dict[Any, float] = defaultdict(float)
    counts: dict[Any, int] = defaultdict(int)
    for s, c in zip(scaled, candidates, strict=True):
        if c.actual is None:
            continue
        totals[c.side] += abs(s - c.actual) - abs(c.predicted - c.actual)
        counts[c.side] += 1
    n = sum(counts.values())
    clusters = len(totals)
    if n == 0 or clusters < 2:
        return {"mean": float("nan"), "se": float("nan"), "players": n, "squads": clusters}
    mean = sum(totals.values()) / n
    # Ratio-estimator variance over clusters.
    spread = sum((totals[k] - mean * counts[k]) ** 2 for k in totals)
    se = math.sqrt(spread * clusters / (clusters - 1)) / n
    return {"mean": round(mean, 5), "se": round(se, 5), "players": n, "squads": clusters}


def squad_target(candidates: Sequence[CandidateMinutes]) -> float:
    """Average minutes a squad's candidates actually logged, per team-game."""
    totals: dict[Any, float] = defaultdict(float)
    for c in candidates:
        totals[c.side] += c.actual or 0.0
    return sum(totals.values()) / len(totals) if totals else 200.0


@dataclass(slots=True)
class ReconciliationFit:
    """The chosen strength, its evidence, and whether it passed."""

    alpha: float
    target_minutes: float
    validation: list[dict[str, float]]
    test_unscaled: dict[str, float]
    test_scaled: dict[str, float]
    #: Paired change in player MAE on test, from :func:`paired_player_delta`.
    player_delta: dict[str, float]
    minutes_model_id: str
    minutes_model: str
    plays_model_id: str
    plays_model: str
    periods: dict[str, str]
    cap: float = MINUTES_CAP
    fitted_at: str = ""

    @property
    def players_significantly_worse(self) -> bool:
        """Scaled player MAE above unscaled by more than two standard errors."""
        mean, se = self.player_delta.get("mean", math.nan), self.player_delta.get("se", math.nan)
        if not (math.isfinite(mean) and math.isfinite(se)):
            return True  # cannot tell, so do not risk it
        return mean > 2.0 * se

    @property
    def enabled(self) -> bool:
        """**The gate.** Squads improve on test and players are not significantly worse."""
        base, new = self.test_unscaled, self.test_scaled
        if self.alpha <= 0 or not all(
            math.isfinite(v) for v in (base["squad_mae"], new["squad_mae"])
        ):
            return False
        return new["squad_mae"] < base["squad_mae"] and not self.players_significantly_worse

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "target_minutes": round(self.target_minutes, 3),
            "cap": self.cap,
            "minutes_model_id": self.minutes_model_id,
            "minutes_model": self.minutes_model,
            "plays_model_id": self.plays_model_id,
            "plays_model": self.plays_model,
            "periods": self.periods,
            "validation": self.validation,
            "test": {
                "unscaled": self.test_unscaled,
                "scaled": self.test_scaled,
                "player_delta": self.player_delta,
                "players_significantly_worse": self.players_significantly_worse,
            },
            "fitted_at": self.fitted_at,
        }


def choose_alpha(
    fit_candidates: Sequence[CandidateMinutes],
    test_candidates: Sequence[CandidateMinutes],
    *,
    target: float,
    grid: Sequence[float] = ALPHA_GRID,
) -> tuple[float, list[dict[str, float]], dict[str, float], dict[str, float]]:
    """Pick the strength on validation, measure it on test.

    The strongest strength whose validation player MAE is no worse than
    unscaled (``grid[0]``, which must be 0). If none is, 0: off.
    """
    if not grid or grid[0] != 0.0:
        raise ValueError("the grid must start at 0.0, the unscaled reference")
    table = [evaluate(fit_candidates, target=target, alpha=a) for a in grid]
    reference = table[0]["player_mae"]
    eligible = [
        row
        for row in table
        if math.isfinite(row["player_mae"]) and row["player_mae"] <= reference + 1e-9
    ]
    alpha = float(max(eligible, key=lambda row: row["alpha"])["alpha"]) if eligible else 0.0
    return (
        alpha,
        table,
        evaluate(test_candidates, target=target, alpha=0.0),
        evaluate(test_candidates, target=target, alpha=alpha),
    )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


async def _actual_minutes(
    session: AsyncSession, game_ids: Sequence[uuid.UUID]
) -> dict[tuple[uuid.UUID, uuid.UUID], float]:
    """``(game, player) -> minutes played`` for players who took the floor."""
    out: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
    ids = list(game_ids)
    for start in range(0, len(ids), 1000):
        chunk = ids[start : start + 1000]
        for game_id, player_id, seconds in (
            await session.execute(
                select(GameTeam.game_id, PlayerGameStats.player_id, PlayerGameStats.seconds_played)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                .where(GameTeam.game_id.in_(chunk), PlayerGameStats.did_not_play.is_(False))
            )
        ).all():
            out[(game_id, player_id)] = (seconds or 0) / 60.0
    return out


def _predict(
    rows: Sequence[AvailabilityRow],
    minutes_model: LoadedModel,
    plays_model: LoadedModel,
    actual: dict[tuple[uuid.UUID, uuid.UUID], float],
) -> list[CandidateMinutes]:
    """Run both active models over every historical candidate."""
    features = [row.features for row in rows]
    medians = minutes_model.predict_batch(features)
    expected = minutes_model.expected_batch(features, medians)
    base = [e if e is not None else m for e, m in zip(expected, medians, strict=True)]
    plays = plays_model.predict_batch(features)
    return [
        CandidateMinutes(
            side=(row.game_id, row.team_id),
            predicted=b,
            plays=p,
            actual=actual.get((row.game_id, row.player_id)) if row.played else None,
        )
        for row, b, p in zip(rows, base, plays, strict=True)
    ]


async def default_periods(
    session: AsyncSession,
    rows: Sequence[AvailabilityRow],
    *,
    feature_set_version: str,
) -> tuple[datetime, datetime]:
    """``(fit_start, test_start)``, out-of-sample for both models.

    Recomputes each model's chronological split from today's data and takes the
    later boundary of each pair. Data ingested since training only pushes the
    boundaries later, which keeps every scored row unseen by both models.
    """
    from euroleague_ai.ml.availability import chronological_split as plays_split
    from euroleague_ai.ml.dataset import load_training_rows
    from euroleague_ai.ml.train import chronological_split as stat_split

    minutes_rows = await load_training_rows(
        session, target=PredictionTarget.MINUTES, feature_set_version=feature_set_version
    )
    _, m_val, m_test = stat_split(minutes_rows)
    _, p_val, p_test = plays_split(rows)
    if not (m_val and m_test and p_val and p_test):
        raise ValueError("not enough history to form validation and test periods")
    return (
        max(m_val[0].tip_off_utc, p_val[0].tip_off_utc),
        max(m_test[0].tip_off_utc, p_test[0].tip_off_utc),
    )


async def fit_minutes_reconciliation(
    session: AsyncSession,
    *,
    models: dict[PredictionTarget, LoadedModel],
    settings: Settings | None = None,
    fit_start: datetime | None = None,
    test_start: datetime | None = None,
    artifact_root: Path | None = None,
    save: bool = True,
    now: datetime | None = None,
) -> ReconciliationFit:
    """Fit, gate and (by default) save the minutes reconciliation.

    Args:
        models: The loaded active models; needs ``MINUTES`` and ``PLAYS``.
        fit_start, test_start: Override the period boundaries (tests do).
    """
    from euroleague_ai.features.registry import FEATURE_SET_VERSION
    from euroleague_ai.ml.availability import load_availability_rows

    minutes_model = models.get(PredictionTarget.MINUTES)
    plays_model = models.get(PredictionTarget.PLAYS)
    if minutes_model is None or plays_model is None:
        raise ValueError("needs active MINUTES and PLAYS models; run `ml train` first")

    rows, _ = await load_availability_rows(
        session, feature_set_version=FEATURE_SET_VERSION, settings=settings
    )
    if not rows:
        raise ValueError("no candidate rows; run `features backfill` first")
    if fit_start is None or test_start is None:
        fit_start, test_start = await default_periods(
            session, rows, feature_set_version=FEATURE_SET_VERSION
        )

    actual = await _actual_minutes(session, sorted({row.game_id for row in rows}))
    in_scope = [row for row in rows if row.tip_off_utc >= fit_start]
    candidates = _predict(in_scope, minutes_model, plays_model, actual)
    fit_set = [c for c, r in zip(candidates, in_scope, strict=True) if r.tip_off_utc < test_start]
    test_set = [c for c, r in zip(candidates, in_scope, strict=True) if r.tip_off_utc >= test_start]
    if not fit_set or not test_set:
        raise ValueError("validation or test period is empty")

    target = squad_target(fit_set)
    alpha, table, base, scaled = choose_alpha(fit_set, test_set, target=target)
    delta = paired_player_delta(test_set, target=target, alpha=alpha)

    def label(model: LoadedModel) -> str:
        version = model.model_version
        return f"{version.name} {version.version}"

    fit = ReconciliationFit(
        alpha=alpha,
        target_minutes=target,
        validation=table,
        test_unscaled=base,
        test_scaled=scaled,
        player_delta=delta,
        minutes_model_id=str(minutes_model.model_version.id),
        minutes_model=label(minutes_model),
        plays_model_id=str(plays_model.model_version.id),
        plays_model=label(plays_model),
        periods={"fit_start": fit_start.isoformat(), "test_start": test_start.isoformat()},
        fitted_at=(now.isoformat() if now else ""),
    )
    logger.info("fitted minutes reconciliation", extra=fit.to_dict()["test"] | {"alpha": alpha})
    if save:
        path = reconciliation_path(artifact_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fit.to_dict(), indent=2), encoding="utf-8")
    return fit


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinutesReconciliation:
    """A saved fit, as inference uses it."""

    alpha: float
    target_minutes: float
    cap: float
    minutes_model_id: str
    plays_model_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def applies_to(self, models: dict[PredictionTarget, LoadedModel]) -> bool:
        """Only for the exact model versions it was fitted on."""
        minutes = models.get(PredictionTarget.MINUTES)
        plays = models.get(PredictionTarget.PLAYS)
        return (
            minutes is not None
            and plays is not None
            and str(minutes.model_version.id) == self.minutes_model_id
            and str(plays.model_version.id) == self.plays_model_id
        )

    def apply(
        self, minutes: Sequence[float], plays: Sequence[float], sides: Sequence[Hashable]
    ) -> list[float]:
        return scale_sides(
            minutes, plays, sides, target=self.target_minutes, alpha=self.alpha, cap=self.cap
        )


def load_reconciliation(root: Path | None = None) -> MinutesReconciliation | None:
    """The saved fit if it exists and passed its gate, else ``None``."""
    path = reconciliation_path(root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("enabled"):
        return None
    return MinutesReconciliation(
        alpha=float(payload["alpha"]),
        target_minutes=float(payload["target_minutes"]),
        cap=float(payload.get("cap", MINUTES_CAP)),
        minutes_model_id=str(payload["minutes_model_id"]),
        plays_model_id=str(payload["plays_model_id"]),
    )

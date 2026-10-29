"""Scoring published predictions against what actually happened.

Everything before this measures the model on held-out history. That is necessary
and it is not sufficient -- this project has already produced one model that
passed its offline gate while a calibration defect quietly degraded every target,
and the gate could not see it because the gate was computed from the same
damaged pipeline. Production truth is the only check that closes that loop.

Three things get measured, and they answer different questions:

**Point accuracy.** MAE against the baseline, on real fixtures. The offline
number is a forecast of this; drift between them is the signal.

**Interval calibration.** With distributions stored per prediction, we can ask
whether the published 80% interval actually contained 80% of outcomes. A model
whose point estimate is fine but whose intervals are too narrow is overconfident
in a way MAE will never reveal.

**Threshold calibration.** Of the players we said had a 30% chance of 15+, did
about 30% get there? This is the strongest available test of the probabilities,
because it needs no model at all -- just counting.

**On non-appearance.** A player can produce a line, dress and never be used, or
not be in the squad. The latter two score as zero and are flagged. They are
excluded from the headline error by default, because a points model should not
be blamed for a rotation decision, but they are counted and reported -- we did
publish those projections, and pretending otherwise would flatter the system.

The rotation decision is not left unscored, though. The ``PLAYS`` prediction
is scored against exactly that outcome: 1 if he took the floor, 0 otherwise,
with the squared error doubling as its Brier score.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.logging import get_logger
from euroleague_ai.ml.dataset import TARGET_COLUMNS, target_value
from euroleague_ai.models.competition import Season
from euroleague_ai.models.enums import GameStatus, PredictionTarget
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.prediction import ModelVersion, Prediction, PredictionOutcome

logger = get_logger(__name__)

#: Probability buckets for the threshold calibration table. Ten-point bands are
#: coarse enough to hold a usable sample after one round of fixtures and fine
#: enough to show a systematic tilt.
PROBABILITY_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
)


@dataclass(slots=True)
class ScoringResult:
    """What one scoring run did."""

    predictions_scored: int = 0
    already_scored: int = 0
    games: int = 0
    did_not_appear: int = 0
    targets: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions_scored": self.predictions_scored,
            "already_scored": self.already_scored,
            "games": self.games,
            "did_not_appear": self.did_not_appear,
            "targets": self.targets,
        }


def _finished_games(season_code: str | None, as_of: datetime) -> Select[Any]:
    """Games that have finished and could therefore be scored."""
    stmt = select(Game.id).where(Game.status == GameStatus.FINISHED, Game.tip_off_utc < as_of)
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    return stmt


def _only_active(stmt: Any, *, active_only: bool) -> Any:
    """Restrict to predictions made by the models active now.

    Season-long tables mix every model version since round 1, so a band that
    is off may only say an old model was off. ``--active`` judges today's.
    """
    if not active_only:
        return stmt
    return stmt.join(ModelVersion, ModelVersion.id == Prediction.model_version_id).where(
        ModelVersion.is_active.is_(True)
    )


async def score_predictions(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    game_codes: list[int] | None = None,
    clock: Clock | None = None,
    rescore: bool = False,
) -> ScoringResult:
    """Join realised box scores onto published predictions.

    Idempotent: a prediction already carrying an outcome is skipped unless
    ``rescore`` is set, so the job can run on a schedule without duplicating
    work or double-counting.

    Args:
        rescore: Recompute outcomes that already exist. Needed only when a box
            score has been corrected upstream.
    """
    clock = clock or SystemClock()
    now = clock.now()
    result = ScoringResult()

    games_stmt = _finished_games(season_code, now)
    if game_codes:
        games_stmt = games_stmt.where(Game.game_code.in_(game_codes))
    game_ids = list((await session.execute(games_stmt)).scalars().all())
    if not game_ids:
        logger.info("no finished games to score", extra={"season": season_code or "all"})
        return result
    result.games = len(game_ids)

    # PLAYS rides along with the stat targets. Its "column" is the box score
    # row's own id: all that matters is whether a playing line exists.
    scored_targets: list[tuple[PredictionTarget, Any]] = [
        (target, getattr(PlayerGameStats, column_name))
        for target, column_name in TARGET_COLUMNS.items()
    ]
    scored_targets.append((PredictionTarget.PLAYS, PlayerGameStats.id))

    for target, column in scored_targets:
        # LEFT joins throughout: a prediction for a player who never dressed has
        # no box score row at all, and that case must produce a scored zero
        # rather than vanishing from the report.
        stmt = (
            select(
                Prediction.id,
                Prediction.predicted_value,
                column,
                PlayerGameStats.did_not_play,
                PredictionOutcome.id,
            )
            .join(Game, Game.id == Prediction.game_id)
            # Straight to the box score line, without joining game_teams as a
            # table. Joining it directly would match both the home and the away
            # row for every game and fan each prediction out into two -- one
            # real and one all-NULL -- which reads as a non-appearance and would
            # have doubled the scored count while halving the apparent accuracy.
            .outerjoin(
                PlayerGameStats,
                (PlayerGameStats.player_id == Prediction.player_id)
                & (
                    PlayerGameStats.game_team_id.in_(
                        select(GameTeam.id).where(GameTeam.game_id == Prediction.game_id)
                    )
                ),
            )
            .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
            .where(Prediction.game_id.in_(game_ids), Prediction.target == target)
        )
        if not rescore:
            stmt = stmt.where(PredictionOutcome.id.is_(None))

        rows = (await session.execute(stmt)).all()
        scored = 0

        for prediction_id, predicted, raw_value, did_not_play, outcome_id in rows:
            if outcome_id is not None and not rescore:
                result.already_scored += 1
                continue

            absent = raw_value is None or bool(did_not_play)
            if target is PredictionTarget.PLAYS:
                # The outcome *is* the appearance. Squared error is then the
                # Brier score, so the existing columns already hold the right
                # numbers for a probability.
                actual = 0.0 if absent else 1.0
            else:
                actual = 0.0 if absent else target_value(target, raw_value)
            error = float(predicted) - actual

            await session.execute(
                pg_insert(PredictionOutcome)
                .values(
                    id=uuid.uuid4(),
                    prediction_id=prediction_id,
                    actual_value=actual,
                    absolute_error=abs(error),
                    squared_error=error * error,
                    did_not_appear=absent,
                    evaluated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["prediction_id"],
                    set_={
                        "actual_value": actual,
                        "absolute_error": abs(error),
                        "squared_error": error * error,
                        "did_not_appear": absent,
                        "evaluated_at": now,
                    },
                )
            )
            scored += 1
            if absent:
                result.did_not_appear += 1

        if scored:
            result.targets[str(target)] = scored
            result.predictions_scored += scored

    logger.info("scored predictions", extra=result.as_dict())
    return result


async def accuracy_report(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    include_absent: bool = False,
    active_only: bool = False,
) -> dict[str, Any]:
    """Realised accuracy and calibration, per target.

    Args:
        include_absent: Fold non-appearances into the error figures. Off by
            default -- see the module docstring -- but available, because
            "how wrong was everything we published" is also a fair question.
    """
    report: dict[str, Any] = {"season": season_code or "all", "targets": {}}

    for target in TARGET_COLUMNS:
        stmt = (
            select(
                func.count(PredictionOutcome.id),
                func.avg(PredictionOutcome.absolute_error),
                func.avg(PredictionOutcome.squared_error),
                func.avg(PredictionOutcome.actual_value),
                func.avg(Prediction.predicted_value - PredictionOutcome.actual_value),
            )
            .join(Prediction, Prediction.id == PredictionOutcome.prediction_id)
            .join(Game, Game.id == Prediction.game_id)
            .where(Prediction.target == target)
        )
        if season_code:
            stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
        if not include_absent:
            stmt = stmt.where(PredictionOutcome.did_not_appear.is_(False))
        stmt = _only_active(stmt, active_only=active_only)

        count, mae, mse, mean_actual, bias = (await session.execute(stmt)).one()
        if not count:
            continue

        report["targets"][str(target)] = {
            "scored": int(count),
            "mae": round(float(mae), 4),
            "rmse": round(float(mse) ** 0.5, 4),
            "mean_actual": round(float(mean_actual), 4),
            # Signed, and reported rather than corrected. Correcting it is what
            # broke the models the last time.
            "bias": round(float(bias), 4),
            "expected": await _expected_accuracy(
                session,
                target=target,
                season_code=season_code,
                include_absent=include_absent,
                active_only=active_only,
            ),
            "interval": await interval_coverage(
                session,
                target=target,
                season_code=season_code,
                include_absent=include_absent,
                active_only=active_only,
            ),
        }

    report["active_only"] = active_only
    report["thresholds"] = await threshold_calibration(
        session, season_code=season_code, include_absent=include_absent, active_only=active_only
    )
    report["thresholds_by_target"] = await threshold_calibration_by_target(
        session, season_code=season_code, include_absent=include_absent, active_only=active_only
    )
    report["thresholds_overall"] = await overall_threshold_calibration(
        session, season_code=season_code, active_only=active_only
    )
    report["availability"] = await availability_report(
        session, season_code=season_code, active_only=active_only
    )
    report["team_totals"] = await team_totals_report(
        session, season_code=season_code, active_only=active_only
    )
    return report


async def _expected_accuracy(
    session: AsyncSession,
    *,
    target: PredictionTarget,
    season_code: str | None,
    include_absent: bool,
    active_only: bool = False,
) -> dict[str, Any]:
    """RMSE and bias of the published expected values, where there are any.

    Judged on RMSE because the mean is what minimises squared error. Its bias
    should sit near zero, where the median's sits clearly below it.
    """
    stmt = (
        select(
            func.count(PredictionOutcome.id),
            func.avg(
                (Prediction.expected_value - PredictionOutcome.actual_value)
                * (Prediction.expected_value - PredictionOutcome.actual_value)
            ),
            func.avg(Prediction.expected_value - PredictionOutcome.actual_value),
        )
        .join(Prediction, Prediction.id == PredictionOutcome.prediction_id)
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.target == target, Prediction.expected_value.isnot(None))
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    if not include_absent:
        stmt = stmt.where(PredictionOutcome.did_not_appear.is_(False))
    stmt = _only_active(stmt, active_only=active_only)
    count, mse, bias = (await session.execute(stmt)).one()
    if not count:
        return {"scored": 0}
    return {
        "scored": int(count),
        "rmse": round(float(mse) ** 0.5, 4),
        "bias": round(float(bias), 4),
    }


#: Targets whose squad totals mean something. Minutes are included on purpose:
#: a squad's projected minutes should add up to roughly 200.
TEAM_TOTAL_TARGETS: tuple[PredictionTarget, ...] = (
    PredictionTarget.MINUTES,
    PredictionTarget.POINTS,
    PredictionTarget.REBOUNDS,
    PredictionTarget.ASSISTS,
    PredictionTarget.THREE_POINTERS_MADE,
)


def summarise_team_totals(
    projected: dict[str, dict[tuple[Any, Any], float]],
    actual: dict[tuple[Any, Any], float],
) -> dict[str, dict[str, float]]:
    """MAE and bias of summed squad totals, one entry per projection kind.

    Pure, so the arithmetic is unit-tested separately from the queries.
    """
    out: dict[str, dict[str, float]] = {}
    for kind, sums in projected.items():
        errors = [sums[key] - actual.get(key, 0.0) for key in sums]
        if not errors:
            continue
        out[kind] = {
            "team_games": len(errors),
            "mae": round(sum(abs(e) for e in errors) / len(errors), 3),
            "bias": round(sum(errors) / len(errors), 3),
        }
    return out


async def team_totals_report(
    session: AsyncSession, *, season_code: str | None = None, active_only: bool = False
) -> dict[str, Any]:
    """Do the projected squads add up to the real ones?

    For every scored team-game, three sums are compared with the squad's real
    total from the box score, *including* players we never projected:

    * ``median``: the published medians, summed over every projected player.
      What the board showed before expected values existed.
    * ``expected``: P(plays) x expected value given he plays, the realistic
      line. Falls back to the expected value alone when there is no
      play-probability, and to the median when there is no expected value.

    Only the newest prediction per player, game and target is used. Summing
    across model versions would count a player twice.
    """
    from euroleague_ai.models.features import PlayerFeatureSnapshot

    targets = (*TEAM_TOTAL_TARGETS, PredictionTarget.PLAYS)
    stmt = (
        select(
            Prediction.game_id,
            PlayerFeatureSnapshot.team_id,
            Prediction.player_id,
            Prediction.target,
            Prediction.predicted_value,
            Prediction.expected_value,
            Prediction.created_at,
        )
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(
            PlayerFeatureSnapshot,
            PlayerFeatureSnapshot.id == Prediction.player_feature_snapshot_id,
        )
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.target.in_(targets))
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    stmt = _only_active(stmt, active_only=active_only)

    latest: dict[tuple[Any, Any, Any], tuple[Any, float, float | None, Any]] = {}
    for game_id, team_id, player_id, target, median, expected, created in (
        await session.execute(stmt)
    ).all():
        prediction_key = (game_id, player_id, target)
        if prediction_key not in latest or created > latest[prediction_key][3]:
            latest[prediction_key] = (team_id, float(median), expected, created)
    if not latest:
        return {}

    plays = {
        (game_id, player_id): median
        for (game_id, player_id, target), (_t, median, _e, _c) in latest.items()
        if target is PredictionTarget.PLAYS
    }
    game_ids = {game_id for game_id, _p, _t in latest}

    report: dict[str, Any] = {}
    for target in TEAM_TOTAL_TARGETS:
        column = getattr(PlayerGameStats, TARGET_COLUMNS[target])
        actual: dict[tuple[Any, Any], float] = {}
        for game_id, team_id, total in (
            await session.execute(
                select(GameTeam.game_id, GameTeam.team_id, func.sum(column))
                .join(PlayerGameStats, PlayerGameStats.game_team_id == GameTeam.id)
                .where(GameTeam.game_id.in_(game_ids))
                .group_by(GameTeam.game_id, GameTeam.team_id)
            )
        ).all():
            actual[(game_id, team_id)] = target_value(target, int(total or 0))

        projected: dict[str, dict[tuple[Any, Any], float]] = {"median": {}, "expected": {}}
        for (game_id, player_id, item), (team_id, median, expected, _c) in latest.items():
            if item is not target:
                continue
            key = (game_id, team_id)
            mean = float(expected) if expected is not None else median
            weight = plays.get((game_id, player_id), 1.0)
            projected["median"][key] = projected["median"].get(key, 0.0) + median
            projected["expected"][key] = projected["expected"].get(key, 0.0) + weight * mean

        summary = summarise_team_totals(projected, actual)
        if summary:
            report[str(target)] = summary
    return report


async def availability_report(
    session: AsyncSession, *, season_code: str | None = None, active_only: bool = False
) -> dict[str, Any]:
    """How the published P(plays) did: calibration, and players used per team.

    Non-appearances are the whole point here, so there is no ``include_absent``
    switch. Excluding them would leave only the players who played, and every
    probability would look too low.
    """
    from euroleague_ai.ml.availability import classification_metrics, reliability_table
    from euroleague_ai.models.features import PlayerFeatureSnapshot

    stmt = (
        select(
            Prediction.predicted_value,
            PredictionOutcome.actual_value,
            Prediction.game_id,
            PlayerFeatureSnapshot.team_id,
        )
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(
            PlayerFeatureSnapshot,
            PlayerFeatureSnapshot.id == Prediction.player_feature_snapshot_id,
        )
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.target == PredictionTarget.PLAYS)
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    stmt = _only_active(stmt, active_only=active_only)

    rows = (await session.execute(stmt)).all()
    if not rows:
        return {"scored": 0}

    probabilities = [float(row[0]) for row in rows]
    labels = [float(row[1]) for row in rows]
    groups = [(row[2], row[3]) for row in rows]
    metrics = classification_metrics(probabilities, labels, groups).as_dict()
    return {
        "scored": len(rows),
        **{key: value for key, value in metrics.items() if key != "rows"},
        "reliability": reliability_table(probabilities, labels),
    }


async def interval_coverage(
    session: AsyncSession,
    *,
    target: PredictionTarget,
    season_code: str | None = None,
    include_absent: bool = False,
    active_only: bool = False,
) -> dict[str, Any]:
    """Did the published 80% interval contain 80% of outcomes?

    Computed in Python rather than SQL: the quantiles live in JSONB and the
    arithmetic is trivial, so clarity wins over a clever query.
    """
    stmt = (
        select(Prediction.quantiles, PredictionOutcome.actual_value)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.target == target, Prediction.quantiles.isnot(None))
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    if not include_absent:
        stmt = stmt.where(PredictionOutcome.did_not_appear.is_(False))
    stmt = _only_active(stmt, active_only=active_only)

    # JSON null is not SQL NULL: a model without a distribution stores the former.
    rows = [row for row in (await session.execute(stmt)).all() if row[0]]
    if not rows:
        return {"rows": 0}

    inside = 0
    below = 0
    for quantiles, actual in rows:
        low = float(quantiles.get("0.1", 0.0))
        high = float(quantiles.get("0.9", 0.0))
        if actual < low:
            below += 1
        elif actual <= high:
            inside += 1

    total = len(rows)
    return {
        "rows": total,
        "covered_80": round(inside / total, 4),
        "below_interval": round(below / total, 4),
        "above_interval": round((total - inside - below) / total, 4),
    }


async def threshold_calibration(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    include_absent: bool = False,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    """Compare stated probabilities against realised frequencies.

    The cleanest test in the whole system, because it involves no model: of
    everything we called 30%, roughly 30% should have happened. A band whose
    realised rate sits far from its stated rate is a probability we should not
    be publishing.
    """
    stmt = (
        select(
            Prediction.target,
            Prediction.threshold_probabilities,
            PredictionOutcome.actual_value,
        )
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.threshold_probabilities.isnot(None))
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    if not include_absent:
        stmt = stmt.where(PredictionOutcome.did_not_appear.is_(False))
    stmt = _only_active(stmt, active_only=active_only)

    return _bands(
        (float(stated), float(actual) >= float(threshold))
        for _target, probabilities, actual in (await session.execute(stmt)).all()
        for threshold, stated in (probabilities or {}).items()
    )


async def threshold_calibration_by_target(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    include_absent: bool = False,
    active_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """The same bands, one table per target: a pooled band can hide which stat is off."""
    stmt = (
        select(
            Prediction.target,
            Prediction.threshold_probabilities,
            PredictionOutcome.actual_value,
        )
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(Game, Game.id == Prediction.game_id)
        .where(Prediction.threshold_probabilities.isnot(None))
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    if not include_absent:
        stmt = stmt.where(PredictionOutcome.did_not_appear.is_(False))
    stmt = _only_active(stmt, active_only=active_only)

    pairs: dict[str, list[tuple[float, bool]]] = {}
    for target, probabilities, actual in (await session.execute(stmt)).all():
        for threshold, stated in (probabilities or {}).items():
            pairs.setdefault(str(target), []).append(
                (float(stated), float(actual) >= float(threshold))
            )
    return {target: _bands(items) for target, items in sorted(pairs.items())}


async def overall_threshold_calibration(
    session: AsyncSession, *, season_code: str | None = None, active_only: bool = False
) -> list[dict[str, Any]]:
    """The same test for the published *overall* chances: P(plays) x P(X+ | plays).

    Every prediction counts here, including players who never appeared: a
    player who sat did not reach 15, and the overall chance said how likely
    that was. Each stat prediction is paired with the P(plays) row made for the
    same player, game and cutoff (the newest, if a retrain left two).
    """
    plays = aliased(Prediction)
    stmt = (
        select(
            Prediction.id,
            Prediction.threshold_probabilities,
            PredictionOutcome.actual_value,
            plays.predicted_value,
        )
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(Game, Game.id == Prediction.game_id)
        .join(
            plays,
            and_(
                plays.player_id == Prediction.player_id,
                plays.game_id == Prediction.game_id,
                plays.prediction_timestamp == Prediction.prediction_timestamp,
                plays.target == PredictionTarget.PLAYS,
            ),
        )
        .where(Prediction.threshold_probabilities.isnot(None))
        .distinct(Prediction.id)
        .order_by(Prediction.id, plays.created_at.desc())
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    stmt = _only_active(stmt, active_only=active_only)
    return _bands(
        (float(stated) * float(p_plays), float(actual) >= float(threshold))
        for _id, probabilities, actual, p_plays in (await session.execute(stmt)).all()
        for threshold, stated in (probabilities or {}).items()
    )


def _bands(pairs: Any) -> list[dict[str, Any]]:
    """``(stated probability, happened)`` pairs into ten-point bands.

    ``expected`` is the mean stated probability inside the band, not its
    midpoint. Probabilities are not spread evenly within a band (most of the
    0-10% band sits near 1-2%, most of 90-100% near 97%), so comparing with the
    midpoint shows drift that is not there, or hides drift that is.
    """
    buckets: dict[tuple[float, float], list[float]] = {
        band: [0.0, 0.0, 0.0] for band in PROBABILITY_BANDS
    }
    for probability, happened in pairs:
        for band in PROBABILITY_BANDS:
            low, high = band
            # Upper-inclusive on the last band only, so 1.0 has a home.
            if low <= probability < high or (high == 1.0 and probability == 1.0):
                buckets[band][0] += 1
                buckets[band][2] += probability
                if happened:
                    buckets[band][1] += 1
                break

    return [
        {
            "band": f"{int(low * 100)}-{int(high * 100)}%",
            "predictions": int(count),
            "realised": round(hits / count, 4) if count else None,
            "expected": round(stated / count, 4),
            "midpoint": round((low + high) / 2, 4),
        }
        for (low, high), (count, hits, stated) in buckets.items()
        if count
    ]


async def unscored_games(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    clock: Clock | None = None,
) -> int:
    """Finished games that still hold predictions without outcomes.

    A standing number that should be zero. Anything else means the scoring job
    has not run, or has been failing quietly.
    """
    clock = clock or SystemClock()
    stmt = (
        select(func.count(func.distinct(Prediction.game_id)))
        .join(Game, Game.id == Prediction.game_id)
        .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(
            Game.status == GameStatus.FINISHED,
            Game.tip_off_utc < clock.now(),
            PredictionOutcome.id.is_(None),
        )
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    return int((await session.execute(stmt)).scalar_one() or 0)


async def model_versions_in_play(session: AsyncSession) -> list[ModelVersion]:
    """Model versions that have scored outcomes, newest first.

    Accuracy is only comparable within a model version, so anything reporting
    across versions has to say which ones it mixed.
    """
    stmt = (
        select(ModelVersion)
        .join(Prediction, Prediction.model_version_id == ModelVersion.id)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .group_by(ModelVersion.id)
        .order_by(ModelVersion.trained_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def accuracy_by_version(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    include_absent: bool = False,
) -> list[dict[str, Any]]:
    """Live accuracy per model version, oldest first within each target.

    The retrain rhythm (roadmap p2-03) needs one question answered before a new
    version is trusted: is its live error no worse than the last one's? Pooled
    accuracy cannot say, because it mixes versions. For P(plays) the error is
    the Brier score, and absences are always included (they are what it
    predicts).
    """
    stmt = (
        select(
            ModelVersion.version,
            ModelVersion.target,
            ModelVersion.feature_set_version,
            ModelVersion.trained_at,
            ModelVersion.is_active,
            func.count(PredictionOutcome.id),
            func.avg(PredictionOutcome.absolute_error),
            func.avg(PredictionOutcome.squared_error),
            func.avg(Prediction.predicted_value - PredictionOutcome.actual_value),
            func.count(func.distinct(Game.id)),
            func.min(Game.round_number),
            func.max(Game.round_number),
        )
        .join(Prediction, Prediction.model_version_id == ModelVersion.id)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .join(Game, Game.id == Prediction.game_id)
        .group_by(ModelVersion.id)
        .order_by(
            ModelVersion.target, ModelVersion.trained_at.asc().nulls_first(), ModelVersion.version
        )
    )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)
    if not include_absent:
        stmt = stmt.where(
            (PredictionOutcome.did_not_appear.is_(False))
            | (Prediction.target == PredictionTarget.PLAYS)
        )

    out: list[dict[str, Any]] = []
    for row in (await session.execute(stmt)).all():
        (
            version,
            target,
            feature_set,
            trained_at,
            active,
            scored,
            mae,
            mse,
            bias,
            games,
            first_round,
            last_round,
        ) = row
        is_plays = target is PredictionTarget.PLAYS
        out.append(
            {
                "target": str(target),
                "version": version,
                "feature_set": feature_set,
                "trained_at": trained_at.isoformat() if trained_at else None,
                "active": bool(active),
                "scored": int(scored),
                "games": int(games),
                "rounds": [first_round, last_round],
                "metric": "brier" if is_plays else "mae",
                "error": round(float(mse if is_plays else mae), 4),
                "bias": round(float(bias), 4),
            }
        )
    return out


async def head_to_head(
    session: AsyncSession,
    *,
    target: PredictionTarget,
    newer: str,
    older: str,
    include_absent: bool = False,
) -> dict[str, Any]:
    """Two versions' live error on exactly the same player-games.

    Separate rounds are different difficulties: a version that only saw a
    round of blowouts looks worse than it is. Where both versions predicted
    the same player in the same game, the comparison is fair.
    """
    new_pred = aliased(Prediction)
    old_pred = aliased(Prediction)
    new_version = aliased(ModelVersion)
    old_version = aliased(ModelVersion)
    new_outcome = aliased(PredictionOutcome)
    old_outcome = aliased(PredictionOutcome)
    is_plays = target is PredictionTarget.PLAYS
    error_new = new_outcome.squared_error if is_plays else new_outcome.absolute_error
    error_old = old_outcome.squared_error if is_plays else old_outcome.absolute_error

    stmt = (
        select(func.count(), func.avg(error_new), func.avg(error_old))
        .select_from(new_pred)
        .join(new_version, new_version.id == new_pred.model_version_id)
        .join(new_outcome, new_outcome.prediction_id == new_pred.id)
        .join(
            old_pred,
            and_(
                old_pred.player_id == new_pred.player_id,
                old_pred.game_id == new_pred.game_id,
                old_pred.target == new_pred.target,
            ),
        )
        .join(old_version, old_version.id == old_pred.model_version_id)
        .join(old_outcome, old_outcome.prediction_id == old_pred.id)
        .where(
            new_pred.target == target,
            new_version.version == newer,
            old_version.version == older,
        )
    )
    if not include_absent and not is_plays:
        stmt = stmt.where(new_outcome.did_not_appear.is_(False))
    shared, new_error, old_error = (await session.execute(stmt)).one()
    return {
        "target": str(target),
        "newer": newer,
        "older": older,
        "shared": int(shared or 0),
        "metric": "brier" if is_plays else "mae",
        "newer_error": round(float(new_error), 4) if new_error is not None else None,
        "older_error": round(float(old_error), 4) if old_error is not None else None,
    }

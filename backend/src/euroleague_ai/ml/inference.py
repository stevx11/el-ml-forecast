"""Predicting upcoming fixtures.

Training rows are selected from box scores -- a row exists because a player
appeared. An upcoming game has no box score, so inference needs a different way
to decide *who to predict for*.

**Roster resolution must be transfer-aware.** Candidates come from the stint
intervals, resolved by :mod:`euroleague_ai.repositories.rosters` -- the same code
the snapshot backfill uses, so the play-probability model is trained on exactly
the population it is served on. See that module for the transfer-window rules.

**The cutoff is the same one training used.** ``tip_off - PREDICTION_LEAD_MINUTES``.
That is not a detail -- if inference computed features closer to tip-off than
training did, the model would be served an information set richer than the one it
learned on, and would quietly underperform its offline metrics.

Everything written here is reproducible: each :class:`Prediction` references the
model version, the cutoff, and the exact feature snapshot it consumed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import NotFoundError
from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.injury_status import ResolvedStatus, resolve_status
from euroleague_ai.features.registry import FEATURE_SET_VERSION, build_player_features, can_serve
from euroleague_ai.features.snapshot import default_prediction_timestamp
from euroleague_ai.ml.distribution import ResidualDistribution, threshold_cutoff, thresholds_for
from euroleague_ai.ml.expectation import (
    SOURCE_BOOSTER,
    SOURCE_NONE,
    SOURCE_SHIFTED_MEDIAN,
    shifted_median,
)
from euroleague_ai.ml.explain import PLAYS_MIN_IMPACT, PLAYS_PHRASES, explain_prediction
from euroleague_ai.ml.matrix import features_to_matrix
from euroleague_ai.ml.reconcile import MinutesReconciliation, load_reconciliation
from euroleague_ai.ml.registry import active_model, distribution_path, expectation_paths
from euroleague_ai.ml.squad_stats import (
    STAT_TARGETS,
    StatsReconciliation,
    load_stats_reconciliation,
)
from euroleague_ai.models.enums import GameStatus, PredictionTarget, StatusSourceType
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game
from euroleague_ai.models.prediction import ModelVersion, Prediction
from euroleague_ai.repositories.availability import last_appearances, status_reports
from euroleague_ai.repositories.rosters import candidate_players, current_club

logger = get_logger(__name__)


@dataclass(slots=True)
class LoadedModel:
    """A booster plus everything needed to use it correctly."""

    model_version: ModelVersion
    booster: Any
    feature_names: list[str]
    calibration_offset: float
    target: PredictionTarget
    #: Held-out error distribution. ``None`` for models trained before item 06,
    #: in which case predictions carry a point estimate only rather than a
    #: fabricated interval.
    distribution: ResidualDistribution | None = None
    #: Where the expected value comes from: ``booster``, ``shifted_median`` or
    #: ``none``. See :mod:`euroleague_ai.ml.expectation`.
    expected_source: str = SOURCE_NONE
    #: The mean-targeting booster, when ``expected_source == "booster"``.
    expected_booster: Any = None

    def predict_batch(self, feature_dicts: list[dict[str, Any]]) -> list[float]:
        """Predict a whole squad at once, calibrated and clipped.

        Batched because a game is ~24 players x 5 targets; calling the booster
        once per row would be 120 round trips through LightGBM for no reason.

        Counting stats cannot be negative, so the calibrated value is clipped at
        zero -- the same treatment applied during training, so offline metrics
        and served predictions agree.
        """
        if not feature_dicts:
            return []
        x = features_to_matrix(feature_dicts, self.feature_names)
        raw = self.booster.predict(x)
        if self.is_probability:
            # A probability is never shifted, only kept inside [0, 1].
            return [min(1.0, max(0.0, float(value))) for value in raw]
        return [max(0.0, float(value) + self.calibration_offset) for value in raw]

    def expected_batch(
        self, feature_dicts: list[dict[str, Any]], medians: list[float]
    ) -> list[float | None]:
        """Expected value given he plays, per row, or ``None`` where there is none.

        ``medians`` are this model's own served predictions for the same rows,
        needed by the shifted-median source.
        """
        missing: list[float | None] = [None] * len(feature_dicts)
        if not feature_dicts or self.is_probability:
            return missing
        if self.expected_source == SOURCE_BOOSTER and self.expected_booster is not None:
            x = features_to_matrix(feature_dicts, self.feature_names)
            return [max(0.0, float(v)) for v in self.expected_booster.predict(x)]
        if self.expected_source == SOURCE_SHIFTED_MEDIAN and self.distribution is not None:
            return list(
                shifted_median(
                    medians,
                    None,
                    self.distribution,
                    histories=[f.get("games_played_career_window") for f in feature_dicts],
                )
            )
        return missing

    @property
    def is_probability(self) -> bool:
        """``True`` for the play-probability classifier, whose output is P(plays)."""
        return self.target is PredictionTarget.PLAYS

    def explain_batch(self, feature_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Exact TreeSHAP contributions per row, grouped into readable themes.

        Uses LightGBM's own ``pred_contrib``, which is exact for trees and needs
        no extra dependency. The returned matrix has one extra trailing column
        holding the base value, so it is ``len(feature_names) + 1`` wide -- that
        trailing term is the model's expected output, not a feature.
        """
        if not feature_dicts:
            return []

        x = features_to_matrix(feature_dicts, self.feature_names)
        contributions = self.booster.predict(x, pred_contrib=True)

        width = len(self.feature_names)
        explanations: list[dict[str, Any]] = []
        for row in contributions:
            values = [float(v) for v in row]
            if len(values) != width + 1:
                # Refuse rather than mis-attribute. A silent off-by-one here
                # would shift every contribution onto the wrong feature name and
                # produce explanations that read perfectly and are entirely wrong.
                raise ValueError(
                    f"pred_contrib returned {len(values)} columns for "
                    f"{width} features; expected {width + 1}"
                )
            if self.is_probability:
                # Binary boosters attribute in log-odds, not in probability, so
                # the totals are recorded with their scale and phrased for the
                # question this model answers.
                explanations.append(
                    explain_prediction(
                        self.feature_names,
                        values[:width],
                        base_value=values[width],
                        phrases=PLAYS_PHRASES,
                        min_impact=PLAYS_MIN_IMPACT,
                        scale="log_odds",
                    )
                )
                continue
            explanations.append(
                explain_prediction(
                    self.feature_names,
                    values[:width],
                    base_value=values[width],
                    calibration_offset=self.calibration_offset,
                )
            )
        return explanations

    def distribution_for(self, predicted: float, features: dict[str, Any]) -> dict[str, Any]:
        """Quantiles, threshold probabilities, spread and confidence.

        Empty when the model predates the distribution work. Publishing an
        interval we cannot justify would be worse than publishing none.
        """
        if self.distribution is None:
            return {}

        history = features.get("games_played_career_window")
        return {
            "quantiles": self.distribution.quantiles(predicted, history),
            "threshold_probabilities": {
                # Keys are strings because this lands in JSONB, where numeric
                # keys would come back as strings anyway.
                str(int(threshold)): self.distribution.probability_at_least(
                    predicted, history, threshold_cutoff(str(self.target), threshold)
                )
                for threshold in thresholds_for(str(self.target))
            },
            "predicted_std": self.distribution.spread(predicted, history),
            "confidence": self.distribution.confidence(predicted, history),
        }


def load_model(model_version: ModelVersion) -> LoadedModel:
    """Load a booster and its pinned column order from disk.

    The column order comes from the artifact, never from the current feature
    registry. A model trained before a new feature family was added must keep
    seeing its original columns in its original positions -- deriving them fresh
    would silently shift every value one slot and produce plausible nonsense.
    """
    import lightgbm as lgb

    if not model_version.artifact_path:
        raise NotFoundError("ModelVersion.artifact_path", model_version.id)

    model_path = Path(model_version.artifact_path)
    columns_path = model_path.parent / "columns.json"
    if not model_path.is_file() or not columns_path.is_file():
        raise NotFoundError("model artifact", str(model_path))

    metadata = json.loads(columns_path.read_text(encoding="utf-8"))
    metrics = model_version.metrics or {}

    # Optional on purpose: a model registered before item 06 has no
    # distribution.json, and must keep loading and serving point estimates.
    residuals: ResidualDistribution | None = None
    dist_path = distribution_path(model_path)
    if dist_path.is_file():
        residuals = ResidualDistribution.from_dict(
            json.loads(dist_path.read_text(encoding="utf-8"))
        )

    # Expected values. A model trained with the expectation head records its
    # gated choice; one trained before it has no record, and gets the shifted
    # median, which needs nothing but the distribution it already carries.
    expected_source = SOURCE_NONE
    expected_booster = None
    booster_path, report_path = expectation_paths(model_path)
    if report_path.is_file():
        expected_source = str(
            json.loads(report_path.read_text(encoding="utf-8")).get("source", SOURCE_NONE)
        )
        if expected_source == SOURCE_BOOSTER:
            if booster_path.is_file():
                expected_booster = lgb.Booster(model_file=str(booster_path))
            else:
                logger.warning(
                    "expectation booster missing; no expected values",
                    extra={"artifact": str(booster_path)},
                )
                expected_source = SOURCE_NONE
    elif residuals is not None and model_version.target is not PredictionTarget.PLAYS:
        expected_source = SOURCE_SHIFTED_MEDIAN

    return LoadedModel(
        model_version=model_version,
        booster=lgb.Booster(model_file=str(model_path)),
        feature_names=list(metadata["feature_names"]),
        calibration_offset=float(metrics.get("calibration_offset", 0.0)),
        target=model_version.target,
        distribution=residuals,
        expected_source=expected_source,
        expected_booster=expected_booster,
    )


async def load_active_models(
    session: AsyncSession, targets: tuple[PredictionTarget, ...]
) -> dict[PredictionTarget, LoadedModel]:
    """Load every active model for the requested targets, skipping missing ones."""
    loaded: dict[PredictionTarget, LoadedModel] = {}
    stale: list[str] = []
    for target in targets:
        model_version = await active_model(session, target)
        if model_version is None:
            logger.warning("no active model", extra={"target": str(target)})
            continue

        # A model may only be served the feature set it was trained on. Feature
        # names are stable across versions while their *meaning* is not -- in v3
        # ``last5_points_mean`` crosses the summer break and in v2 it did not --
        # so a mismatch produces confident, silently wrong predictions rather
        # than an error. Skip loudly instead.
        if not can_serve(model_version.feature_set_version):
            stale.append(
                f"{target}: {model_version.name} {model_version.version} "
                f"trained on {model_version.feature_set_version}"
            )
            logger.warning(
                "active model trained on a different feature set; skipping",
                extra={
                    "target": str(target),
                    "model_feature_set": model_version.feature_set_version,
                    "current_feature_set": FEATURE_SET_VERSION,
                },
            )
            continue

        try:
            loaded[target] = load_model(model_version)
        except NotFoundError as exc:
            logger.warning(
                "model artifact missing", extra={"target": str(target), "error": str(exc)}
            )

    if stale:
        logger.warning(
            "stale models skipped; retrain them",
            extra={"stale": "; ".join(stale), "current_feature_set": FEATURE_SET_VERSION},
        )
    return loaded


async def predict_game(
    session: AsyncSession,
    game: Game,
    models: dict[PredictionTarget, LoadedModel],
    *,
    clock: Clock | None = None,
    settings: Settings | None = None,
    reconciliation: MinutesReconciliation | None = None,
    stats_reconciliation: StatsReconciliation | None = None,
) -> int:
    """Build snapshots and predictions for one upcoming game.

    Args:
        reconciliation: A gated minutes fit from ``ml reconcile``. When it was
            fitted for exactly these MINUTES and PLAYS model versions, each
            side's expected minutes are scaled so the squad adds up. See
            :mod:`euroleague_ai.ml.reconcile`.
        stats_reconciliation: The gated squad fit for points, assists, rebounds
            and threes, from the same command. Pulls each side's expected stat
            totals toward a team-level anchor. See
            :mod:`euroleague_ai.ml.squad_stats`.

    Returns the number of prediction rows written.
    """
    clock = clock or SystemClock()
    settings = settings or get_settings()
    cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
    now = clock.now()

    # Resolved once and shared: both sides need the same transfer-aware view,
    # and a player must not be able to appear for both teams in one fixture.
    club_by_player = await current_club(session, as_of=cutoff)

    # Build every feature vector first, then predict per target in one batch.
    rows: list[tuple[uuid.UUID, uuid.UUID, dict[str, Any]]] = []

    # One cache for the whole fixture. Team ratings and squad data are the same
    # for every player on a side; without this they are re-queried per player.
    feature_cache: dict[Any, Any] = {}

    for team_id, opponent_id, is_home in (
        (game.home_team_id, game.away_team_id, True),
        (game.away_team_id, game.home_team_id, False),
    ):
        for player_id in await candidate_players(
            session,
            team_id=team_id,
            season_id=game.season_id,
            as_of=cutoff,
            club_by_player=club_by_player,
        ):
            features = await build_player_features(
                session,
                player_id=player_id,
                team_id=team_id,
                opponent_team_id=opponent_id,
                as_of=cutoff,
                season_id=game.season_id,
                is_home=is_home,
                cache=feature_cache,
                game=game,
            )
            snapshot_id = await _upsert_snapshot(
                session,
                player_id=player_id,
                game=game,
                team_id=team_id,
                opponent_team_id=opponent_id,
                cutoff=cutoff,
                features=features,
                computed_at=now,
            )
            rows.append((player_id, snapshot_id, features))

    if not rows:
        logger.warning("no candidate players", extra={"game_code": game.game_code})
        return 0

    # Cheap invariant, loud failure. If club resolution ever regresses, a player
    # appearing on both sides is the symptom, and silently projecting them twice
    # is far worse than refusing.
    player_ids = [player_id for player_id, _, _ in rows]
    if len(player_ids) != len(set(player_ids)):
        duplicates = {p for p in player_ids if player_ids.count(p) > 1}
        raise ValueError(
            f"game {game.game_code}: {len(duplicates)} player(s) resolved to both "
            "teams. Club resolution is broken -- refusing to write predictions."
        )

    feature_dicts = [features for _, _, features in rows]
    written = 0

    # Every target first, then the cross-target adjustment, then the writes.
    # Reconciling minutes needs P(plays) for the same players, so nothing can be
    # written until both are known.
    computed: dict[PredictionTarget, tuple[list[float], list[float | None], list[Any]]] = {}
    for target, model in models.items():
        values = model.predict_batch(feature_dicts)
        computed[target] = (
            values,
            model.expected_batch(feature_dicts, values),
            model.explain_batch(feature_dicts),
        )

    # Injury reports known at the cutoff. Applied before minutes reconciliation,
    # so a player ruled out has P(plays) = 0 and his minutes flow to the
    # teammates who will actually be on the floor.
    statuses = await _resolve_statuses(session, game, rows, cutoff=cutoff)
    if statuses:
        _apply_statuses(computed, statuses)

    if (
        reconciliation is not None
        and PredictionTarget.MINUTES in computed
        and PredictionTarget.PLAYS in computed
    ):
        if reconciliation.applies_to(models):
            medians, expected_minutes, minute_explanations = computed[PredictionTarget.MINUTES]
            base = [
                e if e is not None else m for e, m in zip(expected_minutes, medians, strict=True)
            ]
            reconciled = reconciliation.apply(
                base,
                computed[PredictionTarget.PLAYS][0],
                # One factor per side of the fixture.
                [features.get("is_home") for features in feature_dicts],
            )
            computed[PredictionTarget.MINUTES] = (
                medians,
                [round(v, 4) for v in reconciled],
                minute_explanations,
            )
        else:
            logger.warning(
                "minutes reconciliation was fitted for other model versions; "
                "skipping it. Run `euroleague-ai ml reconcile` after training.",
                extra={"game_code": game.game_code},
            )

    if stats_reconciliation is not None and PredictionTarget.PLAYS in computed:
        if stats_reconciliation.applies_to(models):
            _reconcile_stats(computed, stats_reconciliation, feature_dicts)
        else:
            logger.warning(
                "squad stat reconciliation was fitted for other model versions; "
                "skipping it. Run `euroleague-ai ml reconcile` after training.",
                extra={"game_code": game.game_code},
            )

    for target, model in models.items():
        values, expected_values, explanations = computed[target]
        for (player_id, snapshot_id, features), value, expected, explanation in zip(
            rows, values, expected_values, explanations, strict=True
        ):
            spread = model.distribution_for(value, features)
            await session.execute(
                pg_insert(Prediction)
                .values(
                    id=uuid.uuid4(),
                    model_version_id=model.model_version.id,
                    player_feature_snapshot_id=snapshot_id,
                    player_id=player_id,
                    game_id=game.id,
                    target=target,
                    predicted_value=value,
                    expected_value=expected,
                    predicted_std=spread.get("predicted_std"),
                    quantiles=spread.get("quantiles"),
                    threshold_probabilities=spread.get("threshold_probabilities"),
                    confidence=spread.get("confidence"),
                    explanation=explanation,
                    prediction_timestamp=cutoff,
                )
                .on_conflict_do_update(
                    index_elements=[
                        "model_version_id",
                        "player_id",
                        "game_id",
                        "target",
                        "prediction_timestamp",
                    ],
                    set_={
                        "predicted_value": value,
                        "expected_value": expected,
                        "predicted_std": spread.get("predicted_std"),
                        "quantiles": spread.get("quantiles"),
                        "threshold_probabilities": spread.get("threshold_probabilities"),
                        "confidence": spread.get("confidence"),
                        "explanation": explanation,
                    },
                )
            )
            written += 1

    return written


def _reconcile_stats(
    computed: dict[PredictionTarget, tuple[list[float], list[float | None], list[Any]]],
    fit: StatsReconciliation,
    feature_dicts: list[dict[str, Any]],
) -> None:
    """Scale each side's expected stat lines toward its squad anchor.

    Every factor is computed from the unreconciled values first, as in the
    fit, so the order the stats are adjusted in does not matter. The factor is
    kept in the explanation, so a line can be traced back to its model value.
    """
    base: dict[PredictionTarget, list[float]] = {}
    for target in STAT_TARGETS:
        if target not in computed:
            continue
        medians, expected, _ = computed[target]
        if any(e is None for e in expected):
            continue  # nothing published to adjust
        base[target] = [float(e) for e in expected if e is not None]
    factors = fit.factors(
        base,
        computed[PredictionTarget.PLAYS][0],
        [features.get("is_home") for features in feature_dicts],
        feature_dicts,
    )
    for target, per_row in factors.items():
        medians, _, explanations = computed[target]
        computed[target] = (
            medians,
            [round(v * f, 4) for v, f in zip(base[target], per_row, strict=True)],
            [
                {**(explanation or {}), "squad_factor": round(f, 4)}
                for explanation, f in zip(explanations, per_row, strict=True)
            ],
        )


async def _resolve_statuses(
    session: AsyncSession,
    game: Game,
    rows: list[tuple[uuid.UUID, uuid.UUID, dict[str, Any]]],
    *,
    cutoff: datetime,
) -> dict[int, ResolvedStatus]:
    """Row index to the injury report that applies to it in this game."""
    player_ids = [player_id for player_id, _, _ in rows]
    reports = await status_reports(session, player_ids, as_of=cutoff)
    if not reports:
        return {}
    played = await last_appearances(session, list(reports), as_of=cutoff)
    out: dict[int, ResolvedStatus] = {}
    for index, player_id in enumerate(player_ids):
        resolved = resolve_status(
            reports.get(player_id, []),
            game_id=game.id,
            tip_off=game.tip_off_utc,
            last_played=played.get(player_id),
        )
        if resolved is not None:
            out[index] = resolved
    return out


def _apply_statuses(
    computed: dict[PredictionTarget, tuple[list[float], list[float | None], list[Any]]],
    statuses: dict[int, ResolvedStatus],
) -> None:
    """Zero P(plays) for players ruled out; attach the report to every line.

    Stat lines stay "if he plays", like every other projection, so a player
    reinstated late still has a sensible line. The squad totals and minutes
    reconciliation discount him through P(plays) = 0.
    """
    for target, (values, _expected, explanations) in computed.items():
        for index, resolved in statuses.items():
            explanation = dict(explanations[index] or {})
            explanation["availability"] = resolved.as_dict()
            if target is PredictionTarget.PLAYS and resolved.rules_out:
                values[index] = 0.0
                reason = resolved.report.reason
                if resolved.report.source is StatusSourceType.NEWS:
                    label = "Reported out in the news"
                else:
                    label = "Ruled out" + (f": {reason}" if reason else "")
                explanation["drivers"] = [
                    {"theme": "availability", "direction": "-", "label": label, "impact": 0.0}
                ]
            explanations[index] = explanation


async def _upsert_snapshot(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    game: Game,
    team_id: uuid.UUID,
    opponent_team_id: uuid.UUID,
    cutoff: datetime,
    features: dict[str, Any],
    computed_at: datetime,
) -> uuid.UUID:
    """Return the snapshot id for this player-game, creating it if absent.

    Snapshots stay immutable: an existing one is reused rather than rewritten, so
    a prediction made yesterday still points at the inputs it actually used.
    """
    existing = (
        await session.execute(
            select(PlayerFeatureSnapshot.id).where(
                PlayerFeatureSnapshot.player_id == player_id,
                PlayerFeatureSnapshot.game_id == game.id,
                PlayerFeatureSnapshot.feature_set_version == FEATURE_SET_VERSION,
                PlayerFeatureSnapshot.prediction_timestamp == cutoff,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    snapshot_id = uuid.uuid4()
    session.add(
        PlayerFeatureSnapshot(
            id=snapshot_id,
            player_id=player_id,
            game_id=game.id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            prediction_timestamp=cutoff,
            feature_set_version=FEATURE_SET_VERSION,
            features=features,
            computed_at=computed_at,
        )
    )
    await session.flush()
    return snapshot_id


async def upcoming_games(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    within_days: int = 7,
    game_codes: list[int] | None = None,
    clock: Clock | None = None,
) -> list[Game]:
    """Scheduled games to predict, soonest first."""
    from euroleague_ai.models.competition import Season

    clock = clock or SystemClock()
    now = clock.now()

    stmt = select(Game).where(Game.status == GameStatus.SCHEDULED).order_by(Game.tip_off_utc)
    if game_codes:
        stmt = stmt.where(Game.game_code.in_(game_codes))
    else:
        stmt = stmt.where(
            Game.tip_off_utc >= now,
            Game.tip_off_utc <= now + timedelta(days=within_days),
        )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(Season.code == season_code)

    return list((await session.execute(stmt)).scalars().all())


async def predict_upcoming(
    session: AsyncSession,
    *,
    season_code: str | None = None,
    within_days: int = 7,
    game_codes: list[int] | None = None,
    targets: tuple[PredictionTarget, ...] | None = None,
    clock: Clock | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Predict every upcoming fixture in the window."""
    from euroleague_ai.ml.report import SERVED_TARGETS

    clock = clock or SystemClock()
    resolved_targets = targets or SERVED_TARGETS

    models = await load_active_models(session, resolved_targets)
    if not models:
        return {
            "error": "no active models. Run `euroleague-ai ml train` first.",
            "games": 0,
            "predictions": 0,
        }

    games = await upcoming_games(
        session,
        season_code=season_code,
        within_days=within_days,
        game_codes=game_codes,
        clock=clock,
    )
    logger.info(
        "predicting upcoming games",
        extra={"games": len(games), "targets": [str(t) for t in models]},
    )

    reconciliation = load_reconciliation()
    if reconciliation is not None and not reconciliation.applies_to(models):
        logger.warning(
            "minutes reconciliation is stale for the active models; run "
            "`euroleague-ai ml reconcile`"
        )
        reconciliation = None
    stats_fit = load_stats_reconciliation()
    if stats_fit is not None and not stats_fit.applies_to(models):
        logger.warning(
            "squad stat reconciliation is stale for the active models; run "
            "`euroleague-ai ml reconcile`"
        )
        stats_fit = None

    total = 0
    for index, game in enumerate(games, start=1):
        total += await predict_game(
            session,
            game,
            models,
            clock=clock,
            settings=settings,
            reconciliation=reconciliation,
            stats_reconciliation=stats_fit,
        )
        await session.commit()
        logger.info(
            "predicted game",
            extra={"game_code": game.game_code, "progress": f"{index}/{len(games)}"},
        )

    return {
        "games": len(games),
        "predictions": total,
        "targets": [str(t) for t in models],
        "models": {
            str(t): f"{m.model_version.name} {m.model_version.version}" for t, m in models.items()
        },
        "minutes_reconciliation": (
            f"alpha {reconciliation.alpha}, squads to {reconciliation.target_minutes:.1f} min"
            if reconciliation is not None
            else "off"
        ),
        "stats_reconciliation": (
            ", ".join(f"{t.name.lower()} alpha {a}" for t, (a, _) in stats_fit.targets.items())
            if stats_fit is not None
            else "off"
        ),
    }

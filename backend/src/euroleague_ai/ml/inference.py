"""Predicting upcoming fixtures.

Training rows are selected from box scores -- a row exists because a player
appeared. An upcoming game has no box score, so inference needs a different way
to decide *who to predict for*.

**Roster resolution must be transfer-aware.** Candidates come from the stint
intervals, but "registered to this club in this season" is not sufficient: across
a transfer window a player can hold a registration for the season just finished
*and* one for the season starting, at different clubs. Filtering by season alone
returns both, and the player gets projected for a club he has left.

:func:`current_club` resolves it properly -- among registrations in effect at the
cutoff, the latest ``effective_from`` wins. The set is then narrowed to players
who have featured recently *in this season*, which drops the never-dressed tail
without discarding summer signings who have not debuted.

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

from euroleague_ai.core.clock import Clock, SystemClock, ensure_utc
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import NotFoundError
from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.registry import FEATURE_SET_VERSION, build_player_features
from euroleague_ai.ml.distribution import ResidualDistribution, thresholds_for
from euroleague_ai.ml.registry import distribution_path
from euroleague_ai.features.snapshot import default_prediction_timestamp
from euroleague_ai.models.enums import GameStatus, PredictionTarget
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.people import PlayerTeamStint
from euroleague_ai.models.prediction import ModelVersion, Prediction
from euroleague_ai.ml.matrix import features_to_matrix
from euroleague_ai.ml.registry import active_model

logger = get_logger(__name__)

#: A rostered player must have appeared in at least one of the team's last N
#: games to be predicted. Filters out the registered-but-never-dressed tail.
RECENT_APPEARANCE_GAMES = 5


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
        return [max(0.0, float(value) + self.calibration_offset) for value in raw]

    def distribution_for(
        self, predicted: float, features: dict[str, Any]
    ) -> dict[str, Any]:
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
                    predicted, history, threshold
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

    return LoadedModel(
        model_version=model_version,
        booster=lgb.Booster(model_file=str(model_path)),
        feature_names=list(metadata["feature_names"]),
        calibration_offset=float(metrics.get("calibration_offset", 0.0)),
        target=model_version.target,
        distribution=residuals,
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
        if model_version.feature_set_version != FEATURE_SET_VERSION:
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


async def current_club(
    session: AsyncSession, *, as_of: datetime
) -> dict[uuid.UUID, uuid.UUID]:
    """Map every player to the club they belonged to at ``as_of``.

    **This is the transfer-aware answer, and it has to be.** Scoping roster
    lookups to a single season is wrong across a transfer window: in pre-season
    a player can hold a registration for the season just finished *and* one for
    the season about to start, at different clubs. Filtering by season alone
    happily returns both, and a player ends up projected for a club he left.

    The rule is one line: among the registrations in effect at ``as_of``, the one
    with the latest ``effective_from`` wins. A July move to Real Madrid
    supersedes a registration at Partizan that nominally runs to 30 June, because
    it started later.

    Resolved in Python rather than SQL because the working set is a few thousand
    stints and the tie-breaking is far easier to read -- and to get right -- as
    explicit code.
    """
    as_of = ensure_utc(as_of, field="as_of")

    rows = (
        await session.execute(
            select(
                PlayerTeamStint.player_id,
                PlayerTeamStint.team_id,
                PlayerTeamStint.effective_from,
                PlayerTeamStint.observed_at,
            ).where(
                PlayerTeamStint.observed_at <= as_of,
                PlayerTeamStint.effective_from <= as_of,
                (PlayerTeamStint.effective_until.is_(None))
                | (PlayerTeamStint.effective_until > as_of),
            )
        )
    ).all()

    best: dict[uuid.UUID, tuple[datetime, datetime, uuid.UUID]] = {}
    for player_id, team_id, effective_from, observed_at in rows:
        key = (effective_from, observed_at, team_id)
        current = best.get(player_id)
        # Latest registration wins; observed_at breaks ties, so a correction
        # ingested later supersedes the row it corrects.
        if current is None or key[:2] > current[:2]:
            best[player_id] = key

    return {player_id: value[2] for player_id, value in best.items()}


async def candidate_players(
    session: AsyncSession,
    *,
    team_id: uuid.UUID,
    season_id: uuid.UUID,
    as_of: datetime,
    club_by_player: dict[uuid.UUID, uuid.UUID] | None = None,
) -> list[uuid.UUID]:
    """Players plausibly available for this team at ``as_of``.

    Two filters, in this order:

    1. **Club membership**, resolved by :func:`current_club`. A player whose
       latest registration is at another club is excluded outright, whatever the
       appearance history says.
    2. **Recent appearances in this season**, to drop the registered-but-never-
       dressed tail. Skipped before a season's first games, when there is nothing
       to filter on.

    The season scope on the appearance filter is load-bearing. Without it, "the
    team's last five games" reaches back into the previous season, and a squad
    that has since been rebuilt gets projected as though nobody moved.
    """
    as_of = ensure_utc(as_of, field="as_of")

    if club_by_player is None:
        club_by_player = await current_club(session, as_of=as_of)

    rostered = {
        player_id
        for player_id, club_id in club_by_player.items()
        if club_id == team_id
    }
    if not rostered:
        return []

    recent_games = (
        select(Game.id)
        .join(GameTeam, GameTeam.game_id == Game.id)
        .where(
            GameTeam.team_id == team_id,
            # Same season only. Cross-season appearance history describes a squad
            # that no longer exists.
            Game.season_id == season_id,
            Game.observed_at <= as_of,
            Game.tip_off_utc < as_of,
            Game.status == GameStatus.FINISHED,
        )
        .order_by(Game.tip_off_utc.desc())
        .limit(RECENT_APPEARANCE_GAMES)
        .subquery()
    )

    appeared = set(
        (
            await session.execute(
                select(PlayerGameStats.player_id)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                .where(
                    GameTeam.team_id == team_id,
                    GameTeam.game_id.in_(select(recent_games.c.id)),
                    PlayerGameStats.observed_at <= as_of,
                    PlayerGameStats.did_not_play.is_(False),
                )
                .distinct()
            )
        )
        .scalars()
        .all()
    )

    if not appeared:
        # No games played in this season yet. The roster is all we have, and it
        # is already club-correct.
        return sorted(rostered)

    # Everyone who has featured for this club at any point this season. Used to
    # tell "fringe player who keeps being left out" apart from "signing who has
    # not debuted yet" -- the appearance filter should drop the former, never the
    # latter.
    debuted = set(
        (
            await session.execute(
                select(PlayerGameStats.player_id)
                .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                .join(Game, Game.id == GameTeam.game_id)
                .where(
                    GameTeam.team_id == team_id,
                    Game.season_id == season_id,
                    Game.observed_at <= as_of,
                    Game.tip_off_utc < as_of,
                    PlayerGameStats.observed_at <= as_of,
                    PlayerGameStats.did_not_play.is_(False),
                )
                .distinct()
            )
        )
        .scalars()
        .all()
    )

    # Club-correct players who played recently, plus club-correct players who
    # have not yet appeared for this club this season (new signings, returning
    # injuries). The model projects low minutes for the uncertain ones, which is
    # the honest answer; silently omitting a summer signing is not.
    return sorted((rostered & appeared) | (rostered - debuted))


async def predict_game(
    session: AsyncSession,
    game: Game,
    models: dict[PredictionTarget, LoadedModel],
    *,
    clock: Clock | None = None,
    settings: Settings | None = None,
) -> int:
    """Build snapshots and predictions for one upcoming game.

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
        logger.warning(
            "no candidate players", extra={"game_code": game.game_code}
        )
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

    for target, model in models.items():
        values = model.predict_batch(feature_dicts)
        for (player_id, snapshot_id, features), value in zip(rows, values, strict=True):
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
                    predicted_std=spread.get("predicted_std"),
                    quantiles=spread.get("quantiles"),
                    threshold_probabilities=spread.get("threshold_probabilities"),
                    confidence=spread.get("confidence"),
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
                        "predicted_std": spread.get("predicted_std"),
                        "quantiles": spread.get("quantiles"),
                        "threshold_probabilities": spread.get(
                            "threshold_probabilities"
                        ),
                        "confidence": spread.get("confidence"),
                    },
                )
            )
            written += 1

    return written


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

    stmt = (
        select(Game)
        .where(Game.status == GameStatus.SCHEDULED)
        .order_by(Game.tip_off_utc)
    )
    if game_codes:
        stmt = stmt.where(Game.game_code.in_(game_codes))
    else:
        stmt = stmt.where(
            Game.tip_off_utc >= now,
            Game.tip_off_utc <= now + timedelta(days=within_days),
        )
    if season_code:
        stmt = stmt.join(Season, Season.id == Game.season_id).where(
            Season.code == season_code
        )

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
    from euroleague_ai.ml.report import DEFAULT_TARGETS

    clock = clock or SystemClock()
    resolved_targets = targets or DEFAULT_TARGETS

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

    total = 0
    for index, game in enumerate(games, start=1):
        total += await predict_game(
            session, game, models, clock=clock, settings=settings
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
            str(t): f"{m.model_version.name} {m.model_version.version}"
            for t, m in models.items()
        },
    }

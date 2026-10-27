"""Will he play? The play-probability model.

**The gap this closes.** Every stat model is trained on players who took the
floor (``exclude_dnp=True``) and predicts a line *given that he plays*.
Inference then projects every roster candidate, and that list always includes a
few players who will not appear: someone injured since the last game, a
twelfth man left out of the squad, a starter rested in a dead rubber. Without a
model of that, an injured starter keeps his 28 projected minutes, a team's
projected minutes add up to well over 200, and the scoring job quietly logs a
pile of ``did_not_appear`` outcomes that nothing learns from.

This module predicts ``P(plays)``: the probability that a candidate records any
playing time in the game.

**The population is the inference candidate list, exactly.** Rows are rebuilt
per historical game with
:func:`~euroleague_ai.repositories.rosters.candidate_players` at the same
cutoff inference uses. Training on "everyone in the box score" would be easier,
since those rows already exist, but it answers a different question. A box
score never contains the players who were absent from the squad, and those are
most of the zeros.

**Labels.** ``1`` if the player appears in this team's box score with playing
time, ``0`` otherwise. "Dressed but never used" and "not in the squad" are both
``0``: either way he did not play, and either way his projected line was
wrong.

**Features.** The existing feature snapshot, unchanged. The availability
family already carries the strongest signals (``player_played_last_game``, the
player's own ``days_rest``, ``games_last_14d``, his rank at his position), and
reusing the snapshot keeps every prediction reproducible from a stored vector,
with no new feature set version and no full re-backfill. The recent-appearance
counts are *not* model inputs. They come from a separate query and are used
only by the baselines, which keeps the comparison honest: if the model cannot
beat "he played four of the last five", that is a clear signal to promote those
counts into feature set v6.

**The gate is log loss, not accuracy.** With a base rate around 90%, "everyone
plays" scores 90% accuracy and is useless. Log loss and Brier score reward
probabilities that are right, which is what downstream code multiplies by.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.logging import get_logger
from euroleague_ai.features.registry import FEATURE_SET_VERSION
from euroleague_ai.features.snapshot import default_prediction_timestamp
from euroleague_ai.ml.matrix import EXCLUDED_KEYS, features_to_matrix, matrix_summary
from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT, next_version
from euroleague_ai.ml.train import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_PARAMS,
    TEST_FRACTION,
    VALIDATION_FRACTION,
)
from euroleague_ai.models.competition import Season
from euroleague_ai.models.enums import GameStatus, PredictionTarget
from euroleague_ai.models.features import PlayerFeatureSnapshot
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.prediction import ModelVersion
from euroleague_ai.repositories.rosters import (
    candidate_players,
    current_club,
    recent_appearances,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = get_logger(__name__)

PLAYS = PredictionTarget.PLAYS

#: Registry name. Matches ``model_name_for(PLAYS)`` so the naming stays uniform.
MODEL_NAME = "plays_lgbm"

#: Probabilities are clipped here before taking logs. LightGBM can emit
#: 0.99999 for a nailed-on starter, and one such row that does not play would
#: otherwise contribute a log loss of ~11 and swamp the average.
PROBABILITY_EPSILON = 1e-4

#: Reliability bins for the calibration table, ten points wide. Most mass sits
#: above 0.8, which is exactly why the upper bins have to be visible separately.
RELIABILITY_BINS = 10

#: Below this many rows the three-way chronological split is noise.
MIN_ROWS = 2_000

#: Shared with the stat models (seeds, regularisation, determinism), swapping only
#: the objective. Shallow, strongly regularised trees suit ~100k rows with a
#: 90/10 class balance, where a deep tree memorises individual absences.
DEFAULT_CLASSIFIER_PARAMS: dict[str, Any] = {
    **DEFAULT_PARAMS,
    "objective": "binary",
    "metric": "binary_logloss",
}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AvailabilityRow:
    """One roster candidate for one historical game, with what happened."""

    snapshot_id: uuid.UUID
    player_id: uuid.UUID
    game_id: uuid.UUID
    team_id: uuid.UUID
    season_code: str
    tip_off_utc: datetime
    prediction_timestamp: datetime
    features: dict[str, Any]
    #: ``True`` if the player recorded playing time for this team in this game.
    played: bool
    #: ``True`` if he was in the box score at all, played or not. Separates
    #: "left out of the squad" from "dressed, never used" in the diagnostics.
    dressed: bool
    #: Games he played among the team's last ``recent_team_games`` this season.
    #: Baseline input only, never a model feature (see the module docstring).
    recent_played: int
    recent_team_games: int
    #: Had not yet played for this club this season at the cutoff.
    not_debuted: bool

    @property
    def label(self) -> float:
        return 1.0 if self.played else 0.0


@dataclass(slots=True)
class DatasetDiagnostics:
    """What the loader saw. Mostly for spotting a broken backfill."""

    games: int = 0
    candidates: int = 0
    rows: int = 0
    played: int = 0
    dressed_not_used: int = 0
    not_in_squad: int = 0
    #: Candidates with no stored snapshot. Non-zero means the backfill predates
    #: candidate snapshots: run ``features backfill`` again.
    missing_snapshots: int = 0
    #: Candidates who turned up in the *other* team's box score: a roster
    #: resolution error, not an availability outcome. Dropped.
    wrong_side: int = 0
    #: Players who played but were not candidates. Inference never projects
    #: them, so no availability model can help; this is the candidate filter's
    #: miss rate.
    played_not_candidate: int = 0

    def as_dict(self) -> dict[str, Any]:
        played_total = self.played + self.played_not_candidate
        return {
            "games": self.games,
            "candidates": self.candidates,
            "rows": self.rows,
            "played": self.played,
            "dressed_not_used": self.dressed_not_used,
            "not_in_squad": self.not_in_squad,
            "base_rate": round(self.played / self.rows, 4) if self.rows else None,
            "missing_snapshots": self.missing_snapshots,
            "wrong_side": self.wrong_side,
            "played_not_candidate": self.played_not_candidate,
            "candidate_recall": (
                round(self.played / played_total, 4) if played_total else None
            ),
        }


async def _finished_games(
    session: AsyncSession, season_codes: list[str] | None
) -> list[tuple[Game, str]]:
    """Finished games with box scores, oldest first, with their season code."""
    has_box_score = (
        select(GameTeam.game_id)
        .join(PlayerGameStats, PlayerGameStats.game_team_id == GameTeam.id)
        .distinct()
    )
    stmt = (
        select(Game, Season.code)
        .join(Season, Season.id == Game.season_id)
        .where(Game.status == GameStatus.FINISHED, Game.id.in_(has_box_score))
        .order_by(Game.tip_off_utc)
    )
    if season_codes:
        stmt = stmt.where(Season.code.in_(season_codes))
    return [(game, code) for game, code in (await session.execute(stmt)).all()]


async def _box_score(
    session: AsyncSession, game_id: uuid.UUID
) -> dict[uuid.UUID, tuple[uuid.UUID, bool]]:
    """``player -> (team, played)`` for one game.

    No ``observed_at`` filter, deliberately: this is the *label*, read after the
    fact, exactly as the stat datasets read their targets.
    """
    rows = (
        await session.execute(
            select(
                PlayerGameStats.player_id,
                GameTeam.team_id,
                PlayerGameStats.did_not_play,
            )
            .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
            .where(GameTeam.game_id == game_id)
        )
    ).all()
    # ``did_not_play`` is the project's one definition of "did not take the
    # floor" (see providers.parsing.is_did_not_play). The candidate filter and the
    # form windows use it too, so the label agrees with both.
    return {
        player_id: (team_id, not did_not_play)
        for player_id, team_id, did_not_play in rows
    }


async def _snapshots(
    session: AsyncSession,
    *,
    game_id: uuid.UUID,
    cutoff: datetime,
    feature_set_version: str,
) -> dict[uuid.UUID, tuple[uuid.UUID, dict[str, Any]]]:
    """``player -> (snapshot id, features)`` for one game at one cutoff."""
    rows = (
        await session.execute(
            select(
                PlayerFeatureSnapshot.player_id,
                PlayerFeatureSnapshot.id,
                PlayerFeatureSnapshot.features,
            ).where(
                PlayerFeatureSnapshot.game_id == game_id,
                PlayerFeatureSnapshot.feature_set_version == feature_set_version,
                PlayerFeatureSnapshot.prediction_timestamp == cutoff,
            )
        )
    ).all()
    return {player_id: (snapshot_id, features or {}) for player_id, snapshot_id, features in rows}


async def load_availability_rows(
    session: AsyncSession,
    *,
    feature_set_version: str = FEATURE_SET_VERSION,
    season_codes: list[str] | None = None,
    settings: Settings | None = None,
) -> tuple[list[AvailabilityRow], DatasetDiagnostics]:
    """One row per (historical game, roster candidate), oldest first.

    Roughly ten queries per game, so a full ten-season load takes a minute or
    two. That is acceptable for a training job and keeps the candidate logic in
    exactly one place rather than re-expressing it as one giant SQL statement
    that could drift from what inference does.
    """
    settings = settings or get_settings()
    diagnostics = DatasetDiagnostics()
    rows: list[AvailabilityRow] = []

    for game, season_code in await _finished_games(session, season_codes):
        diagnostics.games += 1
        cutoff = default_prediction_timestamp(game.tip_off_utc, settings)
        club_by_player = await current_club(session, as_of=cutoff)
        box = await _box_score(session, game.id)
        snapshots = await _snapshots(
            session, game_id=game.id, cutoff=cutoff, feature_set_version=feature_set_version
        )

        candidates_in_game: set[uuid.UUID] = set()
        for team_id in (game.home_team_id, game.away_team_id):
            appearances = await recent_appearances(
                session, team_id=team_id, season_id=game.season_id, as_of=cutoff
            )
            candidates = await candidate_players(
                session,
                team_id=team_id,
                season_id=game.season_id,
                as_of=cutoff,
                club_by_player=club_by_player,
                appearances=appearances,
            )
            for player_id in candidates:
                diagnostics.candidates += 1
                candidates_in_game.add(player_id)

                box_entry = box.get(player_id)
                if box_entry is not None and box_entry[0] != team_id:
                    diagnostics.wrong_side += 1
                    continue

                snapshot = snapshots.get(player_id)
                if snapshot is None:
                    diagnostics.missing_snapshots += 1
                    continue

                dressed = box_entry is not None
                played = bool(box_entry and box_entry[1])
                rows.append(
                    AvailabilityRow(
                        snapshot_id=snapshot[0],
                        player_id=player_id,
                        game_id=game.id,
                        team_id=team_id,
                        season_code=season_code,
                        tip_off_utc=game.tip_off_utc,
                        prediction_timestamp=cutoff,
                        features=snapshot[1],
                        played=played,
                        dressed=dressed,
                        recent_played=appearances.played.get(player_id, 0),
                        recent_team_games=appearances.team_games,
                        not_debuted=player_id not in appearances.debuted,
                    )
                )
                diagnostics.rows += 1
                if played:
                    diagnostics.played += 1
                elif dressed:
                    diagnostics.dressed_not_used += 1
                else:
                    diagnostics.not_in_squad += 1

        diagnostics.played_not_candidate += sum(
            1
            for player_id, (_team, played) in box.items()
            if played and player_id not in candidates_in_game
        )

    logger.info("loaded availability rows", extra=diagnostics.as_dict())
    if diagnostics.missing_snapshots:
        logger.warning(
            "candidates without snapshots were skipped; re-run `features backfill` "
            "so non-appearing candidates get rows too",
            extra={"missing": diagnostics.missing_snapshots},
        )
    return rows, diagnostics


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _clip(probability: float) -> float:
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, float(probability)))


def log_loss(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    """Mean binary cross-entropy, with clipping. ``nan`` for no rows."""
    if not labels:
        return float("nan")
    total = 0.0
    for p, y in zip(probabilities, labels, strict=True):
        q = _clip(p)
        total -= y * math.log(q) + (1.0 - y) * math.log(1.0 - q)
    return total / len(labels)


def brier_score(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error of the probability. ``nan`` for no rows."""
    if not labels:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)


def roc_auc(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum identity, ties averaged.

    ``nan`` when only one class is present, where AUC is undefined.
    """
    pairs = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0])
    positives = sum(1 for _, y in pairs if y >= 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        rank_sum += average_rank * sum(1 for k in range(index, end + 1) if pairs[k][1] >= 0.5)
        index = end + 1

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def reliability_table(
    probabilities: Sequence[float], labels: Sequence[float], *, bins: int = RELIABILITY_BINS
) -> list[dict[str, Any]]:
    """Stated probability against realised frequency, per bin.

    The direct test of whether the probabilities mean what they say: of the
    candidates given ~70%, about 70% should have played.
    """
    counts = [0] * bins
    stated = [0.0] * bins
    hits = [0.0] * bins
    for p, y in zip(probabilities, labels, strict=True):
        index = min(bins - 1, max(0, int(float(p) * bins)))
        counts[index] += 1
        stated[index] += float(p)
        hits[index] += float(y)
    return [
        {
            "band": f"{int(100 * i / bins)}-{int(100 * (i + 1) / bins)}%",
            "rows": counts[i],
            "mean_predicted": round(stated[i] / counts[i], 4),
            "realised": round(hits[i] / counts[i], 4),
        }
        for i in range(bins)
        if counts[i]
    ]


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[float], *, bins: int = RELIABILITY_BINS
) -> float:
    """Row-weighted mean gap between stated and realised, across bins."""
    table = reliability_table(probabilities, labels, bins=bins)
    total = sum(int(row["rows"]) for row in table)
    if not total:
        return float("nan")
    gap = sum(
        float(row["rows"]) * abs(float(row["mean_predicted"]) - float(row["realised"]))
        for row in table
    )
    return gap / total


def team_count_error(
    probabilities: Sequence[float],
    labels: Sequence[float],
    groups: Sequence[tuple[uuid.UUID, uuid.UUID]],
) -> dict[str, float]:
    """How far off the expected number of players used is, per team-game.

    The most readable realism check: summing ``P(plays)`` over a squad gives
    the expected rotation size, and it can be compared directly with how many
    players actually got on the floor. ``bias`` is predicted minus actual, so a
    positive value means we expect more players to be used than are.
    """
    predicted: dict[tuple[uuid.UUID, uuid.UUID], float] = defaultdict(float)
    actual: dict[tuple[uuid.UUID, uuid.UUID], float] = defaultdict(float)
    for p, y, group in zip(probabilities, labels, groups, strict=True):
        predicted[group] += float(p)
        actual[group] += float(y)
    if not predicted:
        return {"team_games": 0, "mae": float("nan"), "bias": float("nan")}
    errors = [predicted[g] - actual[g] for g in predicted]
    return {
        "team_games": len(errors),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Probabilistic accuracy of one predictor on one set of rows."""

    rows: int
    base_rate: float
    mean_predicted: float
    log_loss: float
    brier: float
    auc: float
    ece: float
    team_count_mae: float
    team_count_bias: float

    def as_dict(self) -> dict[str, Any]:
        def r(value: float) -> float | None:
            return round(value, 4) if math.isfinite(value) else None

        return {
            "rows": self.rows,
            "base_rate": r(self.base_rate),
            "mean_predicted": r(self.mean_predicted),
            "log_loss": r(self.log_loss),
            "brier": r(self.brier),
            "auc": r(self.auc),
            "ece": r(self.ece),
            "team_count_mae": r(self.team_count_mae),
            "team_count_bias": r(self.team_count_bias),
        }


def classification_metrics(
    probabilities: Sequence[float],
    labels: Sequence[float],
    groups: Sequence[tuple[uuid.UUID, uuid.UUID]] | None = None,
) -> ClassificationMetrics:
    """Every metric reported for the availability model, in one place."""
    n = len(labels)
    team = (
        team_count_error(probabilities, labels, groups)
        if groups is not None
        else {"mae": float("nan"), "bias": float("nan")}
    )
    return ClassificationMetrics(
        rows=n,
        base_rate=sum(labels) / n if n else float("nan"),
        mean_predicted=sum(probabilities) / n if n else float("nan"),
        log_loss=log_loss(probabilities, labels),
        brier=brier_score(probabilities, labels),
        auc=roc_auc(probabilities, labels),
        ece=expected_calibration_error(probabilities, labels),
        team_count_mae=float(team["mae"]),
        team_count_bias=float(team["bias"]),
    )


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


class AvailabilityBaseline:
    """A simple predictor fitted on the training split only.

    Subclasses define a grouping key; the fitted probability for a key is the
    training rate for that key, shrunk toward the overall rate so a sparse key
    (six rows of "played 0 of 2") cannot produce a confident 0%.
    """

    name = "baseline"
    #: Pseudo-rows of the overall rate mixed into every key. Twenty is enough
    #: to tame sparse keys without washing out well-populated ones.
    prior_strength = 20.0

    def __init__(self) -> None:
        self._overall = 0.5
        self._rates: dict[Any, float] = {}

    def key(self, row: AvailabilityRow) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, rows: Sequence[AvailabilityRow]) -> AvailabilityBaseline:
        if rows:
            self._overall = sum(row.label for row in rows) / len(rows)
        counts: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            bucket = counts[self.key(row)]
            bucket[0] += row.label
            bucket[1] += 1.0
        self._rates = {
            key: (hits + self.prior_strength * self._overall) / (total + self.prior_strength)
            for key, (hits, total) in counts.items()
        }
        return self

    def predict(self, row: AvailabilityRow) -> float:
        return self._rates.get(self.key(row), self._overall)


class BaseRate(AvailabilityBaseline):
    """Everyone gets the league-wide rate. The floor any model must clear."""

    name = "base_rate"

    def key(self, row: AvailabilityRow) -> Any:
        return None


class PlayedLastGame(AvailabilityBaseline):
    """Rate conditional on whether he took the floor in the team's last game."""

    name = "played_last_game"

    def key(self, row: AvailabilityRow) -> Any:
        return row.features.get("player_played_last_game")


class RecentAppearanceRate(AvailabilityBaseline):
    """Rate conditional on "played k of the team's last n games this season".

    Usually the strongest simple predictor, and the one to beat: it is what a
    person would say looking at the last five box scores. Not-yet-debuted
    candidates get their own key, because "0 of 5 because he only just
    signed" and "0 of 5 because he is the fourteenth man" behave differently.
    """

    name = "recent_appearance_rate"

    def key(self, row: AvailabilityRow) -> Any:
        return (row.recent_played, row.recent_team_games, row.not_debuted)


def default_availability_baselines() -> list[AvailabilityBaseline]:
    return [BaseRate(), PlayedLastGame(), RecentAppearanceRate()]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AvailabilityResult:
    """Everything needed to register, judge and reproduce the model."""

    feature_names: list[str]
    params: dict[str, Any]
    best_iteration: int
    train_rows: int
    validation_rows: int
    test_rows: int
    metrics: ClassificationMetrics
    baseline_name: str
    baseline_metrics: ClassificationMetrics
    baselines: dict[str, ClassificationMetrics]
    reliability: list[dict[str, Any]]
    #: Test rows split by whether the player was in the box score at all, so the
    #: two kinds of absence can be judged separately.
    by_outcome: dict[str, Any]
    feature_importance: list[tuple[str, float]]
    training_data_hash: str
    booster: Any = None
    seasons: list[str] = field(default_factory=list)

    @property
    def target(self) -> PredictionTarget:
        return PLAYS

    @property
    def improvement(self) -> float:
        """Fractional log-loss reduction over the best baseline. Positive is better."""
        base = self.baseline_metrics.log_loss
        if not math.isfinite(base) or base == 0:
            return float("nan")
        return (base - self.metrics.log_loss) / base

    @property
    def beats_baseline(self) -> bool:
        """**The gate.** Lower test log loss than the best baseline, same rows."""
        return (
            math.isfinite(self.metrics.log_loss)
            and self.metrics.log_loss < self.baseline_metrics.log_loss
        )

    def summary(self) -> dict[str, Any]:
        return {
            "target": str(PLAYS),
            "model": self.metrics.as_dict(),
            "baseline": self.baseline_name,
            "baseline_metrics": self.baseline_metrics.as_dict(),
            "improvement_pct": (
                round(self.improvement * 100, 2) if math.isfinite(self.improvement) else None
            ),
            "beats_baseline": self.beats_baseline,
            "best_iteration": self.best_iteration,
            "rows": {
                "train": self.train_rows,
                "validation": self.validation_rows,
                "test": self.test_rows,
            },
        }


def chronological_split(
    rows: Sequence[AvailabilityRow],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[list[AvailabilityRow], list[AvailabilityRow], list[AvailabilityRow]]:
    """Train / validation / test by time, with the same fractions as the stat models.

    Split on *game* boundaries, not row boundaries. A row-level cut can land in
    the middle of a game and put half a squad in train and half in test. That
    leaks nothing directly, but it makes the per-team-game count metric
    meaningless on the boundary game.
    """
    ordered = sorted(rows, key=lambda row: (row.tip_off_utc, str(row.game_id)))
    total = len(ordered)
    test_start = total - int(total * test_fraction)
    validation_start = test_start - int(total * validation_fraction)
    too_small = ValueError(
        f"not enough rows to split: {total} rows with "
        f"{validation_fraction:.0%} validation and {test_fraction:.0%} test"
    )
    if validation_start <= 0 or test_start == validation_start or test_start >= total:
        raise too_small

    def snap(index: int) -> int:
        # Move the cut forward to the next game boundary.
        while 0 < index < total and ordered[index].game_id == ordered[index - 1].game_id:
            index += 1
        return index

    validation_start = snap(validation_start)
    test_start = max(validation_start, snap(test_start))
    if test_start in (validation_start, total):
        # Snapping swallowed a whole split. Only happens with a handful of games.
        raise too_small
    return (
        ordered[:validation_start],
        ordered[validation_start:test_start],
        ordered[test_start:],
    )


def derive_feature_names(rows: Sequence[AvailabilityRow]) -> list[str]:
    """Sorted union of snapshot keys, minus row metadata. Same rule as the stat models."""
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    return sorted(names - EXCLUDED_KEYS)


def _hash_rows(rows: Sequence[AvailabilityRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.snapshot_id).encode())
        digest.update(row.prediction_timestamp.isoformat().encode())
        digest.update(b"1" if row.played else b"0")
    return digest.hexdigest()


def _groups(rows: Sequence[AvailabilityRow]) -> list[tuple[uuid.UUID, uuid.UUID]]:
    return [(row.game_id, row.team_id) for row in rows]


def _by_outcome(
    rows: Sequence[AvailabilityRow], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Mean predicted probability for each realised outcome.

    A model that separates the cases should give clearly lower numbers to
    "not in squad" than to "played". If "dressed, not used" looks like
    "played", that is expected: those are coach's decisions made during the
    game, and nothing pre-game can see them.
    """
    buckets: dict[str, list[float]] = {"played": [], "dressed_not_used": [], "not_in_squad": []}
    for row, p in zip(rows, probabilities, strict=True):
        key = "played" if row.played else ("dressed_not_used" if row.dressed else "not_in_squad")
        buckets[key].append(float(p))
    return {
        key: {
            "rows": len(values),
            "mean_predicted": round(sum(values) / len(values), 4) if values else None,
        }
        for key, values in buckets.items()
    }


def train_availability(
    rows: Sequence[AvailabilityRow],
    *,
    params: dict[str, Any] | None = None,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    early_stopping: int = DEFAULT_EARLY_STOPPING,
) -> AvailabilityResult:
    """Train the classifier and judge it against every baseline on the test split.

    Raises:
        ValueError: too few rows, or a split that holds only one class.
    """
    import lightgbm as lgb
    import numpy as np

    resolved = {**DEFAULT_CLASSIFIER_PARAMS, **(params or {})}
    train_rows, validation_rows, test_rows = chronological_split(rows)
    for name, split in (("train", train_rows), ("validation", validation_rows)):
        labels = {row.played for row in split}
        if len(labels) < 2:
            raise ValueError(f"{name} split has only one class; cannot fit a classifier")

    feature_names = derive_feature_names(rows)

    def matrix(split: Sequence[AvailabilityRow]) -> tuple[Any, Any]:
        x = features_to_matrix([row.features for row in split], feature_names)
        y = np.asarray([row.label for row in split], dtype=np.float64)
        return x, y

    x_train, y_train = matrix(train_rows)
    x_validation, y_validation = matrix(validation_rows)
    x_test, _ = matrix(test_rows)

    logger.info(
        "training availability",
        extra={
            "features": len(feature_names),
            "base_rate": round(float(y_train.mean()), 4),
            **matrix_summary(x_train, feature_names),
        },
    )

    booster = lgb.train(
        resolved,
        lgb.Dataset(x_train, label=y_train, feature_name=list(feature_names)),
        num_boost_round=num_rounds,
        valid_sets=[lgb.Dataset(x_validation, label=y_validation)],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    test_labels = [row.label for row in test_rows]
    groups = _groups(test_rows)
    probabilities = [
        float(p) for p in booster.predict(x_test, num_iteration=booster.best_iteration)
    ]
    metrics = classification_metrics(probabilities, test_labels, groups)

    baselines: dict[str, ClassificationMetrics] = {}
    for baseline in default_availability_baselines():
        baseline.fit(train_rows)
        baselines[baseline.name] = classification_metrics(
            [baseline.predict(row) for row in test_rows], test_labels, groups
        )
    baseline_name, baseline_metrics = min(
        baselines.items(),
        key=lambda item: item[1].log_loss if math.isfinite(item[1].log_loss) else math.inf,
    )

    importance = sorted(
        zip(feature_names, booster.feature_importance(importance_type="gain"), strict=True),
        key=lambda item: -item[1],
    )

    result = AvailabilityResult(
        feature_names=list(feature_names),
        params=resolved,
        best_iteration=int(booster.best_iteration or num_rounds),
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        test_rows=len(test_rows),
        metrics=metrics,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        baselines=baselines,
        reliability=reliability_table(probabilities, test_labels),
        by_outcome=_by_outcome(test_rows, probabilities),
        feature_importance=[(name, float(gain)) for name, gain in importance[:30]],
        training_data_hash=_hash_rows(train_rows),
        booster=booster,
        seasons=sorted({row.season_code for row in rows}),
    )

    logger.info("trained availability", extra=result.summary())
    if not result.beats_baseline:
        logger.warning(
            "AVAILABILITY MODEL FAILED THE GATE - does not beat its baseline",
            extra={
                "model_log_loss": round(metrics.log_loss, 4),
                "baseline": baseline_name,
                "baseline_log_loss": round(baseline_metrics.log_loss, 4),
            },
        )
    return result


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def save_availability_artifact(
    result: AvailabilityResult, *, version: str, root: Path | None = None
) -> Path:
    """Booster plus pinned column order. No distribution file: the output *is* one."""
    import json

    root = root or DEFAULT_ARTIFACT_ROOT
    directory = root / MODEL_NAME / version
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.txt"
    result.booster.save_model(str(model_path), num_iteration=result.best_iteration)
    (directory / "columns.json").write_text(
        json.dumps(
            {
                "feature_names": result.feature_names,
                "target": str(PLAYS),
                "kind": "binary_classifier",
                "best_iteration": result.best_iteration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


async def register_availability_model(
    session: AsyncSession,
    result: AvailabilityResult,
    *,
    feature_set_version: str = FEATURE_SET_VERSION,
    artifact_root: Path | None = None,
    clock: Clock | None = None,
    activate: bool | None = None,
) -> ModelVersion:
    """Persist the model and activate it only if it passed the gate."""
    clock = clock or SystemClock()
    version = await next_version(session, MODEL_NAME)
    artifact_path = save_availability_artifact(result, version=version, root=artifact_root)
    should_activate = result.beats_baseline if activate is None else activate

    model_version = ModelVersion(
        id=uuid.uuid4(),
        name=MODEL_NAME,
        version=version,
        target=PLAYS,
        algorithm="lightgbm",
        feature_set_version=feature_set_version,
        hyperparameters={**result.params, "best_iteration": result.best_iteration},
        training_data_hash=result.training_data_hash,
        metrics={
            **result.summary(),
            "baselines": {name: m.as_dict() for name, m in result.baselines.items()},
            "reliability": result.reliability,
            "by_outcome": result.by_outcome,
            "top_features": result.feature_importance[:15],
            "seasons": result.seasons,
            # Read by LoadedModel; a probability is never shifted.
            "calibration_offset": 0.0,
        },
        trained_at=clock.now(),
        artifact_path=str(artifact_path),
        is_active=should_activate,
    )
    session.add(model_version)
    await session.flush()

    if should_activate:
        await session.execute(
            update(ModelVersion)
            .where(ModelVersion.target == PLAYS, ModelVersion.id != model_version.id)
            .values(is_active=False)
        )

    logger.info(
        "registered availability model",
        extra={
            "version": version,
            "active": should_activate,
            "log_loss": round(result.metrics.log_loss, 4),
            "baseline_log_loss": round(result.baseline_metrics.log_loss, 4),
        },
    )
    return model_version


async def train_and_register_availability(
    session: AsyncSession,
    *,
    season_codes: list[str] | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    artifact_root: Path | None = None,
    params: dict[str, Any] | None = None,
    activate: bool | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Load, train, judge and register. Returns a report for the CLI."""
    rows, diagnostics = await load_availability_rows(
        session,
        feature_set_version=feature_set_version,
        season_codes=season_codes,
        settings=settings,
    )
    report: dict[str, Any] = {"dataset": diagnostics.as_dict()}
    if len(rows) < MIN_ROWS:
        report["error"] = f"only {len(rows)} rows; need at least {MIN_ROWS}"
        if diagnostics.missing_snapshots:
            report["error"] += (
                f" ({diagnostics.missing_snapshots} candidates have no snapshot: "
                "run `euroleague-ai features backfill` first)"
            )
        return report

    from euroleague_ai.features.context_v6 import excluded_feature_names

    excluded = excluded_feature_names((settings or get_settings()).excluded_feature_families)
    if excluded:
        from euroleague_ai.ml.ablation import strip

        rows = strip(rows, excluded)
    result = train_availability(rows, params=params)
    model_version = await register_availability_model(
        session,
        result,
        feature_set_version=feature_set_version,
        artifact_root=artifact_root,
        activate=activate,
    )
    report.update(
        {
            "summary": result.summary(),
            "baselines": {name: m.as_dict() for name, m in result.baselines.items()},
            "reliability": result.reliability,
            "by_outcome": result.by_outcome,
            "top_features": result.feature_importance[:15],
            "model": {
                "name": model_version.name,
                "version": model_version.version,
                "active": model_version.is_active,
                "artifact": model_version.artifact_path,
            },
        }
    )
    return report

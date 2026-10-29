"""How much an injury-report label should move P(plays), measured on scored rounds.

"Probable", "questionable", "doubtful" and "ready" used to change no number:
there was no record of how often a EuroLeague player with each label plays, and
an invented discount would have been a guess. After rounds 7 and 8 there is a
record, and it says the model's own probability is far off for these players.
In round 8, players listed probable got a mean P(plays) of 0.33 and 9 of 13
played. The model sees a player who missed recent games; the label says he is
about to return.

**The adjustment** is one shift per label on the log-odds scale::

    logit(P) = logit(P from the model) + offset[label]

so a probable starter and a probable twelfth man move by the same evidence, but
keep their difference. Each offset is fitted by maximum likelihood on every
scored player who had that label, with a normal prior centred on zero (standard
deviation :data:`PRIOR_SD`), so a label seen a handful of times moves the number
only a little. A label seen fewer than :data:`MIN_PLAYERS` times is not adjusted.

**The check** is leave-one-round-out: each round is scored with offsets fitted on
the other rounds only, and the adjustment is switched on only if that held-out
log loss beats the model's own probabilities. With two rounds this is a thin
check, and it gets stronger every round; refit after each one
(``injuries calibrate``).

The model's own probability is kept in each prediction's explanation
(``availability.model_p_plays``), so later fits always start from the model's
number, never from an already-adjusted one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from euroleague_ai.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

#: Labels that get an offset. "ruled out" sets P(plays) to 0 and needs none;
#: a late report never reached the prediction.
CALIBRATED_GROUPS: tuple[str, ...] = (
    "out (news, not applied)",
    "doubtful",
    "questionable",
    "probable",
    "ready",
)

#: Prior standard deviation of an offset, in log-odds. One unit takes 0.33 to
#: 0.57: large moves need real evidence.
PRIOR_SD = 1.0

#: Below this many labelled players, the label is left alone.
MIN_PLAYERS = 5

#: Probabilities are kept inside (EPS, 1 - EPS) on the log-odds scale.
EPS = 1e-4

CALIBRATION_FILE = Path("reconciliation") / "statuses.json"


def calibration_path(root: Path | None = None) -> Path:
    from euroleague_ai.ml.registry import DEFAULT_ARTIFACT_ROOT

    return (root or DEFAULT_ARTIFACT_ROOT) / CALIBRATION_FILE


def _logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True)
class LabelledPlayer:
    """One scored player with a label: what the model said and what happened."""

    group: str
    model_p: float
    played: bool
    #: Groups the leave-one-out check, e.g. ``"E2026:8"``.
    round_key: str


def fit_offset(probabilities: Sequence[float], played: Sequence[bool], *, prior_sd: float) -> float:
    """MAP log-odds shift for one label (Newton's method, concave objective)."""
    logits = [_logit(p) for p in probabilities]
    precision = 1.0 / (prior_sd * prior_sd)
    offset = 0.0
    for _ in range(50):
        gradient = -precision * offset
        curvature = -precision
        for z, y in zip(logits, played, strict=True):
            q = _sigmoid(z + offset)
            gradient += (1.0 if y else 0.0) - q
            curvature -= q * (1.0 - q)
        step = gradient / curvature
        offset -= step
        if abs(step) < 1e-9:
            break
    return offset


def fit_offsets(
    players: Sequence[LabelledPlayer], *, prior_sd: float = PRIOR_SD, min_players: int = MIN_PLAYERS
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in CALIBRATED_GROUPS:
        members = [p for p in players if p.group == group]
        if len(members) < min_players:
            continue
        out[group] = round(
            fit_offset(
                [p.model_p for p in members], [p.played for p in members], prior_sd=prior_sd
            ),
            4,
        )
    return out


def adjusted(p: float, offset: float) -> float:
    if p <= 0.0:
        return 0.0
    return _sigmoid(_logit(p) + offset)


def log_loss(players: Sequence[LabelledPlayer], offsets: dict[str, float]) -> float:
    if not players:
        return float("nan")
    total = 0.0
    for player in players:
        q = min(1.0 - EPS, max(EPS, adjusted(player.model_p, offsets.get(player.group, 0.0))))
        total -= math.log(q) if player.played else math.log(1.0 - q)
    return total / len(players)


@dataclass(slots=True)
class StatusCalibration:
    """Per-label log-odds offsets, and whether they earned their place."""

    offsets: dict[str, float]
    enabled: bool
    #: Per label: players, played, mean model P(plays), mean adjusted P(plays).
    labels: dict[str, dict[str, float]] = field(default_factory=dict)
    rounds: list[str] = field(default_factory=list)
    #: Leave-one-round-out log loss, model alone vs adjusted.
    held_out: dict[str, float] = field(default_factory=dict)
    fitted_at: str | None = None

    def adjust(self, p: float, group: str) -> float:
        if not self.enabled or group not in self.offsets:
            return p
        return adjusted(p, self.offsets[group])

    def to_dict(self) -> dict[str, Any]:
        return {
            "offsets": self.offsets,
            "enabled": self.enabled,
            "labels": self.labels,
            "rounds": self.rounds,
            "held_out": self.held_out,
            "fitted_at": self.fitted_at,
            "prior_sd": PRIOR_SD,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StatusCalibration:
        return cls(
            offsets={k: float(v) for k, v in (payload.get("offsets") or {}).items()},
            enabled=bool(payload.get("enabled")),
            labels=payload.get("labels") or {},
            rounds=list(payload.get("rounds") or []),
            held_out=payload.get("held_out") or {},
            fitted_at=payload.get("fitted_at"),
        )


def calibrate(players: Sequence[LabelledPlayer], *, now: datetime) -> StatusCalibration:
    """Fit offsets on every round, and judge them leaving one round out at a time."""
    rounds = sorted({p.round_key for p in players})
    offsets = fit_offsets(players)

    held_out: dict[str, float] = {}
    enabled = False
    if len(rounds) >= 2 and offsets:
        before = after = 0.0
        scored = 0
        for key in rounds:
            test = [p for p in players if p.round_key == key and p.group in CALIBRATED_GROUPS]
            if not test:
                continue
            fitted = fit_offsets([p for p in players if p.round_key != key])
            before += log_loss(test, {}) * len(test)
            after += log_loss(test, fitted) * len(test)
            scored += len(test)
        if scored:
            held_out = {
                "model": round(before / scored, 4),
                "adjusted": round(after / scored, 4),
                "players": float(scored),
            }
            enabled = held_out["adjusted"] < held_out["model"]

    labels: dict[str, dict[str, float]] = {}
    for group in CALIBRATED_GROUPS:
        members = [p for p in players if p.group == group]
        if not members:
            continue
        offset = offsets.get(group, 0.0)
        labels[group] = {
            "players": float(len(members)),
            "played": float(sum(p.played for p in members)),
            "mean_model_p": round(sum(p.model_p for p in members) / len(members), 4),
            "mean_adjusted_p": round(
                sum(adjusted(p.model_p, offset) for p in members) / len(members), 4
            ),
            "offset": offset,
        }
    return StatusCalibration(
        offsets=offsets,
        enabled=enabled,
        labels=labels,
        rounds=rounds,
        held_out=held_out,
        fitted_at=now.isoformat(),
    )


async def labelled_players(
    session: AsyncSession, *, season_code: str | None = None
) -> list[LabelledPlayer]:
    """Every scored player who had a label at the cutoff, from finished games."""
    from sqlalchemy import exists, select

    from euroleague_ai.ml.injury_check import check_games
    from euroleague_ai.models.competition import Season
    from euroleague_ai.models.enums import GameStatus, PredictionTarget
    from euroleague_ai.models.game import Game
    from euroleague_ai.models.prediction import Prediction

    stmt = (
        select(Game, Season.code)
        .join(Season, Season.id == Game.season_id)
        .where(
            Game.status == GameStatus.FINISHED,
            exists().where(
                Prediction.game_id == Game.id, Prediction.target == PredictionTarget.PLAYS
            ),
        )
        .order_by(Game.tip_off_utc)
    )
    if season_code:
        stmt = stmt.where(Season.code == season_code)
    rows = (await session.execute(stmt)).all()
    round_of = {game.id: f"{code}:{game.round_number}" for game, code in rows}
    checked = await check_games(session, [game for game, _ in rows])
    return [
        LabelledPlayer(
            group=player.group,
            model_p=player.model_plays,
            played=player.played,
            round_key=round_of[player.game_id],
        )
        for player in checked.players
        if player.group in CALIBRATED_GROUPS and player.game_id in round_of
    ]


def save_calibration(calibration: StatusCalibration, root: Path | None = None) -> Path:
    path = calibration_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return path


def load_status_calibration(root: Path | None = None) -> StatusCalibration | None:
    """The saved calibration if it exists and passed its check, else ``None``."""
    path = calibration_path(root)
    if not path.is_file():
        return None
    calibration = StatusCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return calibration if calibration.enabled else None

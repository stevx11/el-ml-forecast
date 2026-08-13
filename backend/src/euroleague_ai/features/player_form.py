"""Player form, role and schedule features.

Rolling, weighted and trend statistics over the player's own prior box scores,
plus starter/usage rates and schedule congestion.

One of six families composed by :mod:`euroleague_ai.features.registry`; the
others cover team strength, opponent strength and teammate availability. News
features are specified in ``docs/ml-roadmap.md`` phase 5.

``tests/integration/test_no_leakage.py`` exercises this builder against a fixture
world containing deliberately extreme post-cutoff rows, which is what makes the
leakage guarantee demonstrable rather than aspirational.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.repositories.history import (
    PlayerGameLine,
    PlayerHistoryRepository,
    TeamHistoryRepository,
)

#: Family-local tag. The authoritative feature set version is
#: euroleague_ai.features.registry.FEATURE_SET_VERSION.
FORM_FAMILY_VERSION = "v1-player-form"

# Rolling window sizes. Long enough to be stable, short enough to react to a role
# change -- EuroLeague rotations shift sharply, and a 20-game window alone would
# lag every one of those shifts by a month.
WINDOWS: tuple[int, ...] = (3, 5, 10, 20)

# Exponential decay half-life in games. Five is roughly one EuroLeague month and
# matches the timescale on which rotations actually change.
EWMA_HALF_LIFE = 5.0

# Per-game quantities averaged over each window. Attempts are included alongside
# makes because attempt volume is the usage signal; makes conflate usage with
# shooting variance, which is far noisier.
_METRICS: dict[str, str] = {
    "minutes": "minutes",
    "points": "points",
    "rebounds": "total_rebounds",
    "assists": "assists",
    "fg3m": "fg3m",
    "fg3a": "fg3a",
    "fga": "field_goal_attempts",
    "fta": "fta",
    "steals": "steals",
    "blocks": "blocks_favour",
    "turnovers": "turnovers",
    "valuation": "valuation",
}


def _values(lines: list[PlayerGameLine], attribute: str) -> list[float]:
    return [float(getattr(line, attribute)) for line in lines]


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` when there is nothing to average.

    ``None``, never ``0.0``. A rookie's ten-game average is *unknown*, and
    LightGBM handles genuine NaN natively. Substituting zero would teach the
    model that players without history score nothing, which is both false and
    exactly backwards for a highly-touted signing.
    """
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    """Sample standard deviation. Needs at least two observations."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ewma(values: list[float], *, half_life: float = EWMA_HALF_LIFE) -> float | None:
    """Exponentially weighted mean over ``values`` ordered newest first."""
    if not values:
        return None
    decay = 0.5 ** (1.0 / half_life)
    weighted = 0.0
    total_weight = 0.0
    for index, value in enumerate(values):
        weight = decay**index
        weighted += weight * value
        total_weight += weight
    return weighted / total_weight if total_weight else None


def _trend(values: list[float]) -> float | None:
    """Least-squares slope per game over ``values`` ordered newest first.

    Sign convention: positive means improving. Because the input is newest-first,
    the raw slope against index is negated so that a rising recent trend reads as
    a positive number, which is what an explanation like "recent scoring trend +"
    needs.
    """
    count = len(values)
    if count < 3:
        return None
    mean_x = (count - 1) / 2.0
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator == 0:
        return None
    return -(numerator / denominator)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


async def build_player_form_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
) -> dict[str, float | None]:
    """Compute form and schedule features for one player as of one instant.

    ``as_of`` is a mandatory keyword argument with no default anywhere in this
    layer. You cannot compute a feature without stating what the model is allowed
    to see, and every underlying query filters ``observed_at <= as_of``.

    Args:
        session: Database session.
        player_id: Player to build features for.
        team_id: The team they are playing for in the upcoming game.
        as_of: Cutoff. Nothing recorded after this instant may influence the result.
        season_id: Restricts season-scoped features to one season.

    Returns:
        Feature name to value. ``None`` denotes genuinely missing data.
    """
    as_of = ensure_utc(as_of, field="as_of")

    players = PlayerHistoryRepository(session)
    teams = TeamHistoryRepository(session)

    history = await players.recent_games(
        player_id, as_of=as_of, limit=max(WINDOWS), season_id=season_id
    )
    season_history = await players.recent_games(
        player_id, as_of=as_of, limit=200, season_id=season_id
    )

    features: dict[str, float | None] = {"games_played_season": float(len(season_history))}

    for name, attribute in _METRICS.items():
        season_values = _values(season_history, attribute)
        features[f"season_{name}_mean"] = _round(_mean(season_values))

        window_values = _values(history, attribute)
        for window in WINDOWS:
            features[f"last{window}_{name}_mean"] = _round(_mean(window_values[:window]))

        features[f"ewma_{name}"] = _round(_ewma(window_values))
        features[f"std10_{name}"] = _round(_stdev(window_values[:10]))
        features[f"trend10_{name}"] = _round(_trend(window_values[:10]))

    features.update(_role_features(history))
    features.update(
        await _schedule_features(players, teams, player_id, team_id, as_of=as_of)
    )
    return features


def _role_features(history: list[PlayerGameLine]) -> dict[str, float | None]:
    """Starter status and per-40-minute usage rates."""
    if not history:
        return {
            "starter_last_game": None,
            "starter_rate_last5": None,
            "starter_rate_recent": None,
            "fga_per40": None,
            "fg3a_rate": None,
            "fta_rate": None,
        }

    total_minutes = sum(line.minutes for line in history)
    total_fga = sum(line.field_goal_attempts for line in history)
    total_fg3a = sum(line.fg3a for line in history)
    total_fta = sum(line.fta for line in history)

    return {
        "starter_last_game": float(history[0].is_starter),
        "starter_rate_last5": _round(
            _mean([float(line.is_starter) for line in history[:5]])
        ),
        "starter_rate_recent": _round(
            _mean([float(line.is_starter) for line in history])
        ),
        # Per-40 rather than per-game: it separates role from playing time, which
        # is what makes a bench player's usage comparable to a starter's.
        "fga_per40": _round(total_fga * 40.0 / total_minutes) if total_minutes else None,
        "fg3a_rate": _round(total_fg3a / total_fga) if total_fga else None,
        "fta_rate": _round(total_fta / total_fga) if total_fga else None,
    }


async def _schedule_features(
    players: PlayerHistoryRepository,
    teams: TeamHistoryRepository,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    *,
    as_of: datetime,
) -> dict[str, float | None]:
    """Rest, congestion and recent load.

    Disproportionately important in Europe: double-round weeks compress two
    EuroLeague games plus a domestic fixture into six days, and rotations
    visibly shorten under that load.
    """
    last_team_game = await teams.last_game_before(team_id, as_of=as_of)
    days_rest = (as_of - last_team_game).total_seconds() / 86400.0 if last_team_game else None

    games_7d = await players.games_played_between(
        player_id, as_of=as_of, since=as_of - timedelta(days=7)
    )
    games_14d = await players.games_played_between(
        player_id, as_of=as_of, since=as_of - timedelta(days=14)
    )

    recent = await players.recent_games(player_id, as_of=as_of, limit=5)
    minutes_7d = sum(
        line.minutes for line in recent if line.tip_off_utc >= as_of - timedelta(days=7)
    )

    return {
        "days_rest": _round(days_rest, 3),
        "games_last_7d": float(games_7d),
        "games_last_14d": float(games_14d),
        "minutes_last_7d": _round(minutes_7d),
        "previous_game_minutes": _round(recent[0].minutes) if recent else None,
        # Two days or less between games is the threshold at which EuroLeague
        # coaches visibly manage minutes.
        "is_short_rest": float(days_rest < 2.0) if days_rest is not None else None,
    }

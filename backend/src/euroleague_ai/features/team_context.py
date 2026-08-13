"""Team and opponent strength features.

The single most important rule in this module: every rating is computed from the
games that had actually been played **before** ``as_of``, never from a
precomputed season aggregate.

That distinction is the classic leak in basketball modelling. Computing each
team's season-long defensive rating once and reusing it for every game in that
season means a March defensive rating informs an October prediction. The model
looks excellent offline and is useless in production, because at prediction time
the season rating does not exist yet.

Ratings are per-100-possessions so they are comparable across the pace
differences between, say, a Zalgiris game and a Baskonia game.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.repositories.history import TeamGameLine, TeamHistoryRepository

#: How many prior games feed the season-to-date ratings. Twenty is most of a
#: EuroLeague regular season and keeps early-season rows from being dominated by
#: a single blowout.
SEASON_WINDOW = 20

#: Short window for form. Five games is roughly a month in EuroLeague.
FORM_WINDOW = 5

#: League-average possessions per 40 minutes, used only to sanity-bound derived
#: ratings when possession data is missing.
FALLBACK_POSSESSIONS = 72.0

#: Below this many games in the current season, team ratings fall back to recent
#: history regardless of season. Five is enough for a rating to mean something;
#: below it the sample is noise and last season's rating is the better estimate,
#: even accounting for summer roster churn.
MIN_SEASON_GAMES = 5


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Divide, returning ``None`` rather than raising or silently yielding zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _possessions(lines: list[TeamGameLine]) -> float:
    """Total possessions across ``lines``, falling back to a league average."""
    total = 0.0
    for line in lines:
        total += line.possessions if line.possessions is not None else FALLBACK_POSSESSIONS
    return total


def summarise_team(lines: list[TeamGameLine], prefix: str) -> dict[str, float | None]:
    """Turn a team's prior games into rating features.

    Args:
        lines: Prior games, newest first, already filtered by ``as_of``.
        prefix: Feature name prefix, ``"team"`` or ``"opp"``.

    Returns:
        Feature name to value. ``None`` where there is no history to compute from
        -- an unknown rating must stay unknown rather than defaulting to a
        league-average-looking zero.
    """
    if not lines:
        return {
            f"{prefix}_games_played": 0.0,
            f"{prefix}_off_rating": None,
            f"{prefix}_def_rating": None,
            f"{prefix}_net_rating": None,
            f"{prefix}_pace": None,
            f"{prefix}_points_scored": None,
            f"{prefix}_points_allowed": None,
            f"{prefix}_oreb_per_game": None,
            f"{prefix}_dreb_per_game": None,
            f"{prefix}_turnovers_per_game": None,
            f"{prefix}_fg3a_per_game": None,
            f"{prefix}_fta_per_game": None,
            f"{prefix}_win_rate": None,
            f"{prefix}_home_rate": None,
            f"{prefix}_off_rating_last5": None,
            f"{prefix}_def_rating_last5": None,
            f"{prefix}_pace_last5": None,
        }

    window = lines[:SEASON_WINDOW]
    count = float(len(window))
    possessions = _possessions(window)

    scored = sum(line.points for line in window)
    allowed = sum(line.points_allowed for line in window)

    # Per 100 possessions, the standard normalisation. Without it a fast team
    # looks like a good offence and a good defence simultaneously.
    off_rating = _safe_div(scored * 100.0, possessions)
    def_rating = _safe_div(allowed * 100.0, possessions)

    paces = [line.pace for line in window if line.pace is not None]
    recent = lines[:FORM_WINDOW]
    recent_possessions = _possessions(recent)
    recent_paces = [line.pace for line in recent if line.pace is not None]

    return {
        f"{prefix}_games_played": count,
        f"{prefix}_off_rating": _round(off_rating),
        f"{prefix}_def_rating": _round(def_rating),
        f"{prefix}_net_rating": _round(
            off_rating - def_rating if off_rating is not None and def_rating is not None else None
        ),
        f"{prefix}_pace": _round(sum(paces) / len(paces)) if paces else None,
        f"{prefix}_points_scored": _round(scored / count),
        f"{prefix}_points_allowed": _round(allowed / count),
        f"{prefix}_oreb_per_game": _round(
            sum(line.offensive_rebounds for line in window) / count
        ),
        f"{prefix}_dreb_per_game": _round(
            sum(line.defensive_rebounds for line in window) / count
        ),
        f"{prefix}_turnovers_per_game": _round(sum(line.turnovers for line in window) / count),
        f"{prefix}_fg3a_per_game": _round(sum(line.fg3a for line in window) / count),
        f"{prefix}_fta_per_game": _round(sum(line.fta for line in window) / count),
        f"{prefix}_win_rate": _round(
            sum(1.0 for line in window if line.points > line.points_allowed) / count
        ),
        f"{prefix}_home_rate": _round(sum(1.0 for line in window if line.is_home) / count),
        # Recent form, which reacts to a coaching change far faster than the
        # season-to-date figure does.
        f"{prefix}_off_rating_last5": _round(
            _safe_div(sum(line.points for line in recent) * 100.0, recent_possessions)
        ),
        f"{prefix}_def_rating_last5": _round(
            _safe_div(sum(line.points_allowed for line in recent) * 100.0, recent_possessions)
        ),
        f"{prefix}_pace_last5": (
            _round(sum(recent_paces) / len(recent_paces)) if recent_paces else None
        ),
    }


async def build_team_context_features(
    session: AsyncSession,
    *,
    team_id: uuid.UUID,
    opponent_team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
    is_home: bool | None = None,
) -> dict[str, float | None]:
    """Team and opponent strength as of ``as_of``.

    ``as_of`` is mandatory and keyword-only, as everywhere in the feature layer.
    Both teams' histories are read through
    :class:`~euroleague_ai.repositories.history.TeamHistoryRepository`, which
    applies the ``observed_at <= as_of`` filter.
    """
    as_of = ensure_utc(as_of, field="as_of")
    teams = TeamHistoryRepository(session)

    # Same reasoning as the player windows: a club with no games this season is
    # not a club of unknown strength, it is last season's club minus whoever
    # left. Falling back to recent history beats handing the model nulls for
    # every fixture until November.
    own = await teams.recent_games(
        team_id, as_of=as_of, limit=SEASON_WINDOW, season_id=season_id
    )
    if len(own) < MIN_SEASON_GAMES:
        own = await teams.recent_games(team_id, as_of=as_of, limit=SEASON_WINDOW)

    opponent = await teams.recent_games(
        opponent_team_id, as_of=as_of, limit=SEASON_WINDOW, season_id=season_id
    )
    if len(opponent) < MIN_SEASON_GAMES:
        opponent = await teams.recent_games(
            opponent_team_id, as_of=as_of, limit=SEASON_WINDOW
        )

    features: dict[str, float | None] = {}
    features.update(summarise_team(own, "team"))
    features.update(summarise_team(opponent, "opp"))

    # Matchup deltas. A model can derive these from the raw ratings, but stating
    # them explicitly gives tree splits something directly meaningful to cut on.
    team_off = features["team_off_rating"]
    opp_def = features["opp_def_rating"]
    if team_off is not None and opp_def is not None:
        features["matchup_off_vs_def"] = _round(team_off - opp_def)
    else:
        features["matchup_off_vs_def"] = None

    team_pace = features["team_pace"]
    opp_pace = features["opp_pace"]
    if team_pace is not None and opp_pace is not None:
        # Expected game pace drives every counting stat: more possessions means
        # more of everything.
        features["matchup_expected_pace"] = _round((team_pace + opp_pace) / 2.0)
    else:
        features["matchup_expected_pace"] = None

    features["is_home"] = None if is_home is None else float(is_home)
    return features

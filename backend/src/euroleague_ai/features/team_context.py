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

#: Margin at which a game is treated as a full blowout for the saturating risk
#: feature. Fifteen points is roughly where EuroLeague rotations stop being
#: competitive decisions; beyond it, coaches are managing minutes rather than
#: chasing the result, and a thirty-point gap does not shorten a bench twice as
#: much as a fifteen-point one.
BLOWOUT_MARGIN = 15.0

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
        f"{prefix}_oreb_per_game": _round(sum(line.offensive_rebounds for line in window) / count),
        f"{prefix}_dreb_per_game": _round(sum(line.defensive_rebounds for line in window) / count),
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
    own = await teams.recent_games(team_id, as_of=as_of, limit=SEASON_WINDOW, season_id=season_id)
    if len(own) < MIN_SEASON_GAMES:
        own = await teams.recent_games(team_id, as_of=as_of, limit=SEASON_WINDOW)

    opponent = await teams.recent_games(
        opponent_team_id, as_of=as_of, limit=SEASON_WINDOW, season_id=season_id
    )
    if len(opponent) < MIN_SEASON_GAMES:
        opponent = await teams.recent_games(opponent_team_id, as_of=as_of, limit=SEASON_WINDOW)

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
    features.update(game_script_features(features))
    return features


def game_script_features(features: dict[str, float | None]) -> dict[str, float | None]:
    """How the game itself is likely to unfold, and what that does to minutes.

    **Why this exists.** The chain experiment measured minutes MAE at 4.43
    against a mean of roughly twenty played minutes -- a 22% error on the
    quantity every other projection is downstream of. Nothing in the feature set
    told the model whether a game would be close, and closeness is exactly what
    decides whether a starter plays the fourth quarter. A twenty-point game
    empties both benches: the winning coach rests his rotation and the losing
    one has nothing left to save them for. The two effects push in the same
    direction, which is why the signal is the *absolute* margin rather than the
    signed one.

    **How the estimate is built.** A team's scoring rate in a given game is
    approximated by averaging its own offensive rating with the opponent's
    defensive rating -- the standard simple projection, and one that needs no
    league-average constant. Scaled by expected possessions, that gives expected
    points for each side, hence a total and a margin.

    Deliberately no home-court constant. The size of home advantage is an
    empirical quantity that varies by league and season, and hardcoding a guess
    would bury it where nobody could audit it. ``is_home`` is already a feature,
    so the model can learn the interaction from data instead.

    Every input here is already filtered by ``as_of`` upstream, so this function
    is pure arithmetic over numbers we were entitled to know.
    """
    team_off = features.get("team_off_rating")
    team_def = features.get("team_def_rating")
    opp_off = features.get("opp_off_rating")
    opp_def = features.get("opp_def_rating")
    pace = features.get("matchup_expected_pace")

    blank: dict[str, float | None] = {
        "game_expected_total": None,
        "game_expected_margin": None,
        "game_expected_abs_margin": None,
        "game_blowout_risk": None,
        "game_is_favourite": None,
        "game_expected_team_points": None,
        "game_expected_opp_points": None,
    }
    if None in (team_off, team_def, opp_off, opp_def):
        return blank

    # Possessions. Falls back to the league-average constant already used for
    # ratings when neither side has a measured pace yet.
    possessions = pace if pace is not None else FALLBACK_POSSESSIONS

    # Each side's expected rating is the midpoint of its own attack and the
    # other's defence: a good offence against a good defence lands in between.
    team_rating = (team_off + opp_def) / 2.0  # type: ignore[operator]
    opp_rating = (opp_off + team_def) / 2.0  # type: ignore[operator]

    team_points = team_rating * possessions / 100.0
    opp_points = opp_rating * possessions / 100.0
    margin = team_points - opp_points

    return {
        # Opportunity: more expected points means more of everything to go round.
        "game_expected_total": _round(team_points + opp_points),
        # Signed, from this player's team's perspective. Positive means favoured.
        "game_expected_margin": _round(margin),
        # The blowout signal. Unsigned on purpose -- benches empty at both ends
        # of a rout, so what matters is the gap, not who is ahead.
        "game_expected_abs_margin": _round(abs(margin)),
        # A bounded version of the same thing. Trees can split on the raw margin
        # perfectly well, but a saturating transform means "comfortable" and
        # "utter rout" stop looking like proportionally different worlds:
        # rotations shorten sharply up to about fifteen points and then stop
        # changing much.
        "game_blowout_risk": _round(min(1.0, abs(margin) / BLOWOUT_MARGIN)),
        "game_is_favourite": 1.0 if margin > 0 else 0.0,
        "game_expected_team_points": _round(team_points),
        "game_expected_opp_points": _round(opp_points),
    }

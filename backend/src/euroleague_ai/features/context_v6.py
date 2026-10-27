"""Feature set v6: four families the v5 vector had no way to see.

Every feature here is new; no v5 feature changes. That is what lets the v5
models keep serving while v6 is backfilled (see
:data:`~euroleague_ai.features.registry.SERVES`): a v6 vector holds every v5
column with the same meaning, plus these.

=====================  =======================================================
``appearances``        How often he played in the team's last five games this
                       season, and whether he has played for this club at all.
                       P(plays) beat these counts as a baseline by 13.7%
                       without seeing them; v6 lets it use them.
``opponent_position``  What this opponent has allowed to players of his
                       position group over its last ten games: points,
                       assists, threes, rebounds per game, and the group's
                       share of all points allowed. A guard facing a team that
                       stops guards should project lower than his form says.
``coach``              Games the team's current coach has been in charge, and
                       whether that is five or fewer. New coaches reshape
                       rotations within days.
``schedule_venue``     Days until the team's next game, a flag for one within
                       three days, a flag for play-in, playoffs and Final Four,
                       and how his points and minutes differ at this venue
                       (home or away) from his overall average.
=====================  =======================================================

**Knowability, family by family.**

* Appearances and opponent defence read finished games through box scores
  observed by the cutoff, like every v5 family.
* The coach comes from ``game_teams``, which has no observed time of its own:
  it is filled in when the box score arrives. A game counts only once its team
  box score (``team_game_stats.observed_at``) was knowable at the cutoff.
* The next fixture is read from ``games`` filtered on ``observed_at``.
  Fixtures are knowable seven days ahead (:mod:`euroleague_ai.ingestion.knowability`),
  in the backfill and live alike, so "next game in two days" means the same
  thing in training and in production.
* The phase is a property of the game being predicted, published with the
  fixture.
* Positions come from the stint table, never the mutable ``players.position``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import aliased

from euroleague_ai.models.enums import GameStatus
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.people import PlayerTeamStint
from euroleague_ai.repositories.history import PlayerHistoryRepository
from euroleague_ai.repositories.rosters import recent_appearances

if TYPE_CHECKING:  # pragma: no cover - typing only
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

#: Feature names per family, for ablation and explanations. Order is stable.
V6_FAMILIES: dict[str, tuple[str, ...]] = {
    "appearances": ("recent_played_share", "recent_team_games", "not_debuted_for_club"),
    "opponent_position": (
        "opp_pos_points_pg",
        "opp_pos_assists_pg",
        "opp_pos_fg3m_pg",
        "opp_pos_rebounds_pg",
        "opp_pos_points_share",
    ),
    "coach": ("coach_games_in_charge", "coach_changed_recently"),
    "schedule_venue": (
        "days_to_next_team_game",
        "next_game_within_3d",
        "is_postseason",
        "venue_points_diff",
        "venue_minutes_diff",
    ),
}

V6_FEATURE_NAMES: tuple[str, ...] = tuple(n for names in V6_FAMILIES.values() for n in names)

#: Opponent games the positional-defence family looks back over. Ten is about a
#: month of EuroLeague: long enough to average out one hot shooter, short
#: enough to notice a defence that changed.
OPPONENT_WINDOW = 10
#: Coach tenure is counted up to this many games; beyond it, "settled".
COACH_WINDOW = 30
#: "Recently changed" means in charge for this many games or fewer.
COACH_RECENT = 5
#: Player games read for the home/away split, across seasons.
VENUE_WINDOW = 40
#: Games needed at a venue before its split means anything.
VENUE_MIN_GAMES = 5
#: How far ahead the next fixture is looked for. Fixtures are knowable seven
#: days ahead, so anything beyond is "not known yet", not "no game".
NEXT_GAME_HORIZON = timedelta(days=7)
#: Phase codes of the regular season. Anything else is play-in, playoffs or
#: the Final Four, where rotations tighten.
REGULAR_SEASON_PHASES = frozenset({"RS", "TS"})

_POSITION_GROUPS = ("GUARD", "FORWARD", "CENTER")


def excluded_feature_names(families: list[str]) -> frozenset[str]:
    """Feature names of the v6 families to leave out of training.

    Raises:
        ValueError: on a family name that does not exist, rather than silently
            training with it in.
    """
    unknown = [family for family in families if family not in V6_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown feature famil{'y' if len(unknown) == 1 else 'ies'} {', '.join(unknown)}; "
            f"choose from {', '.join(V6_FAMILIES)}"
        )
    return frozenset(name for family in families for name in V6_FAMILIES[family])


def _blank(family: str) -> dict[str, float | None]:
    return dict.fromkeys(V6_FAMILIES[family])


async def _appearance_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    season_id: uuid.UUID | None,
    as_of: datetime,
    cache: dict[Any, Any] | None,
) -> dict[str, float | None]:
    if season_id is None:
        return _blank("appearances")
    key = ("v6_appearances", team_id, season_id, as_of)
    if cache is not None and key in cache:
        recent = cache[key]
    else:
        recent = await recent_appearances(
            session, team_id=team_id, season_id=season_id, as_of=as_of
        )
        if cache is not None:
            cache[key] = recent
    if recent.team_games == 0:
        # Before the team's first game nobody has debuted, and nobody has a
        # share: unknown, not zero.
        return _blank("appearances")
    return {
        "recent_played_share": recent.played.get(player_id, 0) / recent.team_games,
        "recent_team_games": float(recent.team_games),
        "not_debuted_for_club": 0.0 if player_id in recent.debuted else 1.0,
    }


async def _opponent_allowed(
    session: AsyncSession, opponent_team_id: uuid.UUID, as_of: datetime
) -> dict[str, dict[str, float]] | None:
    """Per position group, per game: what players facing this opponent produced."""
    recent = (
        select(Game.id.label("game_id"), Game.season_id.label("season_id"))
        .join(GameTeam, GameTeam.game_id == Game.id)
        .where(
            GameTeam.team_id == opponent_team_id,
            Game.observed_at <= as_of,
            Game.tip_off_utc < as_of,
            Game.status == GameStatus.FINISHED,
        )
        .order_by(Game.tip_off_utc.desc())
        .limit(OPPONENT_WINDOW)
        .subquery()
    )
    facing = aliased(GameTeam)
    lines = (
        await session.execute(
            select(
                recent.c.game_id,
                recent.c.season_id,
                PlayerGameStats.player_id,
                facing.team_id,
                PlayerGameStats.points,
                PlayerGameStats.total_rebounds,
                PlayerGameStats.assists,
                PlayerGameStats.fg3m,
            )
            .select_from(recent)
            .join(
                facing,
                and_(facing.game_id == recent.c.game_id, facing.team_id != opponent_team_id),
            )
            .join(PlayerGameStats, PlayerGameStats.game_team_id == facing.id)
            .where(PlayerGameStats.observed_at <= as_of, PlayerGameStats.did_not_play.is_(False))
        )
    ).all()
    if not lines:
        return None
    games = {line.game_id for line in lines}

    # Each opposing player's position with the club he played for, as known then.
    pairs = {(line.player_id, line.team_id) for line in lines}
    stints = (
        await session.execute(
            select(
                PlayerTeamStint.player_id,
                PlayerTeamStint.team_id,
                PlayerTeamStint.season_id,
                PlayerTeamStint.position,
                PlayerTeamStint.effective_from,
            )
            .where(
                or_(
                    *(
                        and_(PlayerTeamStint.player_id == p, PlayerTeamStint.team_id == t)
                        for p, t in pairs
                    )
                ),
                PlayerTeamStint.observed_at <= as_of,
                PlayerTeamStint.effective_from <= as_of,
            )
            .order_by(PlayerTeamStint.effective_from)
        )
    ).all()
    by_season: dict[tuple[Any, Any, Any], str] = {}
    by_club: dict[tuple[Any, Any], str] = {}
    for player, team, season, position, _ in stints:  # ascending: latest wins
        by_season[(player, team, season)] = str(position)
        by_club[(player, team)] = str(position)

    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"points": 0.0, "assists": 0.0, "fg3m": 0.0, "rebounds": 0.0}
    )
    all_points = 0.0
    for line in lines:
        group = by_season.get((line.player_id, line.team_id, line.season_id)) or by_club.get(
            (line.player_id, line.team_id)
        )
        all_points += line.points or 0
        if group not in _POSITION_GROUPS:
            continue
        cell = totals[group]
        cell["points"] += line.points or 0
        cell["assists"] += line.assists or 0
        cell["fg3m"] += line.fg3m or 0
        cell["rebounds"] += line.total_rebounds or 0

    count = float(len(games))
    out: dict[str, dict[str, float]] = {}
    for group in _POSITION_GROUPS:
        cell = totals.get(group, {"points": 0.0, "assists": 0.0, "fg3m": 0.0, "rebounds": 0.0})
        out[group] = {
            "points": cell["points"] / count,
            "assists": cell["assists"] / count,
            "fg3m": cell["fg3m"] / count,
            "rebounds": cell["rebounds"] / count,
            "share": cell["points"] / all_points if all_points else 0.0,
        }
    return out


async def _opponent_position_features(
    session: AsyncSession,
    *,
    opponent_team_id: uuid.UUID,
    position: str | None,
    as_of: datetime,
    cache: dict[Any, Any] | None,
) -> dict[str, float | None]:
    if position not in _POSITION_GROUPS:
        return _blank("opponent_position")
    key = ("v6_opponent", opponent_team_id, as_of)
    if cache is not None and key in cache:
        allowed = cache[key]
    else:
        allowed = await _opponent_allowed(session, opponent_team_id, as_of)
        if cache is not None:
            cache[key] = allowed
    if allowed is None:
        return _blank("opponent_position")
    group = allowed[position]
    return {
        "opp_pos_points_pg": round(group["points"], 4),
        "opp_pos_assists_pg": round(group["assists"], 4),
        "opp_pos_fg3m_pg": round(group["fg3m"], 4),
        "opp_pos_rebounds_pg": round(group["rebounds"], 4),
        "opp_pos_points_share": round(group["share"], 4),
    }


async def _coach_features(
    session: AsyncSession, *, team_id: uuid.UUID, as_of: datetime, cache: dict[Any, Any] | None
) -> dict[str, float | None]:
    key = ("v6_coach", team_id, as_of)
    if cache is not None and key in cache:
        return dict(cache[key])
    coaches = list(
        (
            await session.execute(
                select(GameTeam.coach_id)
                .join(Game, Game.id == GameTeam.game_id)
                # The coach is written with the box score: only games whose
                # box score was knowable at the cutoff count.
                .join(TeamGameStats, TeamGameStats.game_team_id == GameTeam.id)
                .where(
                    GameTeam.team_id == team_id,
                    TeamGameStats.observed_at <= as_of,
                    Game.tip_off_utc < as_of,
                    Game.status == GameStatus.FINISHED,
                )
                .order_by(Game.tip_off_utc.desc())
                .limit(COACH_WINDOW)
            )
        ).scalars()
    )
    if not coaches or coaches[0] is None:
        result = _blank("coach")
    else:
        current = coaches[0]
        tenure = 0
        for coach in coaches:
            if coach != current:
                break
            tenure += 1
        changed = tenure < len(coaches) and tenure <= COACH_RECENT
        result = {
            "coach_games_in_charge": float(tenure),
            "coach_changed_recently": 1.0 if changed else 0.0,
        }
    if cache is not None:
        cache[key] = dict(result)
    return result


async def _next_tip_off(
    session: AsyncSession, *, team_id: uuid.UUID, game: Game, as_of: datetime
) -> datetime | None:
    """The team's next fixture after ``game``, if it was known at the cutoff."""
    return (
        await session.execute(
            select(Game.tip_off_utc)
            .where(
                or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                Game.tip_off_utc > game.tip_off_utc,
                Game.tip_off_utc <= game.tip_off_utc + NEXT_GAME_HORIZON,
                Game.observed_at <= as_of,
            )
            .order_by(Game.tip_off_utc)
            .limit(1)
        )
    ).scalar_one_or_none()


async def _schedule_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    as_of: datetime,
    game: Game | None,
    is_home: bool | None,
    cache: dict[Any, Any] | None = None,
) -> dict[str, float | None]:
    features = _blank("schedule_venue")
    if game is not None:
        key = ("v6_next_game", team_id, game.id, as_of)
        if cache is not None and key in cache:
            next_tip = cache[key]
        else:
            next_tip = await _next_tip_off(session, team_id=team_id, game=game, as_of=as_of)
            if cache is not None:
                cache[key] = next_tip
        if next_tip is not None:
            days = (next_tip - game.tip_off_utc).total_seconds() / 86_400
            features["days_to_next_team_game"] = round(days, 2)
            features["next_game_within_3d"] = 1.0 if days <= 3.0 else 0.0
        else:
            # No fixture within the knowable horizon: not soon.
            features["next_game_within_3d"] = 0.0
        if game.phase_code:
            features["is_postseason"] = (
                0.0 if game.phase_code.upper() in REGULAR_SEASON_PHASES else 1.0
            )

    if is_home is not None:
        lines = await PlayerHistoryRepository(session).recent_games(
            player_id, as_of=as_of, limit=VENUE_WINDOW
        )
        here = [line for line in lines if line.is_home == is_home]
        if len(here) >= VENUE_MIN_GAMES and len(lines) > len(here):

            def mean(values: list[float]) -> float:
                return sum(values) / len(values)

            features["venue_points_diff"] = round(
                mean([float(x.points) for x in here]) - mean([float(x.points) for x in lines]), 4
            )
            features["venue_minutes_diff"] = round(
                mean([x.minutes for x in here]) - mean([x.minutes for x in lines]), 4
            )
    return features


async def build_v6_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    opponent_team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None,
    game: Game | None,
    is_home: bool | None,
    position: str | None,
    cache: dict[Any, Any] | None = None,
) -> dict[str, float | None]:
    """Every v6 feature for one player-game, as of the cutoff."""
    features: dict[str, float | None] = {}
    features.update(
        await _appearance_features(
            session,
            player_id=player_id,
            team_id=team_id,
            season_id=season_id,
            as_of=as_of,
            cache=cache,
        )
    )
    features.update(
        await _opponent_position_features(
            session, opponent_team_id=opponent_team_id, position=position, as_of=as_of, cache=cache
        )
    )
    features.update(await _coach_features(session, team_id=team_id, as_of=as_of, cache=cache))
    features.update(
        await _schedule_features(
            session,
            player_id=player_id,
            team_id=team_id,
            as_of=as_of,
            game=game,
            is_home=is_home,
            cache=cache,
        )
    )
    return features

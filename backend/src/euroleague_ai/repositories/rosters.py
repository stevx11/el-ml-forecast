"""Who could play for a club at a given instant.

Moved here from ``ml/inference.py`` once a second caller needed it. Inference
uses it to decide *who to predict for*; the snapshot backfill and the
availability model use it to rebuild exactly that population for historical
games. If those two paths answered differently, the play-probability model
would be trained on one population and served on another. The cheapest way to
stop that is one implementation that both import.

**Roster resolution must be transfer-aware.** Candidates come from the stint
intervals, but "registered to this club in this season" is not enough. Across a
transfer window a player can hold a registration for the season just finished
*and* one for the season starting, at different clubs. Filtering by season
alone returns both, and the player gets projected for a club he has left.
:func:`current_club` handles this: among registrations in effect at the cutoff,
the latest ``effective_from`` wins.

Every query takes ``as_of`` and applies ``observed_at <= as_of``, exactly like
:mod:`~euroleague_ai.repositories.history`. A roster we learned about later must
not decide who appeared in a 2019 candidate list.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.models.enums import GameStatus
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats
from euroleague_ai.models.people import PlayerTeamStint

#: A rostered player must have appeared in at least one of the team's last N
#: games to be predicted. Filters out the registered-but-never-dressed tail.
RECENT_APPEARANCE_GAMES = 5


@dataclass(frozen=True, slots=True)
class RecentAppearances:
    """How often each player featured in the team's last few games this season.

    The candidate filter only needs "appeared at least once". The counts are
    kept anyway because they are also the natural *baseline* for a
    play-probability model: a player who played four of the last five has
    played 80% of the time recently.
    """

    #: Team games in the window. At most :data:`RECENT_APPEARANCE_GAMES`, fewer
    #: early in a season.
    team_games: int
    #: Games each player actually took the floor in, within the window.
    played: dict[uuid.UUID, int] = field(default_factory=dict)
    #: Everyone who has taken the floor for this club at any point this season.
    debuted: frozenset[uuid.UUID] = frozenset()


async def current_club(
    session: AsyncSession, *, as_of: datetime
) -> dict[uuid.UUID, uuid.UUID]:
    """Map every player to the club they belonged to at ``as_of``.

    **This has to be transfer-aware.** Scoping roster lookups to one season is
    wrong across a transfer window: in pre-season a player can hold a
    registration for the season just finished *and* one for the season about
    to start, at different clubs. Filtering by season alone returns both, and
    the player gets projected for a club he left.

    The rule: among the registrations in effect at ``as_of``, the one with the
    latest ``effective_from`` wins. A July move to Real Madrid supersedes a
    registration at Partizan that nominally runs to 30 June, because it started
    later.

    Resolved in Python rather than SQL because the working set is a few thousand
    stints and the tie-breaking is far easier to read, and to get right, as
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


async def recent_appearances(
    session: AsyncSession,
    *,
    team_id: uuid.UUID,
    season_id: uuid.UUID,
    as_of: datetime,
    games: int = RECENT_APPEARANCE_GAMES,
) -> RecentAppearances:
    """Appearance counts over the team's last ``games`` finished games this season.

    The season scope is load-bearing. Without it, "the team's last five games"
    reaches back into the previous season, and a squad that has since been
    rebuilt gets projected as though nobody moved.
    """
    as_of = ensure_utc(as_of, field="as_of")

    recent_ids = list(
        (
            await session.execute(
                select(Game.id)
                .join(GameTeam, GameTeam.game_id == Game.id)
                .where(
                    GameTeam.team_id == team_id,
                    # Same season only. Cross-season appearance history
                    # describes a squad that no longer exists.
                    Game.season_id == season_id,
                    Game.observed_at <= as_of,
                    Game.tip_off_utc < as_of,
                    Game.status == GameStatus.FINISHED,
                )
                .order_by(Game.tip_off_utc.desc())
                .limit(games)
            )
        )
        .scalars()
        .all()
    )

    played: dict[uuid.UUID, int] = {}
    if recent_ids:
        played = {
            player_id: int(count)
            for player_id, count in (
                await session.execute(
                    select(
                        PlayerGameStats.player_id,
                        func.count(func.distinct(GameTeam.game_id)),
                    )
                    .join(GameTeam, GameTeam.id == PlayerGameStats.game_team_id)
                    .where(
                        GameTeam.team_id == team_id,
                        GameTeam.game_id.in_(recent_ids),
                        PlayerGameStats.observed_at <= as_of,
                        PlayerGameStats.did_not_play.is_(False),
                    )
                    .group_by(PlayerGameStats.player_id)
                )
            ).all()
        }

    # Everyone who has featured for this club at any point this season. Used to
    # tell "fringe player who keeps being left out" apart from "signing who has
    # not debuted yet". The appearance filter should drop the former, never the
    # latter.
    debuted = frozenset(
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

    return RecentAppearances(team_games=len(recent_ids), played=played, debuted=debuted)


def select_candidates(
    rostered: set[uuid.UUID], appearances: RecentAppearances
) -> list[uuid.UUID]:
    """The pure part of :func:`candidate_players`, separated so it can be unit-tested.

    Keeps club-correct players who played recently, plus club-correct players
    who have not yet appeared for this club this season (new signings, players
    back from long injuries). The model projects low minutes for the uncertain
    ones, which is the honest answer. Silently omitting a summer signing is not.
    """
    appeared = {pid for pid, count in appearances.played.items() if count > 0}
    if not appeared:
        # No games played this season yet. The roster is all we have, and it is
        # already club-correct.
        return sorted(rostered)
    return sorted((rostered & appeared) | (rostered - appearances.debuted))


async def candidate_players(
    session: AsyncSession,
    *,
    team_id: uuid.UUID,
    season_id: uuid.UUID,
    as_of: datetime,
    club_by_player: dict[uuid.UUID, uuid.UUID] | None = None,
    appearances: RecentAppearances | None = None,
) -> list[uuid.UUID]:
    """Players plausibly available for this team at ``as_of``.

    Two filters, in this order:

    1. **Club membership**, resolved by :func:`current_club`. A player whose
       latest registration is at another club is excluded outright, whatever
       the appearance history says.
    2. **Recent appearances in this season**, to drop the registered-but-never-
       dressed tail. Skipped before a season's first games, when there is
       nothing to filter on.

    Args:
        club_by_player: Precomputed :func:`current_club` result. Pass it when
            resolving both sides of a fixture, so they share one view.
        appearances: Precomputed :func:`recent_appearances` result, for callers
            that also need the counts.
    """
    as_of = ensure_utc(as_of, field="as_of")

    if club_by_player is None:
        club_by_player = await current_club(session, as_of=as_of)

    rostered = {
        player_id for player_id, club_id in club_by_player.items() if club_id == team_id
    }
    if not rostered:
        return []

    if appearances is None:
        appearances = await recent_appearances(
            session, team_id=team_id, season_id=season_id, as_of=as_of
        )
    return select_candidates(rostered, appearances)

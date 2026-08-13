"""Teammate availability context, derived from prior appearances.

**Why this module exists in this shape.**

The obvious design would read ``player_status_events`` and count how many
teammates were listed OUT before tip-off. That table is empty for every
historical season, and permanently so: nobody was running this system in 2018 to
observe that a player was ruled out at 10:00 that morning. Bitemporal data cannot
be reconstructed after the fact -- if a fact was never observed, there is no
honest way to say what was knowable at a past instant.

So the historical signal has to come from something that *was* recorded. Who
actually took the floor in the team's previous games is exactly that, and it is
leak-free: the previous game finished before this one tipped off, so the
information was genuinely available at prediction time.

What this buys, concretely: if a team's leading scorer played the last four games
and then missed the fifth, that absence is visible before game six, and the usage
it frees up is the single largest driver of a teammate's stat line.

When real availability data does arrive (phase 5), these features stay -- they
become a fallback for players and seasons the news pipeline does not cover.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.repositories.history import SquadAppearance, TeamHistoryRepository

#: How many prior team games define "the recent squad". Five is long enough to
#: establish who the rotation is, short enough to react to a change.
SQUAD_WINDOW = 5

#: A player is considered part of the rotation if they averaged at least this
#: many minutes across the games they appeared in. Below it, absence carries no
#: information -- deep bench players are in and out of the squad constantly.
ROTATION_MINUTES_THRESHOLD = 10.0


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def summarise_squad(
    appearances: list[SquadAppearance], *, player_id: uuid.UUID
) -> dict[str, float | None]:
    """Derive teammate-availability features from prior appearances.

    Args:
        appearances: Rows from
            :meth:`~euroleague_ai.repositories.history.TeamHistoryRepository.squad_appearances`,
            newest game first.
        player_id: The player being predicted, excluded from teammate aggregates.

    Returns:
        Feature name to value, ``None`` where there is no history.
    """
    if not appearances:
        return {
            "squad_games_observed": 0.0,
            "rotation_size_last_game": None,
            "rotation_size_avg": None,
            "teammates_absent_last_game": None,
            "teammate_minutes_absent": None,
            "teammate_minutes_absent_share": None,
            "top_teammate_absent": None,
            "player_played_last_game": None,
            "player_missed_recent_games": None,
        }

    # Group by game, newest first.
    by_game: dict[uuid.UUID, list[SquadAppearance]] = defaultdict(list)
    order: list[uuid.UUID] = []
    for appearance in appearances:
        if appearance.game_id not in by_game:
            order.append(appearance.game_id)
        by_game[appearance.game_id].append(appearance)

    order = order[:SQUAD_WINDOW]

    # Per-player minutes across the window, used both to define the rotation and
    # to weight an absence. A missing 30-minute starter matters enormously more
    # than a missing 4-minute reserve, so a raw count of absentees is a poor
    # feature -- weighting by minutes is the point.
    minutes_by_player: dict[uuid.UUID, list[float]] = defaultdict(list)
    for game_id in order:
        for appearance in by_game[game_id]:
            if appearance.played:
                minutes_by_player[appearance.player_id].append(appearance.minutes)

    avg_minutes = {
        pid: sum(values) / len(values) for pid, values in minutes_by_player.items() if values
    }
    rotation = {
        pid: minutes
        for pid, minutes in avg_minutes.items()
        if minutes >= ROTATION_MINUTES_THRESHOLD and pid != player_id
    }

    last_game_id = order[0]
    last_game = by_game[last_game_id]
    played_last = {a.player_id for a in last_game if a.played}

    rotation_sizes = [
        sum(1 for a in by_game[game_id] if a.played) for game_id in order
    ]

    absent = {pid: minutes for pid, minutes in rotation.items() if pid not in played_last}
    top_teammate = max(rotation, key=lambda pid: rotation[pid], default=None)

    player_appearances = [
        a for game_id in order for a in by_game[game_id] if a.player_id == player_id
    ]
    player_played_last = any(a.played for a in last_game if a.player_id == player_id)

    return {
        "squad_games_observed": float(len(order)),
        "rotation_size_last_game": float(rotation_sizes[0]),
        "rotation_size_avg": _round(sum(rotation_sizes) / len(rotation_sizes)),
        "teammates_absent_last_game": float(len(absent)),
        # The headline feature: how much usual playing time is unaccounted for.
        # This is the quantity that actually redistributes to everyone else.
        "teammate_minutes_absent": _round(sum(absent.values())),
        "teammate_minutes_absent_share": _round(
            sum(absent.values()) / sum(rotation.values()) if rotation else None
        ),
        "top_teammate_absent": (
            None if top_teammate is None else float(top_teammate in absent)
        ),
        # The player's own recent participation. A player returning from an
        # absence is usually eased back in, which suppresses minutes.
        "player_played_last_game": float(player_played_last),
        "player_missed_recent_games": float(
            sum(1 for a in player_appearances if not a.played)
        ),
    }


async def build_availability_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
) -> dict[str, float | None]:
    """Teammate availability context as of ``as_of``.

    Leak-free by construction: every input comes from games that finished before
    the cutoff, read through the ``observed_at <= as_of`` filter.
    """
    as_of = ensure_utc(as_of, field="as_of")

    appearances = await TeamHistoryRepository(session).squad_appearances(
        team_id, as_of=as_of, games=SQUAD_WINDOW, season_id=season_id
    )
    return summarise_squad(appearances, player_id=player_id)

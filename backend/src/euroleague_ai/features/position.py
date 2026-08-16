"""Position and positional competition.

**The observation this exists to encode.** Five guards cannot all play twenty
minutes. A team has exactly two hundred player-minutes in regulation, and a
coach does not spread them evenly across whoever happens to be registered --
he plays the best guard heavily, the second one moderately, and the fourth one
barely at all. Predicting each player in isolation, as the model does, has no
way to know that. Two players with identical recent form get identical
projections even when one is behind three better players at his position and
the other is behind nobody.

**A caveat about granularity, stated plainly.** The EuroLeague feed publishes
three positions -- Guard, Forward, Centre -- not the five of PG/SG/SF/PF/C.
There is no way to distinguish a point guard from a shooting guard from this
source, so nothing here pretends to. Three buckets still carry most of the
signal: guards assist and shoot threes, centres rebound, and the competition
effect operates at roughly this granularity anyway in a sport that has spent a
decade becoming less positional.

**Where the real work is.** The one-hot flags are the obvious part and probably
the least useful, since the model can already infer a centre from his rebound
rate. The features that carry the observation are the competition ones:
``same_position_rank`` -- where this player sits in the pecking order at his
position, by recent minutes -- and ``same_position_minutes_share``, his slice of
the minutes his position group has been getting. Being the fourth-ranked guard
is a fact about the squad, not about the player, and it is invisible in his own
box scores.

Positions are resolved through the stint table, which carries ``observed_at``,
so a position we only learned this season cannot leak into a 2018 projection.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import ensure_utc
from euroleague_ai.repositories.history import SquadAppearance, TeamHistoryRepository

#: Prior team games the competition window looks back over. Ten is long enough
#: to describe a rotation and short enough to notice one changing.
COMPETITION_WINDOW = 10

#: The positions the upstream feed actually distinguishes.
POSITIONS: tuple[str, ...] = ("GUARD", "FORWARD", "CENTER")


def _minutes_by_player(appearances: list[SquadAppearance]) -> dict[uuid.UUID, float]:
    """Total minutes per player across the window.

    Non-appearances count as zero rather than being dropped: a player who was
    available and did not play is genuinely behind the others, and dropping
    those rows would flatter him.
    """
    totals: dict[uuid.UUID, float] = {}
    for appearance in appearances:
        totals[appearance.player_id] = totals.get(appearance.player_id, 0.0) + (
            0.0 if appearance.did_not_play else appearance.minutes
        )
    return totals


def summarise_position(
    *,
    player_id: uuid.UUID,
    positions: dict[uuid.UUID, str],
    appearances: list[SquadAppearance],
) -> dict[str, float | None]:
    """Position flags plus the player's standing within his position group."""
    own = positions.get(player_id, "UNKNOWN")

    features: dict[str, float | None] = {
        f"position_{name.lower()}": 1.0 if own == name else 0.0 for name in POSITIONS
    }
    # Kept as an explicit flag rather than left implicit in three zeros. "We do
    # not know this player's position" is different from "he is not a guard",
    # and a model that cannot tell them apart will treat every unmapped player
    # as a non-guard non-forward non-centre.
    features["position_unknown"] = 1.0 if own not in POSITIONS else 0.0

    minutes = _minutes_by_player(appearances)

    # The squad as registered, not merely those who have appeared: a fit player
    # who has not featured is still competition for minutes.
    group = [pid for pid, position in positions.items() if position == own]
    features["same_position_count"] = float(len(group))

    if own not in POSITIONS or not group:
        # No usable position means no usable competition signal. NaN rather than
        # a fabricated zero, which would read as "no competition at all".
        features["same_position_rank"] = None
        features["same_position_minutes_share"] = None
        features["same_position_minutes_total"] = None
        features["same_position_rank_pct"] = None
        return features

    group_minutes = {pid: minutes.get(pid, 0.0) for pid in group}
    own_minutes = group_minutes.get(player_id, 0.0)
    total = sum(group_minutes.values())

    # Rank by minutes, most first. 1.0 means the leading player at this
    # position; being fourth is the fact that stops four guards all projecting
    # twenty minutes.
    ordered = sorted(group_minutes.values(), reverse=True)
    rank = ordered.index(own_minutes) + 1 if own_minutes in ordered else len(ordered)

    features["same_position_rank"] = float(rank)
    # Normalised too, because a rank of three means something different in a
    # group of three than in a group of six.
    features["same_position_rank_pct"] = round(rank / len(group), 4)
    features["same_position_minutes_total"] = round(total, 2)
    features["same_position_minutes_share"] = (
        round(own_minutes / total, 4) if total > 0 else None
    )
    return features


async def build_position_features(
    session: AsyncSession,
    *,
    player_id: uuid.UUID,
    team_id: uuid.UUID,
    as_of: datetime,
    season_id: uuid.UUID | None = None,
) -> dict[str, float | None]:
    """Position and positional competition as of ``as_of``.

    Both inputs are read through the history repository, which applies the
    ``observed_at <= as_of`` filter, so nothing here can see a registration or
    an appearance we had not yet recorded.
    """
    as_of = ensure_utc(as_of, field="as_of")
    teams = TeamHistoryRepository(session)

    positions = await teams.squad_positions(team_id, as_of=as_of, season_id=season_id)
    appearances = await teams.squad_appearances(
        team_id, as_of=as_of, games=COMPETITION_WINDOW, season_id=season_id
    )
    return summarise_position(
        player_id=player_id, positions=positions, appearances=appearances
    )

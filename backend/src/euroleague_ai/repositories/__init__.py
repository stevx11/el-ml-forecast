"""Data access. All SQL lives here.

Split by temporal semantics rather than by table:

* :mod:`~euroleague_ai.repositories.catalog` -- current-state reads for the API.
  No ``as_of``; these serve "what does the site show now?".
* :mod:`~euroleague_ai.repositories.history` -- time-filtered reads for features.
  ``as_of`` is mandatory. **This is the leakage choke point.**

Keeping them apart means an API convenience query can never be reused
accidentally inside a feature builder.
"""

from euroleague_ai.repositories.catalog import (
    GameRepository,
    LeagueRepository,
    PlayerRepository,
    TeamRepository,
)
from euroleague_ai.repositories.history import (
    PlayerGameLine,
    PlayerHistoryRepository,
    TeamGameLine,
    TeamHistoryRepository,
)

__all__ = [
    "GameRepository",
    "LeagueRepository",
    "PlayerGameLine",
    "PlayerHistoryRepository",
    "PlayerRepository",
    "TeamGameLine",
    "TeamHistoryRepository",
    "TeamRepository",
]

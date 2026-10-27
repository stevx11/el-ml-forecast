"""Offline provider backed by committed JSON fixtures.

Satisfies the same protocol as the live adapter, so the entire ingestion
pipeline -- identity resolution, temporal upserts, feature building -- is
exercised end to end in tests with no network access and no flakiness.

The fixture files mirror the exact key names and value types of the live API,
including its quirks (``foulsCommited``, space-padded person codes, floats for
counting stats). ``scripts/capture_fixtures.py`` regenerates them from the live
API when the upstream shape changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from euroleague_ai.core.errors import ProviderError
from euroleague_ai.providers.base import (
    GameDTO,
    GameStatsDTO,
    PlayerStintDTO,
    SeasonRef,
    TeamDTO,
)
from euroleague_ai.providers.mapping import (
    map_game,
    map_game_stats,
    map_player_stint,
    map_team,
)

PROVIDER_KEY = "fixture"

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "euroleague"


class FixtureProvider:
    """Reads provider payloads from disk.

    Layout under ``root``::

        clubs.json                  {"data": [...]}
        people.json                 {"data": [...]}
        games.json                  {"data": [...]}
        game_stats/{game_code}.json {"local": {...}, "road": {...}}
    """

    provider_key = PROVIDER_KEY

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else DEFAULT_FIXTURE_DIR
        if not self._root.is_dir():
            raise ProviderError(self.provider_key, f"fixture directory not found: {self._root}")

    def _load(self, *parts: str) -> Any:
        path = self._root.joinpath(*parts)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProviderError(self.provider_key, f"invalid JSON in {path}") from exc

    @staticmethod
    def _rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            payload = payload.get("data")
        if not isinstance(payload, list):
            return []
        return [row for row in payload if isinstance(row, dict)]

    async def fetch_teams(self, season: SeasonRef) -> list[TeamDTO]:
        rows = self._rows(self._load("clubs.json"))
        return [team for row in rows if (team := map_team(row)) is not None]

    async def fetch_players(self, season: SeasonRef) -> list[PlayerStintDTO]:
        rows = self._rows(self._load("people.json"))
        return [stint for row in rows if (stint := map_player_stint(row)) is not None]

    async def fetch_games(
        self, season: SeasonRef, *, round_number: int | None = None
    ) -> list[GameDTO]:
        rows = self._rows(self._load("games.json"))
        games = [game for row in rows if (game := map_game(row)) is not None]
        if round_number is not None:
            games = [game for game in games if game.round_number == round_number]
        return games

    async def fetch_game_stats(self, season: SeasonRef, game_code: int) -> GameStatsDTO | None:
        payload = self._load("game_stats", f"{game_code}.json")
        if not isinstance(payload, dict):
            return None
        return map_game_stats(game_code, payload)

    async def aclose(self) -> None:
        """No resources to release."""
        return None

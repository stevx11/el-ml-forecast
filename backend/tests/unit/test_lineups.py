"""The lineup probe: reading starters off a game sheet, and reading it fresh."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from euroleague_ai.ingestion.lineups import (
    append_log,
    first_complete,
    read_lineup,
    side_lineup,
)
from euroleague_ai.providers.euroleague import EuroLeagueProvider, ResponseCache
from euroleague_ai.providers.factory import season_ref

TIP_OFF = datetime(2026, 10, 30, 18, 0, tzinfo=UTC)


def _side(team: str, players: int, starters: int) -> dict[str, Any]:
    return {
        "players": [
            {
                "player": {"club": {"code": team}, "person": {"name": f"{team} {i}"}},
                "stats": {"startFive": i < starters, "timePlayed": None},
            }
            for i in range(players)
        ]
    }


def _sheet(home_starters: int, away_starters: int) -> dict[str, Any]:
    return {"local": _side("PAN", 12, home_starters), "road": _side("OLY", 11, away_starters)}


class TestReading:
    def test_counts_listed_players_and_starters(self) -> None:
        side = side_lineup(_side("PAN", 12, 5))
        assert (side.team, side.players, side.starters) == ("PAN", 12, 5)
        assert side.starter_names == tuple(f"PAN {i}" for i in range(5))

    def test_missing_side_is_empty(self) -> None:
        assert side_lineup(None).players == 0

    def test_a_404_is_no_sheet(self) -> None:
        record = read_lineup(61, TIP_OFF, None, now=TIP_OFF - timedelta(minutes=30))
        assert not record.found
        assert not record.complete
        assert record.minutes_to_tip_off == 30.0

    def test_complete_needs_both_fives(self) -> None:
        now = TIP_OFF - timedelta(minutes=12)
        assert read_lineup(61, TIP_OFF, _sheet(5, 5), now=now).complete
        assert not read_lineup(61, TIP_OFF, _sheet(5, 0), now=now).complete
        assert not read_lineup(61, TIP_OFF, _sheet(0, 0), now=now).complete


class TestSummary:
    def test_first_time_both_fives_show(self) -> None:
        looks = [
            read_lineup(61, TIP_OFF, None, now=TIP_OFF - timedelta(minutes=40)),
            read_lineup(61, TIP_OFF, _sheet(0, 0), now=TIP_OFF - timedelta(minutes=20)),
            read_lineup(61, TIP_OFF, _sheet(5, 5), now=TIP_OFF - timedelta(minutes=8)),
            read_lineup(61, TIP_OFF, _sheet(5, 5), now=TIP_OFF - timedelta(minutes=2)),
            read_lineup(62, TIP_OFF, None, now=TIP_OFF - timedelta(minutes=40)),
            read_lineup(62, TIP_OFF, _sheet(5, 5), now=TIP_OFF + timedelta(minutes=1)),
        ]
        assert first_complete(looks) == {61: (40.0, 8.0), 62: (40.0, -1.0)}

    def test_never_complete(self) -> None:
        looks = [read_lineup(61, TIP_OFF, None, now=TIP_OFF - timedelta(minutes=5))]
        assert first_complete(looks) == {61: (5.0, None)}

    def test_log_appends_one_line_per_look(self, tmp_path: Path) -> None:
        now = TIP_OFF - timedelta(minutes=10)
        record = read_lineup(61, TIP_OFF, _sheet(5, 5), now=now)
        append_log([record], root=tmp_path, now=now)
        path = append_log([record], root=tmp_path, now=now)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["complete"] is True


class _MemoryRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


class TestFreshFetch:
    async def test_the_probe_never_reads_or_writes_the_cache(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            # The sheet fills in between the two calls.
            return httpx.Response(200, json=_sheet(5, 5) if len(calls) > 1 else _sheet(0, 0))

        redis = _MemoryRedis()
        client = httpx.AsyncClient(
            base_url="https://example.test/v2", transport=httpx.MockTransport(handler)
        )
        provider = EuroLeagueProvider(client=client, cache=ResponseCache(redis, ttl_seconds=3600))
        season = season_ref("E2025")
        try:
            first = await provider.fetch_game_stats_payload(season, 61)
            second = await provider.fetch_game_stats_payload(season, 61)
        finally:
            await client.aclose()

        assert len(calls) == 2
        assert redis.store == {}
        assert first is not None and second is not None
        assert side_lineup(first["local"]).starters == 0
        assert side_lineup(second["local"]).starters == 5

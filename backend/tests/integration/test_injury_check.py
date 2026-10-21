"""The round check: statuses the predictions used, against the box score.

Game 6 (Nov 7): Jovanovic is missing from the box score, everyone else played.
Before the prediction, the table rules Jovanovic and Petrovic out and calls
Hakansson ready, and a vague news item has Nikolic out. After the prediction,
a late report calls Dubois questionable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from euroleague_ai.ml.injury_check import check_games
from euroleague_ai.models.availability import PlayerStatusEvent
from euroleague_ai.models.enums import AvailabilityStatus, EntityType, StatusSourceType
from tests.conftest import requires_db
from tests.integration.test_availability import _game, _id, _seed_absences
from tests.integration.test_expectation import _predict_game6

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from euroleague_ai.core.config import Settings
    from euroleague_ai.providers.base import SeasonRef
    from euroleague_ai.providers.fixture import FixtureProvider

pytestmark = [pytest.mark.integration, requires_db]

BEFORE_GAME6 = datetime(2025, 11, 6, 9, 0, tzinfo=UTC)


async def _report(
    session: AsyncSession,
    code: str,
    team: str,
    status: AvailabilityStatus,
    *,
    source: StatusSourceType = StatusSourceType.REPORT,
    confidence: float | None = None,
) -> None:
    game = await _game(session, 6)
    session.add(
        PlayerStatusEvent(
            player_id=await _id(session, EntityType.PLAYER, code),
            team_id=await _id(session, EntityType.TEAM, team),
            game_id=game.id if source is StatusSourceType.REPORT else None,
            status=status,
            reason=f"{status.lower()} for game 6",
            source_type=source,
            confidence=confidence,
            event_timestamp=BEFORE_GAME6,
            observed_at=BEFORE_GAME6,
        )
    )
    await session.flush()


class TestRoundCheck:
    async def test_groups_and_mismatches(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await _report(session, "011002", "PAR", AvailabilityStatus.OUT)  # Jovanovic: sat
        await _report(session, "011001", "PAR", AvailabilityStatus.OUT)  # Petrovic: played
        await _report(session, "013001", "BAS", AvailabilityStatus.AVAILABLE)  # Hakansson
        await _report(
            session,
            "011003",
            "PAR",
            AvailabilityStatus.OUT,
            source=StatusSourceType.NEWS,
            confidence=0.5,
        )  # Nikolic: too vague
        await _predict_game6(session, settings)
        await _report(session, "013002", "BAS", AvailabilityStatus.QUESTIONABLE)  # late

        result = await check_games(session, [await _game(session, 6)])
        by_player = {p.player: p for p in result.players}
        assert {name: p.group for name, p in by_player.items()} == {
            "JOVANOVIC, LUKA": "ruled out",
            "PETROVIC, MARKO": "ruled out",
            "NIKOLIC, STEFAN": "out (news, not applied)",
            "HAKANSSON, ERIK": "ready",
            "DUBOIS, MATHIEU": "late report",
        }
        jovanovic = by_player["JOVANOVIC, LUKA"]
        assert not jovanovic.played and not jovanovic.dressed and jovanovic.plays == 0.0
        assert [p.player for p in result.mismatches] == ["PETROVIC, MARKO"]

        summary = result.summary()
        assert summary["ruled out"] == {
            "players": 2,
            "played": 1,
            "play_rate": 0.5,
            "mean_p_plays": 0.0,
            "mismatches": 1,
        }
        assert summary["ready"]["mismatches"] == 0
        assert summary["out (news, not applied)"]["mean_p_plays"] == pytest.approx(0.9)
        assert list(summary) == [
            "ruled out",
            "out (news, not applied)",
            "ready",
            "late report",
        ]

    async def test_players_without_a_status_are_left_out(
        self,
        session: AsyncSession,
        fixture_provider: FixtureProvider,
        season: SeasonRef,
        settings: Settings,
    ) -> None:
        await _seed_absences(session, fixture_provider, season)
        await _predict_game6(session, settings)
        result = await check_games(session, [await _game(session, 6)])
        assert result.games == 1 and result.players == []

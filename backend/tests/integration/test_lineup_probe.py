"""Which games the lineup probe looks at: this season's, near tip-off."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import FixedClock
from euroleague_ai.ingestion.jobs import ingest_all
from euroleague_ai.ingestion.lineups import AFTER_TIP_OFF, games_near_tip_off
from euroleague_ai.models.game import Game
from euroleague_ai.providers.base import SeasonRef
from euroleague_ai.providers.fixture import FixtureProvider
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

CLOCK = FixedClock(datetime(2025, 9, 1, 12, 0, tzinfo=UTC))


async def test_games_in_the_window_soonest_first(
    session: AsyncSession, fixture_provider: FixtureProvider, season: SeasonRef
) -> None:
    await ingest_all(session, fixture_provider, season, clock=CLOCK)
    games = list((await session.execute(select(Game).order_by(Game.tip_off_utc))).scalars())
    assert games
    tip_off = games[0].tip_off_utc

    found = await games_near_tip_off(
        session, season_code=season.season_code, now=tip_off - timedelta(hours=1), hours=2
    )
    assert games[0].game_code in [g.game_code for g in found]
    assert [g.tip_off_utc for g in found] == sorted(g.tip_off_utc for g in found)
    assert all(
        tip_off - timedelta(hours=1) - AFTER_TIP_OFF
        <= g.tip_off_utc
        <= tip_off + timedelta(hours=1)
        for g in found
    )

    # Shortly after tip-off the game is still watched; long after, it is not.
    later = await games_near_tip_off(
        session, season_code=season.season_code, now=tip_off + timedelta(minutes=5), hours=0
    )
    assert games[0].game_code in [g.game_code for g in later]
    gone = await games_near_tip_off(
        session, season_code=season.season_code, now=tip_off + timedelta(hours=1), hours=0
    )
    assert games[0].game_code not in [g.game_code for g in gone]

    other = await games_near_tip_off(
        session, season_code="E1999", now=tip_off - timedelta(hours=1), hours=2
    )
    assert other == []

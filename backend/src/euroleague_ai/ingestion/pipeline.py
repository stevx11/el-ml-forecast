"""Provider DTOs into the time-aware store.

Three write disciplines are used here, and which one applies depends entirely on
the temporal semantics of the fact:

* **Immutable reference data** (leagues, seasons, teams, players) -- upsert.
  Correcting a spelling loses nothing.
* **Interval state** (roster stints) -- close the old interval, open a new one.
  Overwriting would destroy the answer to "who was on this roster in November?".
* **Observations** (box scores, status events) -- append, stamped with
  ``observed_at``.

Every write records ``observed_at`` from an injected clock rather than
``now()``, so an ingestion run has one coherent observation time and tests can
pin it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from euroleague_ai.core.clock import Clock, SystemClock
from euroleague_ai.core.logging import get_logger
from euroleague_ai.models.competition import League, Season, Team, TeamSeason
from euroleague_ai.models.enums import EntityType, GameStatus, TeamSide
from euroleague_ai.models.game import Game, GameTeam, PlayerGameStats, TeamGameStats
from euroleague_ai.models.people import Coach, Player, PlayerTeamStint
from euroleague_ai.providers.base import (
    GameDTO,
    GameStatsDTO,
    PlayerStintDTO,
    TeamDTO,
    TeamLineDTO,
    TeamStatsDTO,
)
from euroleague_ai.providers.parsing import estimate_pace, estimate_possessions
from euroleague_ai.ingestion.identity import IdentityResolver

logger = get_logger(__name__)

DEFAULT_LEAGUE_CODE = "EUROLEAGUE"
DEFAULT_LEAGUE_NAME = "EuroLeague"


@dataclass(slots=True)
class IngestionResult:
    """Row counts for one ingestion job."""

    rows_read: int = 0
    rows_written: int = 0
    skipped: int = 0
    details: dict[str, int] = field(default_factory=dict)

    def merge(self, other: IngestionResult) -> None:
        self.rows_read += other.rows_read
        self.rows_written += other.rows_written
        self.skipped += other.skipped
        for key, value in other.details.items():
            self.details[key] = self.details.get(key, 0) + value


class IngestionPipeline:
    """Writes provider DTOs into the domain store."""

    def __init__(
        self,
        session: AsyncSession,
        resolver: IdentityResolver,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._session = session
        self._resolver = resolver
        self._clock = clock or SystemClock()

    # -- competition scaffolding ------------------------------------------

    async def ensure_league(
        self, code: str = DEFAULT_LEAGUE_CODE, name: str = DEFAULT_LEAGUE_NAME
    ) -> League:
        """Return the league row, creating it if absent."""
        existing = (
            await self._session.execute(select(League).where(League.code == code))
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        league = League(id=uuid.uuid4(), code=code, name=name, tier=1)
        self._session.add(league)
        await self._session.flush()
        return league

    async def ensure_season(self, league: League, code: str, start_year: int) -> Season:
        """Return the season row, creating it if absent."""
        existing = (
            await self._session.execute(
                select(Season).where(Season.league_id == league.id, Season.code == code)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        season = Season(
            id=uuid.uuid4(),
            league_id=league.id,
            code=code,
            name=f"{league.name} {start_year}-{str(start_year + 1)[-2:]}",
            start_year=start_year,
        )
        self._session.add(season)
        await self._session.flush()
        return season

    # -- teams -------------------------------------------------------------

    async def upsert_teams(
        self, teams: list[TeamDTO], season: Season, raw_names: dict[str, str] | None = None
    ) -> IngestionResult:
        """Upsert clubs and register them for the season.

        Clubs are stable identity; the sponsored, season-specific name goes on
        :class:`TeamSeason` so a rebrand never orphans historical statistics.
        """
        result = IngestionResult(rows_read=len(teams))
        await self._resolver.preload(EntityType.TEAM)

        for dto in teams:
            team_id, _ = await self._resolver.resolve(EntityType.TEAM, dto.external_id)

            stmt = (
                pg_insert(Team)
                .values(
                    id=team_id,
                    code=dto.code,
                    name=dto.name,
                    short_name=dto.short_name,
                    country_code=dto.country_code,
                    city=dto.city,
                    venue_name=dto.venue_name,
                )
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "name": dto.name,
                        "short_name": dto.short_name,
                        "country_code": dto.country_code,
                        "city": dto.city,
                    },
                )
            )
            await self._session.execute(stmt)

            display_name = (raw_names or {}).get(dto.external_id, dto.name)
            await self._session.execute(
                pg_insert(TeamSeason)
                .values(
                    id=uuid.uuid4(),
                    team_id=team_id,
                    season_id=season.id,
                    display_name=display_name,
                )
                .on_conflict_do_update(
                    index_elements=["team_id", "season_id"],
                    set_={"display_name": display_name},
                )
            )
            result.rows_written += 1

        await self._session.flush()
        return result

    # -- players -----------------------------------------------------------

    async def upsert_player_stints(
        self, stints: list[PlayerStintDTO], season: Season
    ) -> IngestionResult:
        """Upsert players and their roster intervals."""
        result = IngestionResult(rows_read=len(stints))
        await self._resolver.preload(EntityType.PLAYER)
        await self._resolver.preload(EntityType.TEAM)
        observed_at = self._clock.now()

        for dto in stints:
            player_id, _ = await self._resolver.resolve(EntityType.PLAYER, dto.external_id)

            await self._session.execute(
                pg_insert(Player)
                .values(
                    id=player_id,
                    full_name=dto.full_name,
                    last_name=dto.last_name,
                    first_name=dto.first_name,
                    birth_date=dto.birth_date,
                    country_code=dto.country_code,
                    birth_country_code=dto.birth_country_code,
                    height_cm=dto.height_cm,
                    weight_kg=dto.weight_kg,
                    position=dto.position,
                )
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "full_name": dto.full_name,
                        "last_name": dto.last_name,
                        "first_name": dto.first_name,
                        "height_cm": dto.height_cm,
                        "weight_kg": dto.weight_kg,
                        "position": dto.position,
                    },
                )
            )

            if dto.team_external_id is None:
                result.skipped += 1
                continue

            team_id = await self._resolver.lookup(EntityType.TEAM, dto.team_external_id)
            if team_id is None:
                # A player registered to a club we have not ingested yet. Skipping
                # is correct: inventing a team row would corrupt every team-level
                # aggregate that later joins onto it.
                logger.warning(
                    "player references unknown team",
                    extra={"player": dto.external_id, "team": dto.team_external_id},
                )
                result.skipped += 1
                continue

            written = await self._upsert_stint(dto, player_id, team_id, season, observed_at)
            result.rows_written += written

        await self._session.flush()
        return result

    async def _upsert_stint(
        self,
        dto: PlayerStintDTO,
        player_id: uuid.UUID,
        team_id: uuid.UUID,
        season: Season,
        observed_at: datetime,
    ) -> int:
        """Insert a roster interval, or extend the matching open one.

        Re-running ingestion must not create a duplicate stint, and a transfer
        must not overwrite the previous club's interval. Matching on
        ``(player, team, season, effective_from)`` gives both: an unchanged
        registration updates in place, a genuinely new one inserts alongside.
        """
        existing = (
            await self._session.execute(
                select(PlayerTeamStint).where(
                    PlayerTeamStint.player_id == player_id,
                    PlayerTeamStint.team_id == team_id,
                    PlayerTeamStint.season_id == season.id,
                    PlayerTeamStint.effective_from == dto.effective_from,
                )
            )
        ).scalar_one_or_none()

        if existing is not None:
            existing.effective_until = dto.effective_until
            existing.jersey_number = dto.jersey_number
            existing.position = dto.position
            existing.is_active = dto.is_active
            existing.observed_at = observed_at
            return 0

        self._session.add(
            PlayerTeamStint(
                id=uuid.uuid4(),
                player_id=player_id,
                team_id=team_id,
                season_id=season.id,
                jersey_number=dto.jersey_number,
                position=dto.position,
                is_active=dto.is_active,
                previous_team_name=dto.previous_team_name,
                effective_from=dto.effective_from,
                effective_until=dto.effective_until,
                observed_at=observed_at,
            )
        )
        return 1

    # -- games -------------------------------------------------------------

    async def upsert_games(self, games: list[GameDTO], season: Season) -> IngestionResult:
        """Upsert fixtures and their two sides."""
        result = IngestionResult(rows_read=len(games))
        await self._resolver.preload(EntityType.TEAM)
        await self._resolver.preload(EntityType.GAME)
        observed_at = self._clock.now()

        for dto in games:
            home_id = await self._resolver.lookup(EntityType.TEAM, dto.home_team_external_id)
            away_id = await self._resolver.lookup(EntityType.TEAM, dto.away_team_external_id)
            if home_id is None or away_id is None:
                logger.warning(
                    "game references unknown team",
                    extra={
                        "game_code": dto.game_code,
                        "home": dto.home_team_external_id,
                        "away": dto.away_team_external_id,
                    },
                )
                result.skipped += 1
                continue

            game_key = f"{season.code}:{dto.game_code}"
            game_id, _ = await self._resolver.resolve(EntityType.GAME, game_key)

            await self._session.execute(
                pg_insert(Game)
                .values(
                    id=game_id,
                    season_id=season.id,
                    game_code=dto.game_code,
                    round_number=dto.round_number,
                    phase_code=dto.phase_code,
                    phase_name=dto.phase_name,
                    group_name=dto.group_name,
                    tip_off_utc=dto.tip_off_utc,
                    tip_off_local=dto.tip_off_local,
                    local_timezone=dto.local_timezone,
                    status=dto.status,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_score=dto.home_score,
                    away_score=dto.away_score,
                    is_neutral_venue=dto.is_neutral_venue,
                    venue_name=dto.venue_name,
                    attendance=dto.attendance,
                    observed_at=observed_at,
                )
                .on_conflict_do_update(
                    index_elements=["season_id", "game_code"],
                    set_={
                        # Rescheduling is normal in EuroLeague; a new observed_at
                        # marks when we learned of the change.
                        "tip_off_utc": dto.tip_off_utc,
                        "tip_off_local": dto.tip_off_local,
                        "local_timezone": dto.local_timezone,
                        "status": dto.status,
                        "home_score": dto.home_score,
                        "away_score": dto.away_score,
                        "venue_name": dto.venue_name,
                        "attendance": dto.attendance,
                        "observed_at": observed_at,
                    },
                )
            )

            for side, team_id, score, partials in (
                (TeamSide.HOME, home_id, dto.home_score, dto.home_quarter_scores),
                (TeamSide.AWAY, away_id, dto.away_score, dto.away_quarter_scores),
            ):
                await self._session.execute(
                    pg_insert(GameTeam)
                    .values(
                        id=uuid.uuid4(),
                        game_id=game_id,
                        team_id=team_id,
                        side=side,
                        score=score,
                        quarter_scores=partials,
                    )
                    .on_conflict_do_update(
                        index_elements=["game_id", "side"],
                        set_={"score": score, "quarter_scores": partials},
                    )
                )

            result.rows_written += 1

        await self._session.flush()
        return result

    # -- box scores --------------------------------------------------------

    async def upsert_game_stats(
        self, stats: GameStatsDTO, season: Season
    ) -> IngestionResult:
        """Write both sides of a completed game's box score."""
        result = IngestionResult(rows_read=1)

        game_key = f"{season.code}:{stats.game_code}"
        game_id = await self._resolver.lookup(EntityType.GAME, game_key)
        if game_id is None:
            logger.warning("box score for unknown game", extra={"game_code": stats.game_code})
            result.skipped += 1
            return result

        for side, team_stats in ((TeamSide.HOME, stats.home), (TeamSide.AWAY, stats.away)):
            written = await self._upsert_side_stats(game_id, side, team_stats)
            result.rows_written += written

        await self._session.flush()
        return result

    async def _upsert_side_stats(
        self, game_id: uuid.UUID, side: TeamSide, team_stats: TeamStatsDTO
    ) -> int:
        game_team = (
            await self._session.execute(
                select(GameTeam).where(GameTeam.game_id == game_id, GameTeam.side == side)
            )
        ).scalar_one_or_none()
        if game_team is None:
            logger.warning("no game_team row", extra={"game": str(game_id), "side": side})
            return 0

        observed_at = self._clock.now()

        if team_stats.coach_external_id and team_stats.coach_name:
            game_team.coach_id = await self._ensure_coach(
                team_stats.coach_external_id, team_stats.coach_name
            )

        written = 0
        for line in team_stats.players:
            player_id = await self._resolver.lookup(
                EntityType.PLAYER, line.player_external_id
            )
            if player_id is None:
                # A player appearing in a box score before the roster feed knows
                # about them -- common for mid-season signings. Register them now
                # rather than dropping a real performance.
                player_id, _ = await self._resolver.resolve(
                    EntityType.PLAYER, line.player_external_id
                )
                await self._session.execute(
                    pg_insert(Player)
                    .values(id=player_id, full_name=line.player_name)
                    .on_conflict_do_nothing(index_elements=["id"])
                )

            await self._session.execute(
                pg_insert(PlayerGameStats)
                .values(
                    id=uuid.uuid4(),
                    game_team_id=game_team.id,
                    player_id=player_id,
                    seconds_played=line.seconds_played,
                    is_starter=line.is_starter,
                    did_not_play=line.did_not_play,
                    points=line.points,
                    fg2m=line.fg2m,
                    fg2a=line.fg2a,
                    fg3m=line.fg3m,
                    fg3a=line.fg3a,
                    ftm=line.ftm,
                    fta=line.fta,
                    offensive_rebounds=line.offensive_rebounds,
                    defensive_rebounds=line.defensive_rebounds,
                    total_rebounds=line.total_rebounds,
                    assists=line.assists,
                    steals=line.steals,
                    turnovers=line.turnovers,
                    blocks_favour=line.blocks_favour,
                    blocks_against=line.blocks_against,
                    fouls_committed=line.fouls_committed,
                    fouls_received=line.fouls_received,
                    valuation=line.valuation,
                    plus_minus=line.plus_minus,
                    observed_at=observed_at,
                )
                .on_conflict_do_update(
                    index_elements=["game_team_id", "player_id"],
                    set_={
                        "seconds_played": line.seconds_played,
                        "is_starter": line.is_starter,
                        "did_not_play": line.did_not_play,
                        "points": line.points,
                        "fg2m": line.fg2m,
                        "fg2a": line.fg2a,
                        "fg3m": line.fg3m,
                        "fg3a": line.fg3a,
                        "ftm": line.ftm,
                        "fta": line.fta,
                        "offensive_rebounds": line.offensive_rebounds,
                        "defensive_rebounds": line.defensive_rebounds,
                        "total_rebounds": line.total_rebounds,
                        "assists": line.assists,
                        "steals": line.steals,
                        "turnovers": line.turnovers,
                        "blocks_favour": line.blocks_favour,
                        "blocks_against": line.blocks_against,
                        "fouls_committed": line.fouls_committed,
                        "fouls_received": line.fouls_received,
                        "valuation": line.valuation,
                        "plus_minus": line.plus_minus,
                        "observed_at": observed_at,
                    },
                )
            )
            written += 1

        await self._upsert_team_totals(game_team.id, team_stats.totals, observed_at)
        return written

    async def _upsert_team_totals(
        self, game_team_id: uuid.UUID, totals: TeamLineDTO, observed_at: datetime
    ) -> None:
        """Store team totals plus derived possessions and pace.

        Derived at write time so opponent-strength features can read them
        directly instead of re-aggregating box scores on every feature build.
        """
        fga = totals.fg2a + totals.fg3a
        possessions = estimate_possessions(
            fga=fga,
            offensive_rebounds=totals.offensive_rebounds,
            turnovers=totals.turnovers,
            fta=totals.fta,
        )
        values = {
            "points": totals.points,
            "fg2m": totals.fg2m,
            "fg2a": totals.fg2a,
            "fg3m": totals.fg3m,
            "fg3a": totals.fg3a,
            "ftm": totals.ftm,
            "fta": totals.fta,
            "offensive_rebounds": totals.offensive_rebounds,
            "defensive_rebounds": totals.defensive_rebounds,
            "total_rebounds": totals.total_rebounds,
            "assists": totals.assists,
            "steals": totals.steals,
            "turnovers": totals.turnovers,
            "blocks_favour": totals.blocks_favour,
            "blocks_against": totals.blocks_against,
            "fouls_committed": totals.fouls_committed,
            "fouls_received": totals.fouls_received,
            "valuation": totals.valuation,
            "possessions": possessions,
            "pace": estimate_pace(possessions),
            "observed_at": observed_at,
        }
        await self._session.execute(
            pg_insert(TeamGameStats)
            .values(id=uuid.uuid4(), game_team_id=game_team_id, **values)
            .on_conflict_do_update(index_elements=["game_team_id"], set_=values)
        )

    async def _ensure_coach(self, external_id: str, name: str) -> uuid.UUID:
        coach_id, _ = await self._resolver.resolve(EntityType.COACH, external_id)
        await self._session.execute(
            pg_insert(Coach)
            .values(id=coach_id, full_name=name)
            .on_conflict_do_update(index_elements=["id"], set_={"full_name": name})
        )
        return coach_id

    async def finished_game_codes(self, season: Season) -> list[int]:
        """Game codes for finished fixtures, oldest first.

        Drives box score ingestion. Only finished games are fetched: requesting
        stats for a scheduled game returns an empty payload and wastes a request.
        """
        stmt = (
            select(Game.game_code)
            .where(Game.season_id == season.id, Game.status == GameStatus.FINISHED)
            .order_by(Game.tip_off_utc)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def game_codes_missing_stats(self, season: Season) -> list[int]:
        """Finished games that have no box score yet, oldest first.

        This is what makes a backfill resumable. A season is ~400 requests
        against a rate-limited public API, so a run will sometimes stop partway;
        re-running should pick up where it left off rather than re-fetching
        everything and tripping the same rate limit again.
        """
        has_stats = (
            select(GameTeam.game_id)
            .join(PlayerGameStats, PlayerGameStats.game_team_id == GameTeam.id)
            .distinct()
        )
        stmt = (
            select(Game.game_code)
            .where(
                Game.season_id == season.id,
                Game.status == GameStatus.FINISHED,
                Game.id.notin_(has_stats),
            )
            .order_by(Game.tip_off_utc)
        )
        return list((await self._session.execute(stmt)).scalars().all())

"""Initial time-aware schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-12

Creates the full core schema described in docs/data-model.md: competition
structure, people with interval-valued affiliations, games and box scores,
availability history, news, feature snapshots, the model registry, and the
ingestion audit trail.

Enum types are created once up front and referenced with ``create_type=False``
thereafter. Letting each ``create_table`` emit its own ``CREATE TYPE`` fails on
the second table that uses the same enum.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# --------------------------------------------------------------------------
# Enum types
# --------------------------------------------------------------------------
_ENUMS: dict[str, tuple[str, ...]] = {
    "entitytype": ("LEAGUE", "SEASON", "TEAM", "PLAYER", "COACH", "GAME"),
    "gamestatus": ("SCHEDULED", "LIVE", "FINISHED", "POSTPONED", "CANCELLED"),
    "teamside": ("HOME", "AWAY"),
    "playerposition": ("GUARD", "FORWARD", "CENTER", "UNKNOWN"),
    "availabilitystatus": (
        "AVAILABLE",
        "PROBABLE",
        "QUESTIONABLE",
        "DOUBTFUL",
        "OUT",
        "SUSPENDED",
        "UNKNOWN",
    ),
    "statussourcetype": ("PROVIDER", "NEWS", "MANUAL"),
    "newseventtype": (
        "INJURY",
        "AVAILABILITY_CHANGE",
        "MINUTES_LIMIT",
        "ROLE_CHANGE",
        "STARTER_CHANGE",
        "TRANSFER",
        "COACH_COMMENT",
        "COACH_CHANGE",
        "ROTATION_CHANGE",
        "RETURN_FROM_INJURY",
        "OTHER",
    ),
    "predictiontarget": (
        "MINUTES",
        "POINTS",
        "REBOUNDS",
        "ASSISTS",
        "THREE_POINTERS_MADE",
        "STEALS",
        "BLOCKS",
        "TURNOVERS",
        "FIELD_GOALS_MADE",
        "FIELD_GOALS_ATTEMPTED",
        "THREE_POINTERS_ATTEMPTED",
        "FREE_THROWS_MADE",
        "FREE_THROWS_ATTEMPTED",
        "VALUATION",
    ),
    "ingestionstatus": ("RUNNING", "SUCCEEDED", "FAILED"),
}


def _enum(name: str) -> postgresql.ENUM:
    """Reference an already-created enum type."""
    return postgresql.ENUM(*_ENUMS[name], name=name, create_type=False)


def _uuid_pk() -> sa.Column:
    return sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False)


def _timestamps() -> list[sa.Column]:
    """Row bookkeeping. Audit only -- never read by features."""
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _observed_at() -> sa.Column:
    """When this application learned the fact. The leakage cutoff column."""
    return sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False)


def _counting_stats() -> list[sa.Column]:
    """The shared box score columns, used by player and team stat tables."""
    names = (
        "points",
        "fg2m",
        "fg2a",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "offensive_rebounds",
        "defensive_rebounds",
        "total_rebounds",
        "assists",
        "steals",
        "turnovers",
        "blocks_favour",
        "blocks_against",
        "fouls_committed",
        "fouls_received",
        "valuation",
    )
    return [sa.Column(name, sa.Integer(), nullable=False, server_default="0") for name in names]


def upgrade() -> None:
    bind = op.get_bind()
    for name, values in _ENUMS.items():
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)

    # ---------------------------------------------------------------- roots
    op.create_table(
        "leagues",
        _uuid_pk(),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("country_code", sa.String(3), nullable=True),
        sa.Column("tier", sa.Integer(), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("code", name="uq_leagues_code"),
    )

    op.create_table(
        "teams",
        _uuid_pk(),
        sa.Column("code", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("short_name", sa.String(64), nullable=True),
        sa.Column("country_code", sa.String(3), nullable=True),
        sa.Column("city", sa.String(128), nullable=True),
        sa.Column("venue_name", sa.String(160), nullable=True),
        sa.Column("venue_latitude", sa.Numeric(9, 6), nullable=True),
        sa.Column("venue_longitude", sa.Numeric(9, 6), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("code", name="uq_teams_code"),
    )

    op.create_table(
        "coaches",
        _uuid_pk(),
        sa.Column("full_name", sa.String(160), nullable=False),
        sa.Column("country_code", sa.String(3), nullable=True),
        *_timestamps(),
    )

    op.create_table(
        "players",
        _uuid_pk(),
        sa.Column("full_name", sa.String(160), nullable=False),
        sa.Column("last_name", sa.String(96), nullable=True),
        sa.Column("first_name", sa.String(96), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("country_code", sa.String(3), nullable=True),
        sa.Column("birth_country_code", sa.String(3), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("weight_kg", sa.Integer(), nullable=True),
        sa.Column("position", _enum("playerposition"), nullable=False, server_default="UNKNOWN"),
        *_timestamps(),
    )
    op.create_index("ix_players_last_name", "players", ["last_name"])

    op.create_table(
        "seasons",
        _uuid_pk(),
        sa.Column("league_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("start_year", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"], name="fk_seasons_league_id_leagues"),
        sa.UniqueConstraint("league_id", "code", name="uq_seasons_league_code"),
    )
    op.create_index("ix_seasons_league_id", "seasons", ["league_id"])

    op.create_table(
        "team_seasons",
        _uuid_pk(),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("season_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_name", sa.String(160), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], name="fk_team_seasons_team_id_teams"),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"], name="fk_team_seasons_season_id_seasons"
        ),
        sa.UniqueConstraint("team_id", "season_id", name="uq_team_seasons_team_season"),
    )
    op.create_index("ix_team_seasons_season", "team_seasons", ["season_id"])

    # ------------------------------------------------------------- identity
    op.create_table(
        "external_refs",
        _uuid_pk(),
        sa.Column("provider_key", sa.String(64), nullable=False),
        sa.Column("entity_type", _enum("entitytype"), nullable=False),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("internal_id", postgresql.UUID(as_uuid=True), nullable=False),
        _observed_at(),
        *_timestamps(),
        sa.UniqueConstraint(
            "provider_key", "entity_type", "external_id", name="uq_external_refs_provider_type_id"
        ),
    )
    op.create_index("ix_external_refs_observed_at", "external_refs", ["observed_at"])
    op.create_index("ix_external_refs_internal", "external_refs", ["entity_type", "internal_id"])

    # -------------------------------------------------- interval affiliations
    op.create_table(
        "player_team_stints",
        _uuid_pk(),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("season_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("jersey_number", sa.String(8), nullable=True),
        sa.Column("position", _enum("playerposition"), nullable=False, server_default="UNKNOWN"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("previous_team_name", sa.String(160), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_player_team_stints_player_id_players"
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_player_team_stints_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"], name="fk_player_team_stints_season_id_seasons"
        ),
        sa.CheckConstraint(
            "effective_until IS NULL OR effective_until > effective_from",
            name="effective_interval_valid",
        ),
    )
    op.create_index("ix_player_team_stints_observed_at", "player_team_stints", ["observed_at"])
    op.create_index(
        "ix_player_team_stints_effective_from", "player_team_stints", ["effective_from"]
    )
    op.create_index(
        "ix_player_team_stints_effective_until", "player_team_stints", ["effective_until"]
    )
    op.create_index(
        "ix_player_team_stints_player_from", "player_team_stints", ["player_id", "effective_from"]
    )
    op.create_index(
        "ix_player_team_stints_team_season", "player_team_stints", ["team_id", "season_id"]
    )

    op.create_table(
        "team_coach_stints",
        _uuid_pk(),
        sa.Column("coach_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("season_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["coach_id"], ["coaches.id"], name="fk_team_coach_stints_coach_id_coaches"
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_team_coach_stints_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"], name="fk_team_coach_stints_season_id_seasons"
        ),
        sa.CheckConstraint(
            "effective_until IS NULL OR effective_until > effective_from",
            name="effective_interval_valid",
        ),
    )
    op.create_index("ix_team_coach_stints_observed_at", "team_coach_stints", ["observed_at"])
    op.create_index("ix_team_coach_stints_effective_from", "team_coach_stints", ["effective_from"])
    op.create_index(
        "ix_team_coach_stints_effective_until", "team_coach_stints", ["effective_until"]
    )
    op.create_index(
        "ix_team_coach_stints_team_from", "team_coach_stints", ["team_id", "effective_from"]
    )

    # ---------------------------------------------------------------- games
    op.create_table(
        "games",
        _uuid_pk(),
        sa.Column("season_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_code", sa.Integer(), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=True),
        sa.Column("phase_code", sa.String(16), nullable=True),
        sa.Column("phase_name", sa.String(64), nullable=True),
        sa.Column("group_name", sa.String(64), nullable=True),
        sa.Column("tip_off_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tip_off_local", sa.DateTime(timezone=True), nullable=True),
        sa.Column("local_timezone", sa.String(64), nullable=True),
        sa.Column("status", _enum("gamestatus"), nullable=False, server_default="SCHEDULED"),
        sa.Column("home_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("away_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("is_neutral_venue", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("venue_name", sa.String(160), nullable=True),
        sa.Column("attendance", sa.Integer(), nullable=True),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"], name="fk_games_season_id_seasons"),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"], name="fk_games_home_team_id_teams"),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"], name="fk_games_away_team_id_teams"),
        sa.UniqueConstraint("season_id", "game_code", name="uq_games_season_game_code"),
    )
    op.create_index("ix_games_observed_at", "games", ["observed_at"])
    op.create_index("ix_games_round_number", "games", ["round_number"])
    op.create_index("ix_games_tip_off_utc", "games", ["tip_off_utc"])
    # These three carry the rest and schedule-density features, which are the
    # hottest queries in the feature layer.
    op.create_index("ix_games_season_tipoff", "games", ["season_id", "tip_off_utc"])
    op.create_index("ix_games_home_tipoff", "games", ["home_team_id", "tip_off_utc"])
    op.create_index("ix_games_away_tipoff", "games", ["away_team_id", "tip_off_utc"])

    op.create_table(
        "game_teams",
        _uuid_pk(),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("side", _enum("teamside"), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("quarter_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("coach_id", postgresql.UUID(as_uuid=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["game_id"], ["games.id"], name="fk_game_teams_game_id_games", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], name="fk_game_teams_team_id_teams"),
        sa.ForeignKeyConstraint(
            ["coach_id"], ["coaches.id"], name="fk_game_teams_coach_id_coaches"
        ),
        sa.UniqueConstraint("game_id", "side", name="uq_game_teams_game_side"),
    )
    op.create_index("ix_game_teams_game_id", "game_teams", ["game_id"])
    op.create_index("ix_game_teams_team_id", "game_teams", ["team_id"])

    op.create_table(
        "player_game_stats",
        _uuid_pk(),
        sa.Column("game_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seconds_played", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_starter", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("did_not_play", sa.Boolean(), nullable=False, server_default=sa.false()),
        *_counting_stats(),
        sa.Column("plus_minus", sa.Integer(), nullable=True),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["game_team_id"],
            ["game_teams.id"],
            name="fk_player_game_stats_game_team_id_game_teams",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_player_game_stats_player_id_players"
        ),
        sa.UniqueConstraint("game_team_id", "player_id", name="uq_player_game_stats_team_player"),
        sa.CheckConstraint("seconds_played >= 0", name="seconds_played_non_negative"),
    )
    op.create_index("ix_player_game_stats_observed_at", "player_game_stats", ["observed_at"])
    op.create_index("ix_player_game_stats_player", "player_game_stats", ["player_id"])

    op.create_table(
        "team_game_stats",
        _uuid_pk(),
        sa.Column("game_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        *_counting_stats(),
        sa.Column("possessions", sa.Numeric(8, 3), nullable=True),
        sa.Column("pace", sa.Numeric(8, 3), nullable=True),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["game_team_id"],
            ["game_teams.id"],
            name="fk_team_game_stats_game_team_id_game_teams",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("game_team_id", name="uq_team_game_stats_game_team"),
    )
    op.create_index("ix_team_game_stats_observed_at", "team_game_stats", ["observed_at"])

    # ----------------------------------------------------------------- news
    op.create_table(
        "news_sources",
        _uuid_pk(),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("feed_url", sa.String(1024), nullable=False),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("reliability_weight", sa.Numeric(4, 3), nullable=False, server_default="1.0"),
        *_timestamps(),
        sa.UniqueConstraint("key", name="uq_news_sources_key"),
    )

    op.create_table(
        "news_articles",
        _uuid_pk(),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(256), nullable=True),
        sa.Column("url", sa.String(1024), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("author", sa.String(160), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["source_id"], ["news_sources.id"], name="fk_news_articles_source_id_news_sources"
        ),
        sa.UniqueConstraint("url", name="uq_news_articles_url"),
    )
    op.create_index("ix_news_articles_observed_at", "news_articles", ["observed_at"])
    op.create_index("ix_news_articles_published", "news_articles", ["published_at"])
    op.create_index("ix_news_articles_hash", "news_articles", ["content_hash"])

    op.create_table(
        "news_events",
        _uuid_pk(),
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", _enum("newseventtype"), nullable=False),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"
        ),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("extraction_model", sa.String(128), nullable=True),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["article_id"], ["news_articles.id"], name="fk_news_events_article_id_news_articles"
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_news_events_player_id_players"
        ),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], name="fk_news_events_team_id_teams"),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_in_unit_interval"
        ),
    )
    op.create_index("ix_news_events_observed_at", "news_events", ["observed_at"])
    op.create_index("ix_news_events_player_observed", "news_events", ["player_id", "observed_at"])
    op.create_index("ix_news_events_type_observed", "news_events", ["event_type", "observed_at"])

    # --------------------------------------------------------- availability
    op.create_table(
        "player_status_events",
        _uuid_pk(),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", _enum("availabilitystatus"), nullable=False),
        sa.Column("reason", sa.String(512), nullable=True),
        sa.Column("injury_type", sa.String(128), nullable=True),
        sa.Column("expected_return_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "source_type", _enum("statussourcetype"), nullable=False, server_default="PROVIDER"
        ),
        sa.Column("source_article_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_player_status_events_player_id_players"
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_player_status_events_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["game_id"], ["games.id"], name="fk_player_status_events_game_id_games"
        ),
        sa.ForeignKeyConstraint(
            ["source_article_id"],
            ["news_articles.id"],
            name="fk_player_status_events_source_article_id_news_articles",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_in_unit_interval",
        ),
    )
    op.create_index("ix_player_status_events_observed_at", "player_status_events", ["observed_at"])
    op.create_index(
        "ix_player_status_events_event_timestamp", "player_status_events", ["event_timestamp"]
    )
    # Backs "latest known status for this player as of T", the dominant query.
    op.create_index(
        "ix_player_status_events_player_observed",
        "player_status_events",
        ["player_id", "observed_at", "event_timestamp"],
    )
    op.create_index("ix_player_status_events_game", "player_status_events", ["game_id"])

    op.create_table(
        "starting_lineup_entries",
        _uuid_pk(),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_confirmed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("announced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_type", _enum("statussourcetype"), nullable=False, server_default="PROVIDER"
        ),
        _observed_at(),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["game_id"], ["games.id"], name="fk_starting_lineup_entries_game_id_games"
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_starting_lineup_entries_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_starting_lineup_entries_player_id_players"
        ),
    )
    op.create_index(
        "ix_starting_lineup_entries_observed_at", "starting_lineup_entries", ["observed_at"]
    )
    op.create_index(
        "ix_starting_lineup_game_team_observed",
        "starting_lineup_entries",
        ["game_id", "team_id", "observed_at"],
    )

    # ------------------------------------------------------------- features
    op.create_table(
        "player_feature_snapshots",
        _uuid_pk(),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("opponent_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_player_feature_snapshots_player_id_players"
        ),
        sa.ForeignKeyConstraint(
            ["game_id"], ["games.id"], name="fk_player_feature_snapshots_game_id_games"
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_player_feature_snapshots_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["opponent_team_id"],
            ["teams.id"],
            name="fk_player_feature_snapshots_opponent_team_id_teams",
        ),
        sa.UniqueConstraint(
            "player_id",
            "game_id",
            "feature_set_version",
            "prediction_timestamp",
            name="uq_player_feature_snapshots_key",
        ),
    )
    op.create_index(
        "ix_player_feature_snapshots_prediction_timestamp",
        "player_feature_snapshots",
        ["prediction_timestamp"],
    )
    op.create_index("ix_player_feature_snapshots_game", "player_feature_snapshots", ["game_id"])
    op.create_index(
        "ix_player_feature_snapshots_player_ts",
        "player_feature_snapshots",
        ["player_id", "prediction_timestamp"],
    )

    op.create_table(
        "team_feature_snapshots",
        _uuid_pk(),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("opponent_team_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_team_feature_snapshots_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["game_id"], ["games.id"], name="fk_team_feature_snapshots_game_id_games"
        ),
        sa.ForeignKeyConstraint(
            ["opponent_team_id"],
            ["teams.id"],
            name="fk_team_feature_snapshots_opponent_team_id_teams",
        ),
        sa.UniqueConstraint(
            "team_id",
            "game_id",
            "feature_set_version",
            "prediction_timestamp",
            name="uq_team_feature_snapshots_key",
        ),
    )
    op.create_index(
        "ix_team_feature_snapshots_prediction_timestamp",
        "team_feature_snapshots",
        ["prediction_timestamp"],
    )
    op.create_index("ix_team_feature_snapshots_game", "team_feature_snapshots", ["game_id"])

    # ---------------------------------------------------------- predictions
    op.create_table(
        "model_versions",
        _uuid_pk(),
        sa.Column("name", sa.String(96), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("target", _enum("predictiontarget"), nullable=False),
        sa.Column("algorithm", sa.String(64), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column(
            "hyperparameters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("training_data_hash", sa.String(64), nullable=True),
        sa.Column(
            "metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"
        ),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_path", sa.String(512), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        *_timestamps(),
        sa.UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
    )
    op.create_index("ix_model_versions_target_active", "model_versions", ["target", "is_active"])

    op.create_table(
        "predictions",
        _uuid_pk(),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("player_feature_snapshot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target", _enum("predictiontarget"), nullable=False),
        sa.Column("predicted_value", sa.Float(), nullable=False),
        sa.Column("predicted_std", sa.Float(), nullable=True),
        sa.Column("quantiles", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "threshold_probabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_predictions_model_version_id_model_versions",
        ),
        sa.ForeignKeyConstraint(
            ["player_feature_snapshot_id"],
            ["player_feature_snapshots.id"],
            # Shortened deliberately: the convention-generated name would be 66
            # characters, past PostgreSQL's 63-byte identifier limit. Must stay
            # in sync with the explicit name on the ORM column.
            name="fk_predictions_player_feature_snapshot_id",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"], name="fk_predictions_player_id_players"
        ),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], name="fk_predictions_game_id_games"),
        sa.UniqueConstraint(
            "model_version_id",
            "player_id",
            "game_id",
            "target",
            "prediction_timestamp",
            name="uq_predictions_key",
        ),
    )
    op.create_index("ix_predictions_prediction_timestamp", "predictions", ["prediction_timestamp"])
    op.create_index("ix_predictions_game_target", "predictions", ["game_id", "target"])
    op.create_index(
        "ix_predictions_player_ts", "predictions", ["player_id", "prediction_timestamp"]
    )

    op.create_table(
        "prediction_outcomes",
        _uuid_pk(),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actual_value", sa.Float(), nullable=False),
        sa.Column("absolute_error", sa.Float(), nullable=False),
        sa.Column("squared_error", sa.Float(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_prediction_outcomes_prediction_id_predictions",
        ),
        sa.UniqueConstraint("prediction_id", name="uq_prediction_outcomes_prediction"),
    )

    # ------------------------------------------------------------ audit
    op.create_table(
        "ingestion_runs",
        _uuid_pk(),
        sa.Column("job_name", sa.String(64), nullable=False),
        sa.Column("provider_key", sa.String(64), nullable=False),
        sa.Column("season_code", sa.String(32), nullable=True),
        sa.Column("status", _enum("ingestionstatus"), nullable=False, server_default="RUNNING"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_read", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "details", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"
        ),
        *_timestamps(),
    )
    op.create_index("ix_ingestion_runs_job_started", "ingestion_runs", ["job_name", "started_at"])


def downgrade() -> None:
    # Reverse dependency order. Indexes go with their tables.
    for table in (
        "ingestion_runs",
        "prediction_outcomes",
        "predictions",
        "model_versions",
        "team_feature_snapshots",
        "player_feature_snapshots",
        "starting_lineup_entries",
        "player_status_events",
        "news_events",
        "news_articles",
        "news_sources",
        "team_game_stats",
        "player_game_stats",
        "game_teams",
        "games",
        "team_coach_stints",
        "player_team_stints",
        "external_refs",
        "team_seasons",
        "seasons",
        "players",
        "coaches",
        "teams",
        "leagues",
    ):
        op.drop_table(table)

    bind = op.get_bind()
    for name in _ENUMS:
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)

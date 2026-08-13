"""Translate EuroLeague API payloads into provider-neutral DTOs.

Pure functions over already-decoded JSON. Keeping them separate from the HTTP
client means every mapping rule is testable against a committed fixture with no
network, which matters because these are the rules most likely to drift when the
upstream API changes.

Mapping is deliberately defensive: fields are read with ``.get()`` chains and
unparseable rows are skipped rather than raising. A single malformed record in a
340-game backfill should cost one row, not the whole run.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from euroleague_ai.models.enums import GameStatus, PlayerPosition
from euroleague_ai.providers.base import (
    GameDTO,
    GameStatsDTO,
    PlayerLineDTO,
    PlayerStintDTO,
    TeamDTO,
    TeamLineDTO,
    TeamStatsDTO,
)
from euroleague_ai.providers.parsing import (
    is_did_not_play,
    normalise_person_code,
    parse_clock_to_seconds,
    parse_date,
    parse_datetime,
    parse_int,
    parse_optional_int,
)

# The upstream API encodes position as a small integer, with positionName as a
# parallel string. The integer is more stable across localisations.
_POSITION_BY_CODE: dict[int, PlayerPosition] = {
    1: PlayerPosition.GUARD,
    2: PlayerPosition.FORWARD,
    3: PlayerPosition.CENTER,
}
_POSITION_BY_NAME: dict[str, PlayerPosition] = {
    "GUARD": PlayerPosition.GUARD,
    "FORWARD": PlayerPosition.FORWARD,
    "CENTER": PlayerPosition.CENTER,
    "CENTRE": PlayerPosition.CENTER,
}

_GAME_STATUS_BY_NAME: dict[str, GameStatus] = {
    "SCHEDULED": GameStatus.SCHEDULED,
    "PENDING": GameStatus.SCHEDULED,
    "LIVE": GameStatus.LIVE,
    "PLAYING": GameStatus.LIVE,
    "INPROGRESS": GameStatus.LIVE,
    "FINISHED": GameStatus.FINISHED,
    "PLAYED": GameStatus.FINISHED,
    "COMPLETE": GameStatus.FINISHED,
    "POSTPONED": GameStatus.POSTPONED,
    "SUSPENDED": GameStatus.POSTPONED,
    "CANCELLED": GameStatus.CANCELLED,
    "CANCELED": GameStatus.CANCELLED,
}


def _dig(payload: Any, *keys: str) -> Any:
    """Walk nested dicts, returning ``None`` at the first missing or non-dict level."""
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str | None:
    """Return a stripped non-empty string, or ``None``.

    The API uses ``""`` where it means "absent" (empty ``twitterAccount``,
    ``abbreviatedName``), and an empty string stored as a name would be worse
    than a null.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def map_position(payload: dict[str, Any]) -> PlayerPosition:
    """Resolve a position from the numeric code, falling back to the name."""
    code = payload.get("position")
    if isinstance(code, int) and code in _POSITION_BY_CODE:
        return _POSITION_BY_CODE[code]
    name = _text(payload.get("positionName"))
    if name:
        return _POSITION_BY_NAME.get(name.upper(), PlayerPosition.UNKNOWN)
    return PlayerPosition.UNKNOWN


def map_game_status(payload: dict[str, Any]) -> GameStatus:
    """Resolve game status from ``gameStatus``, falling back to the ``played`` flag.

    ``gameStatus`` is free-form upstream and its vocabulary is not documented, so
    the boolean ``played`` is used as a backstop. Getting this wrong matters:
    box scores are only ingested for finished games, and a game wrongly marked
    finished would produce an empty stat line that pollutes form averages.
    """
    raw = _text(payload.get("gameStatus"))
    if raw:
        normalised = raw.upper().replace(" ", "").replace("_", "")
        if normalised in _GAME_STATUS_BY_NAME:
            return _GAME_STATUS_BY_NAME[normalised]
    return GameStatus.FINISHED if payload.get("played") is True else GameStatus.SCHEDULED


def map_team(payload: dict[str, Any]) -> TeamDTO | None:
    """Map one entry of the ``/clubs`` response."""
    code = _text(payload.get("code"))
    name = _text(payload.get("name"))
    if not code or not name:
        return None
    return TeamDTO(
        external_id=code,
        code=code,
        # clubPermanentName is the stable identity ("Saski Baskonia"); `name` is
        # the sponsored form and changes yearly ("Kosner Baskonia Vitoria-Gasteiz").
        # Season-specific naming belongs on TeamSeason, not on the club row.
        name=_text(payload.get("clubPermanentName")) or name,
        short_name=_text(payload.get("abbreviatedName")) or _text(payload.get("editorialName")),
        country_code=_text(_dig(payload, "country", "code")),
        city=_text(payload.get("city")),
        venue_name=None,
    )


def map_team_display_name(payload: dict[str, Any]) -> str:
    """Season-specific sponsored name, for :class:`TeamSeason`."""
    return _text(payload.get("name")) or _text(payload.get("clubPermanentName")) or "Unknown"


def _split_person_name(person: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Return ``(full_name, last_name, first_name)``.

    The API gives ``name`` as ``"SURNAME, FORENAME"`` plus separate
    ``passportSurname`` / ``passportName`` fields. The passport fields are
    preferred because they survive the comma-splitting ambiguity introduced by
    suffixed names such as ``"DOWTIN JR, JEFFREY"`` and ``"CLARK III, JIMMY"``.
    """
    full_name = _text(person.get("name")) or ""
    last_name = _text(person.get("passportSurname"))
    first_name = _text(person.get("passportName"))

    if not last_name and "," in full_name:
        surname, _, forename = full_name.partition(",")
        last_name = _text(surname)
        first_name = first_name or _text(forename)
    return full_name, last_name, first_name


def map_player_stint(payload: dict[str, Any]) -> PlayerStintDTO | None:
    """Map one entry of the ``/people`` response.

    Each entry is a *registration*, not just a person: it carries ``startDate``
    and ``endDate``, which become the validity interval of a
    :class:`~euroleague_ai.models.people.PlayerTeamStint`. That is what makes
    mid-season transfers representable without rewriting history.
    """
    person = payload.get("person")
    if not isinstance(person, dict):
        return None

    external_id = normalise_person_code(_text(person.get("code")))
    full_name, last_name, first_name = _split_person_name(person)
    effective_from = parse_datetime(_text(payload.get("startDate")))

    if not external_id or not full_name or effective_from is None:
        return None

    return PlayerStintDTO(
        external_id=external_id,
        full_name=full_name,
        last_name=last_name,
        first_name=first_name,
        birth_date=parse_date(_text(person.get("birthDate"))),
        country_code=_text(_dig(person, "country", "code")),
        birth_country_code=_text(_dig(person, "birthCountry", "code")),
        height_cm=parse_optional_int(person.get("height")),
        weight_kg=parse_optional_int(person.get("weight")),
        position=map_position(payload),
        team_external_id=_text(_dig(payload, "club", "code")),
        jersey_number=_text(payload.get("dorsal")),
        previous_team_name=_text(payload.get("lastTeam")),
        is_active=bool(payload.get("active", True)),
        effective_from=effective_from,
        effective_until=parse_datetime(_text(payload.get("endDate"))),
    )


def map_club_roster_entry(
    payload: dict[str, Any],
    *,
    club_code: str,
    season_start: date,
    season_end: date,
) -> PlayerStintDTO | None:
    """Map one entry of ``/clubs/{code}/people`` -- the authoritative roster.

    **This endpoint, not the season-wide ``/people``.** The season-wide feed
    returns a partial list of recently-created registrations, not a squad: for
    E2026 it held 204 rows against 20 full rosters. The club-scoped endpoint is
    what the official site renders, and it returns the real squad.

    Two shape differences from the season feed:

    * Coaches are included as ``type: "E"`` and must be filtered out here.
    * ``startDate`` / ``endDate`` are generic contract windows (2025-08-13 to
      2027-08-13) rather than registration dates, so they span neighbouring
      seasons. Left as-is they would make a player simultaneously registered in
      two seasons, which is exactly the ambiguity transfer resolution cannot
      tolerate. They are therefore clamped to the season's own window.
    """
    if payload.get("type") != "J":
        return None  # coach or staff

    person = payload.get("person")
    if not isinstance(person, dict):
        return None

    external_id = normalise_person_code(_text(person.get("code")))
    full_name, last_name, first_name = _split_person_name(person)
    if not external_id or not full_name:
        return None

    season_from = datetime.combine(season_start, time.min, tzinfo=UTC)
    season_until = datetime.combine(season_end, time.min, tzinfo=UTC)

    raw_from = parse_datetime(_text(payload.get("startDate")))
    raw_until = parse_datetime(_text(payload.get("endDate")))

    effective_from = max(raw_from, season_from) if raw_from else season_from
    effective_until = min(raw_until, season_until) if raw_until else season_until

    if effective_until <= effective_from:
        # A contract window that does not overlap the season at all.
        return None

    return PlayerStintDTO(
        external_id=external_id,
        full_name=full_name,
        last_name=last_name,
        first_name=first_name,
        birth_date=parse_date(_text(person.get("birthDate"))),
        country_code=_text(_dig(person, "country", "code")),
        birth_country_code=_text(_dig(person, "birthCountry", "code")),
        height_cm=parse_optional_int(person.get("height")),
        weight_kg=parse_optional_int(person.get("weight")),
        position=map_position(payload),
        team_external_id=_text(_dig(payload, "club", "code")) or club_code,
        jersey_number=_text(payload.get("dorsal")),
        previous_team_name=_text(payload.get("lastTeam")),
        is_active=bool(payload.get("active", True)),
        effective_from=effective_from,
        effective_until=effective_until,
    )


def _map_partials(payload: Any) -> dict[str, int] | None:
    """Normalise the per-quarter score object into ``{"1": 21, "2": 19, ...}``."""
    if not isinstance(payload, dict):
        return None
    result: dict[str, int] = {}
    for key, value in payload.items():
        if value is None:
            continue
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        if digits:
            result[digits] = parse_int(value)
    return result or None


def map_game(payload: dict[str, Any]) -> GameDTO | None:
    """Map one entry of the ``/games`` response."""
    home = payload.get("local")
    away = payload.get("road")
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None

    home_code = _text(_dig(home, "club", "code"))
    away_code = _text(_dig(away, "club", "code"))
    game_code = payload.get("gameCode")

    # utcDate is authoritative. localDate is kept only for travel and circadian
    # features and must never be used for time-window arithmetic.
    tip_off_utc = parse_datetime(_text(payload.get("utcDate")))
    if tip_off_utc is None:
        tip_off_utc = parse_datetime(_text(payload.get("date")))

    if not home_code or not away_code or game_code is None or tip_off_utc is None:
        return None

    # `round` has been observed both at the top level and nested under `group`.
    round_number = payload.get("round")
    if round_number is None:
        round_number = _dig(payload, "group", "round")

    return GameDTO(
        external_id=str(_text(payload.get("identifier")) or f"{game_code}"),
        game_code=parse_int(game_code),
        round_number=parse_optional_int(round_number),
        phase_code=_text(_dig(payload, "group", "phaseType", "code")),
        phase_name=_text(_dig(payload, "group", "phaseType", "name")),
        group_name=_text(_dig(payload, "group", "name")),
        tip_off_utc=tip_off_utc,
        tip_off_local=parse_datetime(_text(payload.get("localDate")), as_utc=False),
        local_timezone=_text(payload.get("localTimeZone")),
        status=map_game_status(payload),
        home_team_external_id=home_code,
        away_team_external_id=away_code,
        home_score=parse_optional_int(home.get("score")),
        away_score=parse_optional_int(away.get("score")),
        home_quarter_scores=_map_partials(home.get("partials")),
        away_quarter_scores=_map_partials(away.get("partials")),
        is_neutral_venue=bool(payload.get("isNeutralVenue", False)),
        venue_name=_text(_dig(payload, "venue", "name")),
        attendance=parse_optional_int(payload.get("audience")),
    )


def _map_player_line(entry: dict[str, Any]) -> PlayerLineDTO | None:
    """Map one ``players[]`` entry of the game stats response."""
    person = _dig(entry, "player", "person")
    stats = entry.get("stats")
    if not isinstance(person, dict) or not isinstance(stats, dict):
        return None

    external_id = normalise_person_code(_text(person.get("code")))
    if not external_id:
        return None

    raw_time = stats.get("timePlayed")
    seconds = parse_clock_to_seconds(raw_time)

    return PlayerLineDTO(
        player_external_id=external_id,
        player_name=_text(person.get("name")) or external_id,
        jersey_number=_text(_dig(entry, "player", "dorsal")),
        seconds_played=seconds,
        # startFive is the confirmed starter flag. startFive2 marks the second-half
        # starting five, which is not the same thing and is deliberately ignored.
        is_starter=bool(stats.get("startFive", False)),
        did_not_play=is_did_not_play(raw_time, seconds),
        points=parse_int(stats.get("points")),
        fg2m=parse_int(stats.get("fieldGoalsMade2")),
        fg2a=parse_int(stats.get("fieldGoalsAttempted2")),
        fg3m=parse_int(stats.get("fieldGoalsMade3")),
        fg3a=parse_int(stats.get("fieldGoalsAttempted3")),
        ftm=parse_int(stats.get("freeThrowsMade")),
        fta=parse_int(stats.get("freeThrowsAttempted")),
        offensive_rebounds=parse_int(stats.get("offensiveRebounds")),
        defensive_rebounds=parse_int(stats.get("defensiveRebounds")),
        total_rebounds=parse_int(stats.get("totalRebounds")),
        assists=parse_int(stats.get("assistances")),
        steals=parse_int(stats.get("steals")),
        turnovers=parse_int(stats.get("turnovers")),
        blocks_favour=parse_int(stats.get("blocksFavour")),
        blocks_against=parse_int(stats.get("blocksAgainst")),
        # "foulsCommited" is misspelled upstream. Mirroring the typo in our own
        # schema would spread it; it is corrected here at the boundary.
        fouls_committed=parse_int(stats.get("foulsCommited", stats.get("foulsCommitted"))),
        fouls_received=parse_int(stats.get("foulsReceived")),
        valuation=parse_int(stats.get("valuation")),
        plus_minus=parse_optional_int(stats.get("plusMinus")),
    )


def _map_team_line(totals: Any) -> TeamLineDTO:
    """Map the ``total`` object of one side into team totals."""
    if not isinstance(totals, dict):
        return TeamLineDTO()
    return TeamLineDTO(
        points=parse_int(totals.get("points")),
        fg2m=parse_int(totals.get("fieldGoalsMade2")),
        fg2a=parse_int(totals.get("fieldGoalsAttempted2")),
        fg3m=parse_int(totals.get("fieldGoalsMade3")),
        fg3a=parse_int(totals.get("fieldGoalsAttempted3")),
        ftm=parse_int(totals.get("freeThrowsMade")),
        fta=parse_int(totals.get("freeThrowsAttempted")),
        offensive_rebounds=parse_int(totals.get("offensiveRebounds")),
        defensive_rebounds=parse_int(totals.get("defensiveRebounds")),
        total_rebounds=parse_int(totals.get("totalRebounds")),
        assists=parse_int(totals.get("assistances")),
        steals=parse_int(totals.get("steals")),
        turnovers=parse_int(totals.get("turnovers")),
        blocks_favour=parse_int(totals.get("blocksFavour")),
        blocks_against=parse_int(totals.get("blocksAgainst")),
        fouls_committed=parse_int(totals.get("foulsCommited", totals.get("foulsCommitted"))),
        fouls_received=parse_int(totals.get("foulsReceived")),
        valuation=parse_int(totals.get("valuation")),
    )


def _map_team_stats(side: Any) -> TeamStatsDTO:
    """Map one side (``local`` or ``road``) of the game stats response."""
    if not isinstance(side, dict):
        return TeamStatsDTO()

    entries = side.get("players")
    lines: list[PlayerLineDTO] = []
    team_external_id: str | None = None

    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if team_external_id is None:
                team_external_id = _text(_dig(entry, "player", "club", "code"))
            if (line := _map_player_line(entry)) is not None:
                lines.append(line)

    return TeamStatsDTO(
        team_external_id=team_external_id,
        coach_external_id=normalise_person_code(_text(_dig(side, "coach", "code"))),
        coach_name=_text(_dig(side, "coach", "name")),
        players=tuple(lines),
        totals=_map_team_line(side.get("total")),
    )


def map_game_stats(game_code: int, payload: dict[str, Any]) -> GameStatsDTO | None:
    """Map the ``/games/{gameCode}/stats`` response.

    ``local`` is the home side and ``road`` the away side. Returns ``None`` when
    neither side lists any players, which is how an unplayed game presents.
    """
    home = _map_team_stats(payload.get("local"))
    away = _map_team_stats(payload.get("road"))
    if not home.players and not away.players:
        return None
    return GameStatsDTO(game_code=game_code, home=home, away=away)

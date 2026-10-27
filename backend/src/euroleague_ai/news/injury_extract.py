"""Injury statuses from article text, by rule.

**Why rules rather than a language model, for now.** EuroLeague injury coverage
is formulaic: "X is ruled out for Friday's game", "Y remains a game-time
decision", "Z returns from a knee injury". Rules over the names of rostered
players read that reliably, cost nothing, give the same answer every time, and
show exactly why they decided what they decided. Every extraction carries the
sentence it came from. If the rules prove too blunt, a model-based extractor can
sit behind the same function signature and write the same rows.

**How it reads an article.**

1. Find rostered players by full name ("Jean Montero"), in either name order.
   After a full mention, the surname alone ("Montero") also counts in the same
   article, which is how journalists write. A surname that was never introduced
   in full is ignored: "James" or "Hall" on their own are too likely to be
   someone else.
2. Split each sentence into parts at contrast words (";", "but", "while",
   "whereas", a dash), and each part into clauses at commas, so "Fournier is
   out, Dorsey remains uncertain, and Montero could miss Friday" gives three
   separate readings. A clause that is only names takes the status of its own
   part ("Nunn, Sloukas and Bonga are all available"), never one from across a
   contrast word ("... while Rogkavopoulos is ruled out"). A part that names
   nobody and starts with a verb ("... but is back available") continues the
   previous part's subject.
3. Match status phrases in each clause: out, suspended, doubtful, questionable,
   available.
4. Skip negated phrases ("has not been ruled out", "no longer sidelined") and
   lower confidence for past tense ("was ruled out").

**Confidence.** Strong phrases ("ruled out", "will miss") start at 0.9 and
softer ones lower. Penalties apply for surname-only mentions, past tense, and
sentences naming several players with one shared phrase. Only "out" readings at
or above :data:`RULE_OUT_CONFIDENCE` change a prediction; everything is stored.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from euroleague_ai.features.injury_status import NEWS_RULE_OUT_CONFIDENCE
from euroleague_ai.ingestion.injuries import normalise
from euroleague_ai.models.enums import AvailabilityStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from euroleague_ai.ingestion.injuries import RosterPlayer

#: Bump when the rules change, so re-extraction can tell old readings from new.
EXTRACTOR_VERSION = "rules-v3"

#: A news-derived "out" changes a prediction only at or above this confidence.
#: Owned by the prediction side, imported here so the two can never disagree.
RULE_OUT_CONFIDENCE = NEWS_RULE_OUT_CONFIDENCE

# (pattern, status, base confidence). Order within a status does not matter;
# the highest-confidence match in a clause wins.
_PHRASES: tuple[tuple[str, AvailabilityStatus, float], ...] = (
    # Out
    (r"ruled out", AvailabilityStatus.OUT, 0.9),
    (r"will miss", AvailabilityStatus.OUT, 0.9),
    (r"(?:set|expected|going) to miss", AvailabilityStatus.OUT, 0.85),
    (r"not expected to (?:play|feature|be available)", AvailabilityStatus.OUT, 0.8),
    (r"downgraded to out", AvailabilityStatus.OUT, 0.9),
    (
        r"(?:is|are|remains?|will be|is still|are still|are all|are both) out\b",
        AvailabilityStatus.OUT,
        0.85,
    ),
    (r"(?:has|have) been out\b", AvailabilityStatus.OUT, 0.85),
    (r"out (?:for|with|indefinitely|of action)\b", AvailabilityStatus.OUT, 0.85),
    (r"sidelined", AvailabilityStatus.OUT, 0.85),
    (
        r"(?:won t|will not|cannot|can t) (?:play|feature|travel|be available)",
        AvailabilityStatus.OUT,
        0.85,
    ),
    (r"(?:did|does|will) not travel", AvailabilityStatus.OUT, 0.85),
    (r"(?:is|are|remains?|will be) unavailable", AvailabilityStatus.OUT, 0.85),
    (r"(?:is|are) injured", AvailabilityStatus.OUT, 0.7),
    (r"(?:is|are) (?:now )?listed as out\b", AvailabilityStatus.OUT, 0.85),
    (r"(?:is|are|was|has been) suspended", AvailabilityStatus.SUSPENDED, 0.9),
    # Doubtful
    (r"(?:is|remains?) doubtful", AvailabilityStatus.DOUBTFUL, 0.75),
    (r"downgraded to doubtful", AvailabilityStatus.DOUBTFUL, 0.85),
    (r"in doubt", AvailabilityStatus.DOUBTFUL, 0.7),
    (r"unlikely to play", AvailabilityStatus.DOUBTFUL, 0.75),
    # Questionable
    (r"game time decision", AvailabilityStatus.QUESTIONABLE, 0.8),
    # A change of status is news about now, so it outranks background lines.
    (r"(?:upgraded|downgraded) to questionable", AvailabilityStatus.QUESTIONABLE, 0.85),
    (r"questionable", AvailabilityStatus.QUESTIONABLE, 0.75),
    (r"day to day", AvailabilityStatus.QUESTIONABLE, 0.7),
    (
        r"(?:is|are) (?:now )?listed as (?:questionable|doubtful|a game time decision)",
        AvailabilityStatus.QUESTIONABLE,
        0.75,
    ),
    (r"(?:could|may|might) miss", AvailabilityStatus.QUESTIONABLE, 0.65),
    (
        r"(?:status|future|availability) (?:remains? |is )?(?:uncertain|unclear)",
        AvailabilityStatus.QUESTIONABLE,
        0.6,
    ),
    (r"(?:is|remains?) uncertain", AvailabilityStatus.QUESTIONABLE, 0.6),
    (r"question mark", AvailabilityStatus.QUESTIONABLE, 0.6),
    # Available
    (
        r"(?:is|are) (?:all |both |also |now )?(?:back|available|fit)\b",
        AvailabilityStatus.AVAILABLE,
        0.8,
    ),
    (r"(?:is|are) back (?:available|in action|in the squad)", AvailabilityStatus.AVAILABLE, 0.85),
    (r"available (?:once )?again", AvailabilityStatus.AVAILABLE, 0.8),
    (r"(?:is|are) (?:now )?listed as (?:available|probable)", AvailabilityStatus.AVAILABLE, 0.8),
    (r"(?:will|should) be (?:back|available)", AvailabilityStatus.AVAILABLE, 0.75),
    (r"(?:will|set to|expected to) return", AvailabilityStatus.AVAILABLE, 0.7),
    (r"returns? (?:to|from|for)\b", AvailabilityStatus.AVAILABLE, 0.75),
    (r"cleared to play", AvailabilityStatus.AVAILABLE, 0.85),
    (r"(?:will be|is) available", AvailabilityStatus.AVAILABLE, 0.8),
    (r"back in the (?:lineup|rotation|squad)", AvailabilityStatus.AVAILABLE, 0.8),
)
_COMPILED = tuple((re.compile(rf"\b{p}"), s, c) for p, s, c in _PHRASES)

#: A negation shortly before a phrase flips or voids it. Phrases that already
#: contain their own negation ("will not play") are not subject to this.
_NEGATION = re.compile(r"\b(?:not|no longer|never|n t|nt)\b(?:\s+\w+){0,2}\s*$")
_PAST = re.compile(r"\b(?:was|were|had been|had)\b(?:\s+\w+){0,1}\s*$")
#: Whole-sentence markers of background rather than current news ("Minaya
#: played the first three rounds, but ahead of Round 4 it was communicated that
#: he will be out"). Every reading from such a sentence loses confidence, so the
#: same article's current line ("Minaya is upgraded to questionable") wins.
_PAST_CONTEXT = re.compile(
    r"\b(?:last played|played the first|it was communicated|was communicated|"
    r"was announced|earlier this season|last season|last week|last month|"
    r"in the offseason|over the summer)\b"
)
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_GAME_WORD = r"(?:game|clash|match|contest|derby|meeting|showdown|trip|visit|encounter|matchup)"
#: A statement about one fixture: "ruled out for Efes's game at Zalgiris on
#: Friday", "for Friday's clash", "for Round 7", "tonight". Such readings cover
#: that game only, not every game until he plays again.
_GAME_REF = re.compile(
    rf"\b{_GAME_WORD}\b.*\b(?:{_WEEKDAY}|tonight|tomorrow)\b"
    rf"|\b{_WEEKDAY}(?: s)? {_GAME_WORD}\b"
    r"|\bfor round \d+\b|\btonight\b"
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a couple of": 2,
    "a few": 2,
    "several": 2,
}
_NUMBER = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few|several)"
#: "out for 6 to 8 weeks", "set to miss 20 days", "sidelined for several months".
#: The lower bound is used: coming back early is far more common than a player
#: staying out past the low end, and an over-long absence is the costlier error.
_DURATION = re.compile(
    rf"(?<!last )(?<!past )(?<!previous )\b{_NUMBER}(?: (?:to|or) {_NUMBER})? "
    r"(days?|weeks?|months?)\b"
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30}
#: Applied to the raw sentence, before punctuation is normalised away. Commas
#: separate players in a list of absences ("X is out, Y is uncertain"); a name
#: left without a phrase of its own borrows the sentence's only phrase.
#: Parts of a sentence that say different things about different players. A
#: status never crosses one of these.
_SEGMENT_SPLIT = re.compile(r";|\bbut\b|\bwhile\b|\bwhereas\b|\s[-–—]\s")
#: Clauses within a part: a list of names, an aside, a parenthesis.
_LIST_SPLIT = re.compile(r"[,:()]")
#: A part that starts like this has no subject of its own: "... but is back
#: available for Round 7" is still about the player named before "but".
_CONTINUATION = re.compile(
    r"^(?:is|are|was|were|will|would|has|have|had|remains?|should|could|may|might)\b"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ſ\"'“])|\n+")

_SURNAME_PENALTY = 0.1
_PAST_PENALTY = 0.25
_SHARED_PENALTY = 0.05
_PRIORITY = {
    AvailabilityStatus.OUT: 4,
    AvailabilityStatus.SUSPENDED: 4,
    AvailabilityStatus.DOUBTFUL: 3,
    AvailabilityStatus.QUESTIONABLE: 2,
    AvailabilityStatus.AVAILABLE: 1,
}


@dataclass(frozen=True, slots=True)
class Extracted:
    """One player's status as read from one article."""

    player_id: uuid.UUID
    team_id: uuid.UUID
    full_name: str
    status: AvailabilityStatus
    confidence: float
    phrase: str
    sentence: str
    #: The sentence is about one fixture ("for Friday's game").
    game_scoped: bool = False
    #: Stated length of the absence, in days (lower bound), if any.
    duration_days: int | None = None

    @property
    def scope(self) -> str:
        if self.duration_days is not None:
            return "until"
        return "game" if self.game_scoped else "open"

    @property
    def rules_out(self) -> bool:
        return (
            self.status in (AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED)
            and self.confidence >= RULE_OUT_CONFIDENCE
        )


@dataclass(frozen=True, slots=True)
class _NameForms:
    player: RosterPlayer
    full: tuple[re.Pattern[str], ...]
    surname: re.Pattern[str] | None


def _name_forms(player: RosterPlayer) -> _NameForms:
    """Regexes for "First Last", "Last First" and the bare surname."""
    raw = player.full_name
    if "," in raw:
        last_raw, first_raw = raw.split(",", 1)
    else:
        parts = raw.rsplit(" ", 1)
        first_raw, last_raw = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    last, first = normalise(last_raw), normalise(first_raw)
    forms = []
    if first and last:
        forms.append(re.compile(rf"\b{re.escape(first)} {re.escape(last)}\b"))
        forms.append(re.compile(rf"\b{re.escape(last)} {re.escape(first)}\b"))
    elif last:
        forms.append(re.compile(rf"\b{re.escape(last)}\b"))
    surname = re.compile(rf"\b{re.escape(last)}\b") if last and len(last) >= 3 else None
    return _NameForms(player, tuple(forms), surname)


class RosterIndex:
    """Name patterns for every rostered player, built once per extraction run."""

    def __init__(self, roster: Sequence[RosterPlayer]) -> None:
        self.forms = [_name_forms(p) for p in roster]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _best_phrase(clause: str) -> tuple[AvailabilityStatus, float, str] | None:
    """The strongest non-negated status phrase in a normalised clause."""
    best: tuple[AvailabilityStatus, float, str] | None = None
    for pattern, status, confidence in _COMPILED:
        for match in pattern.finditer(clause):
            before = clause[: match.start()]
            phrase = match.group(0)
            if (
                " not " not in f" {phrase} "
                and " t " not in f" {phrase} "
                and _NEGATION.search(before)
            ):
                continue
            score = confidence - (_PAST_PENALTY if _PAST.search(before) else 0.0)
            if best is None or (score, _PRIORITY[status]) > (best[1], _PRIORITY[best[0]]):
                best = (status, score, phrase)
    return best


def _number(raw: str | None) -> int | None:
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)


def _duration_days(norm: str) -> int | None:
    """Lower bound of a stated absence ("6 to 8 weeks" -> 42), or ``None``."""
    match = _DURATION.search(norm)
    if match is None:
        return None
    low = _number(match.group(1))
    unit = match.group(3).rstrip("s")
    if low is None or unit not in _UNIT_DAYS or low <= 0:
        return None
    return low * _UNIT_DAYS[unit]


_LIST_FILLER = frozenset({"and", "also", "both", "plus", "as", "well"})


def _only_names(clause: str, entries: list[tuple[_NameForms, bool]]) -> bool:
    """True when a clause holds nothing but player names (and "and")."""
    rest = clause
    for forms, surname_only in entries:
        patterns = (forms.surname,) if surname_only and forms.surname else forms.full
        for pattern in patterns:
            rest = pattern.sub(" ", rest)
    return all(word in _LIST_FILLER for word in rest.split())


def _next_phrase(
    phrases: list[tuple[AvailabilityStatus, float, str] | None],
    index: int,
    segments: list[int],
) -> int | None:
    """Index of the phrase a list of names belongs to: the next one along in
    the same part of the sentence, else the last one before it in that part."""
    part = segments[index]
    for offset in range(index + 1, len(phrases)):
        if segments[offset] != part:
            break
        if phrases[offset] is not None:
            return offset
    for offset in range(index - 1, -1, -1):
        if segments[offset] != part:
            break
        if phrases[offset] is not None:
            return offset
    return None


def _split_clauses(sentence: str) -> tuple[list[str], list[int]]:
    """Normalised clauses, and the part of the sentence each belongs to."""
    clauses: list[str] = []
    segments: list[int] = []
    for part, segment in enumerate(_SEGMENT_SPLIT.split(sentence)):
        for piece in _LIST_SPLIT.split(segment):
            text = normalise(piece)
            if text:
                clauses.append(text)
                segments.append(part)
    return clauses, segments


def _carry_subjects(
    clauses: list[str],
    segments: list[int],
    clause_players: list[list[tuple[_NameForms, bool]]],
    phrases: list[tuple[AvailabilityStatus, float, str] | None],
) -> None:
    """Give a subjectless part ("... but is back available") the subject of
    the part before it: the names in that part's first clause."""
    for index, clause in enumerate(clauses):
        starts_part = index == 0 or segments[index - 1] != segments[index]
        if (
            not starts_part
            or index == 0
            or clause_players[index]
            or phrases[index] is None
            or not _CONTINUATION.match(clause)
        ):
            continue
        previous = segments[index - 1]
        first = segments.index(previous)
        if clause_players[first]:
            clause_players[index] = list(clause_players[first])


def extract_statuses(text: str, index: RosterIndex) -> list[Extracted]:
    """One reading per player mentioned with a status phrase, strongest kept."""
    if not text:
        return []
    introduced: set[uuid.UUID] = set()
    readings: dict[uuid.UUID, Extracted] = {}

    for sentence in split_sentences(text):
        norm = normalise(sentence)
        if not norm:
            continue
        # Who is named in this sentence, and how.
        full_here = {f.player.player_id for f in index.forms if any(p.search(norm) for p in f.full)}
        introduced |= full_here

        clauses, segments = _split_clauses(sentence)
        clause_players: list[list[tuple[_NameForms, bool]]] = []
        for clause in clauses:
            found: list[tuple[_NameForms, bool]] = []
            for forms in index.forms:
                pid = forms.player.player_id
                if any(p.search(clause) for p in forms.full):
                    found.append((forms, False))
                elif (
                    pid in introduced and forms.surname is not None and forms.surname.search(clause)
                ):
                    found.append((forms, True))
            clause_players.append(found)

        phrases = [_best_phrase(c) for c in clauses]
        _carry_subjects(clauses, segments, clause_players, phrases)
        named = [entry for entries in clause_players for entry in entries]
        if not named or not any(phrases):
            continue
        background = bool(_PAST_CONTEXT.search(norm))
        # The fixture is a sentence-wide fact ("X is not expected to play while
        # Y is questionable for Zvezda's game on Friday": both mean Friday). A
        # duration belongs to its own clause ("Fournier set to miss 20 days,
        # Montero questionable" gives Montero no 20 days).
        game_scoped = bool(_GAME_REF.search(norm))
        durations = [_duration_days(c) for c in clauses]

        for clause_index, entries in enumerate(clause_players):
            for forms, surname_only in entries:
                phrase = phrases[clause_index]
                source = clause_index
                shared = False
                if phrase is None:
                    # Only a clause that is nothing but names borrows a phrase:
                    # "Mike James, Isaia Cordinier and Georgios Papagiannis are
                    # ruled out", or "Lessort, who hurt his ankle, will not play".
                    # "Fournier set to miss 20 days, Montero questionable" must
                    # not hand Montero's status to Fournier.
                    if not _only_names(clauses[clause_index], entries):
                        continue
                    borrowed = _next_phrase(phrases, clause_index, segments)
                    if borrowed is None:
                        continue
                    source, phrase, shared = borrowed, phrases[borrowed], True
                    if phrase is None:  # pragma: no cover - _next_phrase never returns one
                        continue
                status, confidence, matched = phrase
                confidence -= _SURNAME_PENALTY if surname_only else 0.0
                confidence -= _PAST_PENALTY if background else 0.0
                if shared or len(entries) > 1:
                    confidence -= _SHARED_PENALTY
                confidence = round(max(0.0, min(1.0, confidence)), 3)
                reading = Extracted(
                    player_id=forms.player.player_id,
                    team_id=forms.player.team_id,
                    full_name=forms.player.full_name,
                    status=status,
                    confidence=confidence,
                    phrase=matched,
                    sentence=sentence[:500],
                    game_scoped=game_scoped and status is not AvailabilityStatus.AVAILABLE,
                    duration_days=(
                        durations[source] if status is not AvailabilityStatus.AVAILABLE else None
                    ),
                )
                current = readings.get(reading.player_id)
                if current is None or (reading.confidence, _PRIORITY[reading.status]) > (
                    current.confidence,
                    _PRIORITY[current.status],
                ):
                    readings[reading.player_id] = reading
    return list(readings.values())

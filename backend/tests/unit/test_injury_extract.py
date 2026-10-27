"""Reading injury statuses out of article text, and parsing full-text feeds.

The sample sentences follow the shapes EuroLeague coverage actually uses: the
official round report ("X is ruled out for Paris's game at Olympiacos on
Friday"), news write-ups, and fantasy previews ("X is out, Y's future remains
uncertain, and Z could miss Friday's clash").
"""

from __future__ import annotations

import uuid

import pytest

from euroleague_ai.features.injury_status import resolve_status
from euroleague_ai.ingestion.injuries import RosterPlayer
from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType
from euroleague_ai.news.injury_extract import RosterIndex, extract_statuses, split_sentences
from euroleague_ai.news.rss import html_to_text, parse_feed
from euroleague_ai.news.store import source_key
from euroleague_ai.repositories.availability import StatusReport

OUT, Q, D, OK, SUSP = (
    AvailabilityStatus.OUT,
    AvailabilityStatus.QUESTIONABLE,
    AvailabilityStatus.DOUBTFUL,
    AvailabilityStatus.AVAILABLE,
    AvailabilityStatus.SUSPENDED,
)

NAMES = [
    "FOURNIER, EVAN",
    "DORSEY, TYLER",
    "MONTERO, JEAN",
    "HIFI, NADIR",
    "YILMAZ, ERKAN",
    "RIVERO, JASIEL",
    "FORREST, TRENT",
    "JAMES, MIKE",
    "HEZONJA, MARIO",
    "LESSORT, MATHIAS",
    "SLOUKAS, KOSTAS",
    "WALKUP, THOMAS",
    "MILUTINOV, NIKOLA",
    "VEZENKOV, SASHA",
    "MILLER-MCINTYRE, CODI",
    "ŠARIĆ, DARIO",
]


@pytest.fixture(scope="module")
def index() -> RosterIndex:
    return RosterIndex([RosterPlayer(uuid.uuid4(), n, uuid.uuid4(), "XXX", None) for n in NAMES])


def _read(text: str, index: RosterIndex) -> dict[str, tuple[AvailabilityStatus, float, bool]]:
    return {
        e.full_name: (e.status, e.confidence, e.rules_out) for e in extract_statuses(text, index)
    }


class TestPhrases:
    @pytest.mark.parametrize(
        ("sentence", "name", "status"),
        [
            (
                "Nadir Hifi is ruled out for Paris's game at Olympiacos on Friday, the club has confirmed.",
                "HIFI, NADIR",
                OUT,
            ),
            (
                "Erkan Yilmaz is out for the next three weeks with a left knee injury.",
                "YILMAZ, ERKAN",
                OUT,
            ),
            (
                "Jasiel Rivero hurt his foot and is expected to be sidelined for several months.",
                "RIVERO, JASIEL",
                OUT,
            ),
            ("Trent Forrest (hamstring) has been out since October 17.", "FORREST, TRENT", OUT),
            (
                "Mathias Lessort, who hurt his ankle on Tuesday, will not play against Real.",
                "LESSORT, MATHIAS",
                OUT,
            ),
            ("Mike James will be a game-time decision.", "JAMES, MIKE", Q),
            ("Kostas Sloukas is day-to-day with back spasms.", "SLOUKAS, KOSTAS", Q),
            ("Mario Hezonja is doubtful for Thursday.", "HEZONJA, MARIO", D),
            ("Thomas Walkup returns from injury after missing two games.", "WALKUP, THOMAS", OK),
            ("Nikola Milutinov has been cleared to play.", "MILUTINOV, NIKOLA", OK),
            ("Dario Šarić has been suspended for one game.", "ŠARIĆ, DARIO", SUSP),
        ],
    )
    def test_reads_the_status(
        self, sentence: str, name: str, status: AvailabilityStatus, index: RosterIndex
    ) -> None:
        assert _read(sentence, index)[name][0] is status

    def test_strong_outs_rule_the_player_out(self, index: RosterIndex) -> None:
        result = _read("Nadir Hifi is ruled out for Friday's game.", index)
        assert result["HIFI, NADIR"] == (OUT, pytest.approx(0.9), True)

    def test_a_scoring_line_is_not_an_injury(self, index: RosterIndex) -> None:
        assert _read("Sasha Vezenkov scored 25 points as Olympiacos beat Real Madrid.", index) == {}


class TestClausesAndGuards:
    def test_each_player_in_a_list_gets_his_own_status(self, index: RosterIndex) -> None:
        result = _read(
            "Evan Fournier is out, Tyler Dorsey’s immediate future remains uncertain, "
            "and Jean Montero could miss Friday’s clash with Fenerbahce.",
            index,
        )
        assert result["FOURNIER, EVAN"][0] is OUT
        assert result["DORSEY, TYLER"][0] is Q
        assert result["MONTERO, JEAN"][0] is Q
        assert not result["MONTERO, JEAN"][2]

    def test_shared_phrase_covers_every_name(self, index: RosterIndex) -> None:
        result = _read("Thomas Walkup and Nikola Milutinov are both out with minor knocks.", index)
        assert result["WALKUP, THOMAS"][0] is OUT and result["MILUTINOV, NIKOLA"][0] is OUT

    def test_negation_is_respected(self, index: RosterIndex) -> None:
        result = _read("Mike James has not been ruled out and will be a game-time decision.", index)
        assert result["JAMES, MIKE"][0] is Q
        assert _read("Evan Fournier is no longer sidelined.", index) == {}

    def test_past_tense_loses_to_the_present(self, index: RosterIndex) -> None:
        result = _read(
            "Mario Hezonja returns from injury after missing two games. Hezonja was ruled out last week.",
            index,
        )
        assert result["HEZONJA, MARIO"][0] is OK

    def test_past_tense_alone_is_not_confident_enough(self, index: RosterIndex) -> None:
        status, confidence, rules_out = _read(
            "Mario Hezonja was ruled out of the last game.", index
        )["HEZONJA, MARIO"]
        assert status is OUT and confidence < 0.8 and not rules_out

    def test_surname_counts_only_after_a_full_mention(self, index: RosterIndex) -> None:
        assert _read("James is out for Friday.", index) == {}
        result = _read("Mike James sat out practice. James is out for Friday.", index)
        assert result["JAMES, MIKE"][0] is OUT
        assert result["JAMES, MIKE"][1] == pytest.approx(0.75)  # surname-only penalty

    def test_hyphenated_and_accented_names(self, index: RosterIndex) -> None:
        assert _read("Codi Miller-McIntyre is ruled out.", index)["MILLER-MCINTYRE, CODI"][0] is OUT
        assert _read("Dario Saric is ruled out.", index)["ŠARIĆ, DARIO"][0] is OUT

    def test_sentences_split_on_paragraphs_and_stops(self) -> None:
        assert split_sentences("One is out. Two is back!\nThree") == [
            "One is out.",
            "Two is back!",
            "Three",
        ]


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>Euroleague - Example</title>
<item>
  <title>Injury report, Round 7</title>
  <link>https://example.test/en/euroleague/1/injury-report-round-7/</link>
  <pubDate>Wed, 21 Oct 2026 13:55:58 +0000</pubDate>
  <dc:creator><![CDATA[Example Writer]]></dc:creator>
  <guid isPermaLink="false">https://example.test/?p=1</guid>
  <description><![CDATA[<p>Short teaser</p>]]></description>
  <content:encoded><![CDATA[<p><strong>By Example</strong></p><p>Nadir Hifi is ruled out for Friday&#8217;s game.</p><h3>Next</h3><p>Mike James is a game-time decision.</p>]]></content:encoded>
</item>
</channel></rss>"""


class TestFeeds:
    def test_full_text_is_read_from_content_encoded(self) -> None:
        [article] = parse_feed(SAMPLE_FEED)
        assert article.title == "Injury report, Round 7"
        assert article.author == "Example Writer"
        assert article.summary == "Short teaser"
        assert article.body is not None
        assert "Nadir Hifi is ruled out for Friday’s game." in article.body.split("\n")
        assert article.published_at is not None and article.published_at.year == 2026

    def test_html_to_text_keeps_paragraphs(self) -> None:
        assert html_to_text("<p>A &amp; B</p><p>C<br>D</p>") == "A & B\nC\nD"
        assert html_to_text("") is None

    def test_source_key_is_readable_and_bounded(self) -> None:
        assert (
            source_key("https://www.eurohoops.net/en/euroleague/feed/")
            == "www.eurohoops.net/en/euroleague/feed"
        )
        assert len(source_key("https://x.test/" + "a" * 200)) == 64


def _news_report(confidence: float) -> StatusReport:
    from datetime import UTC, datetime

    at = datetime(2026, 10, 20, tzinfo=UTC)
    return StatusReport(
        id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        game_id=None,
        status=OUT,
        reason="is out",
        expected_return=None,
        source=StatusSourceType.NEWS,
        event_timestamp=at,
        observed_at=at,
        confidence=confidence,
    )


class TestNewsThreshold:
    def test_confident_news_rules_out_and_uncertain_news_only_flags(self) -> None:
        from datetime import UTC, datetime

        tip = datetime(2026, 10, 23, tzinfo=UTC)
        strong = resolve_status(
            [_news_report(0.85)], game_id=uuid.uuid4(), tip_off=tip, last_played=None
        )
        weak = resolve_status(
            [_news_report(0.7)], game_id=uuid.uuid4(), tip_off=tip, last_played=None
        )
        assert strong is not None and strong.rules_out and strong.as_dict()["applied"] is True
        assert weak is not None and not weak.rules_out and weak.as_dict()["applied"] is False


# Sentences taken verbatim from Eurohoops coverage in October 2026. Each one was
# misread by the first rules: an old injury kept a player out, or a one-game
# report lingered past that game.
REAL_NAMES = [
    "MINAYA, JUSTIN",
    "MAKOUNDOU, YOAN",
    "JAMES, MIKE",
    "CORDINIER, ISAIA",
    "PAPAGIANNIS, GEORGIOS",
    "MOTLEY, JOHNATHAN",
    "IZUNDU, EBUKA",
    "MILJENOVIC, STEFAN",
    "NTILIKINA, FRANK",
    "WILLIAMS-GOSS, NIGEL",
    "LUNDBERG, IFFE",
    "FOURNIER, EVAN",
    "MONTERO, JEAN",
    "DORSEY, TYLER",
    "WILLIS, DEREK",
    "BONGA, ISAAC",
]


@pytest.fixture(scope="module")
def real() -> RosterIndex:
    return RosterIndex(
        [RosterPlayer(uuid.uuid4(), n, uuid.uuid4(), "XXX", None) for n in REAL_NAMES]
    )


def _scoped(
    text: str, index: RosterIndex
) -> dict[str, tuple[AvailabilityStatus, bool, str, int | None]]:
    return {
        e.full_name: (e.status, e.rules_out, e.scope, e.duration_days)
        for e in extract_statuses(text, index)
    }


class TestScope:
    def test_round_report_lines_are_for_that_game(self, real: RosterIndex) -> None:
        result = _scoped(
            "Mike James, Isaia Cordinier and Georgios Papagiannis are the only players ruled out "
            "for Efes’s game at Zalgiris on Friday, the club has confirmed.",
            real,
        )
        assert {name: r[:3] for name, r in result.items()} == dict.fromkeys(
            ("JAMES, MIKE", "CORDINIER, ISAIA", "PAPAGIANNIS, GEORGIOS"), (OUT, True, "game")
        )

    def test_upgrade_wins_over_the_out_beside_it(self, real: RosterIndex) -> None:
        result = _scoped(
            "Justin Minaya is upgraded to questionable while Yoan Makoundou is ruled out for "
            "Barcelona’s game against Olympiacos on Friday, the club has confirmed.",
            real,
        )
        assert result["MINAYA, JUSTIN"][:3] == (Q, False, "game")
        assert result["MAKOUNDOU, YOAN"][:3] == (OUT, True, "game")

    def test_each_name_takes_the_phrase_that_follows_it(self, real: RosterIndex) -> None:
        result = _scoped(
            "Johnathan Motley is not expected to play while Ebuka Izundu and Stefan Miljenovic "
            "are questionable for Crvena Zvezda’s game against Paris on Friday.",
            real,
        )
        assert result["MOTLEY, JOHNATHAN"][:2] == (OUT, True)
        assert result["IZUNDU, EBUKA"][0] is Q
        assert result["MILJENOVIC, STEFAN"][0] is Q

    @pytest.mark.parametrize(
        ("sentence", "name", "days"),
        [
            (
                "Iffe Lundberg suffers an injury and will be sidelined for 6 to 8 weeks",
                "LUNDBERG, IFFE",
                42,
            ),
            (
                "Evan Fournier set to miss 20 days, Jean Montero questionable for Fenerbahce",
                "FOURNIER, EVAN",
                20,
            ),
        ],
    )
    def test_a_stated_absence_lasts_its_lower_bound(
        self, sentence: str, name: str, days: int, real: RosterIndex
    ) -> None:
        assert _scoped(sentence, real)[name] == (OUT, True, "until", days)

    def test_the_other_name_in_a_headline_keeps_its_own_status(self, real: RosterIndex) -> None:
        result = _scoped(
            "Evan Fournier set to miss 20 days, Jean Montero questionable for Fenerbahce", real
        )
        assert result["MONTERO, JEAN"] == (Q, False, "open", None)

    def test_a_plain_out_is_open(self, real: RosterIndex) -> None:
        assert _scoped("Evan Fournier is out.", real)["FOURNIER, EVAN"][2] == "open"


class TestBackground:
    """Recaps of earlier absences must not rule anyone out today."""

    @pytest.mark.parametrize(
        ("sentence", "name"),
        [
            (
                "Justin Minaya played the first three rounds of the season, but ahead of Round 4, it "
                "was communicated that he will be out for approximately 10 days with a calf issue.",
                "MINAYA, JUSTIN",
            ),
            (
                "Frank Ntilikina last played in Round 2 and was ruled out for Round 5 with muscle pain.",
                "NTILIKINA, FRANK",
            ),
            (
                "Nigel Williams-Goss was ruled out for two weeks with a hamstring injury on September 25.",
                "WILLIAMS-GOSS, NIGEL",
            ),
        ],
    )
    def test_recaps_do_not_rule_out(self, sentence: str, name: str, real: RosterIndex) -> None:
        reading = _scoped(sentence, real).get(name)
        assert reading is None or not reading[1]

    def test_a_missed_game_in_the_past_is_not_an_injury(self, real: RosterIndex) -> None:
        result = _scoped(
            "Due to his comments on social media, Tyler Dorsey missed the match against Barcelona, "
            "and his future with Olympiacos remains uncertain.",
            real,
        )
        assert "DORSEY, TYLER" not in result or not result["DORSEY, TYLER"][1]


# Round 7 report sentences (Eurohoops, 22 October 2026). The first one ruled
# Nunn out for a game he played 20 minutes in: "Kendrick Nunn" alone before the
# comma borrowed "ruled out" from across "while".
R7_NAMES = [
    "NUNN, KENDRICK",
    "SLOUKAS, KOSTAS",
    "BONGA, ISAAC",
    "ROGKAVOPOULOS, NIKOS",
    "MANTZOUKAS, LEFTERIS",
    "WALKER, LONNIE",
    "JEFFRIES, DAQUAN",
    "STEVENS, LAMAR",
    "JEKIRI, TONYE",
    "MARINKOVIC, VANJA",
    "WILLIS, DEREK",
    "MILJENOVIC, STEFAN",
]


@pytest.fixture(scope="module")
def r7() -> RosterIndex:
    return RosterIndex([RosterPlayer(uuid.uuid4(), n, uuid.uuid4(), "XXX", None) for n in R7_NAMES])


class TestContrastWords:
    def test_a_list_takes_the_status_of_its_own_part(self, r7: RosterIndex) -> None:
        result = _scoped(
            "Kendrick Nunn, Kostas Sloukas and Isaac Bonga are all available while Nikos "
            "Rogkavopoulos and Lefteris Mantzoukas are ruled out for Panathinaikos’s game "
            "against Baskonia on Thursday, the club has confirmed.",
            r7,
        )
        for name in ("NUNN, KENDRICK", "SLOUKAS, KOSTAS", "BONGA, ISAAC"):
            assert result[name][:2] == (OK, False), name
        for name in ("ROGKAVOPOULOS, NIKOS", "MANTZOUKAS, LEFTERIS"):
            assert result[name][:3] == (OUT, True, "game"), name

    def test_a_subjectless_part_continues_the_previous_subject(self, r7: RosterIndex) -> None:
        result = _scoped(
            "Isaac Bonga, who was ruled out for Round 5, also missed Round 6 but is back "
            "available for Round 7.",
            r7,
        )
        assert result["BONGA, ISAAC"][0] is OK

    @pytest.mark.parametrize(
        ("sentence", "name"),
        [
            (
                "Lamar Stevens available once again while Tonye Jekiri, Vanja Marinkovic and Derek "
                "Willis are ruled out for Partizan’s game against Crvena Zvezda on Thursday.",
                "STEVENS, LAMAR",
            ),
            (
                "Lonnie Walker is listed as available ahead of his possible debut while DaQuan "
                "Jeffries remains sidelined for Besiktas’s game at Efes on Thursday.",
                "WALKER, LONNIE",
            ),
        ],
    )
    def test_more_ways_of_saying_available(self, sentence: str, name: str, r7: RosterIndex) -> None:
        result = _scoped(sentence, r7)
        assert result[name][0] is OK
        # And the out on the other side of "while" stays with its own players.
        outs = {n for n, r in result.items() if r[0] is OUT}
        assert name not in outs and outs

    def test_a_past_absence_with_an_aside_does_not_rule_out(self, r7: RosterIndex) -> None:
        result = _scoped(
            "Stefan Miljenovic, who didn’t play in Round 4, was sidelined for Rounds 5 and 6 "
            "with an ankle injury.",
            r7,
        )
        assert "MILJENOVIC, STEFAN" not in result or not result["MILJENOVIC, STEFAN"][1]

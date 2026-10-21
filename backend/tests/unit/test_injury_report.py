"""The daily injury table: parsing, status and round columns, player matching.

Pages are built by :func:`page`, which reproduces the markup of the real
BasketNews table (team heading rows, then five cells per player) without
copying its content.
"""

from __future__ import annotations

import uuid

import pytest

from euroleague_ai.ingestion.injuries import RosterPlayer
from euroleague_ai.models.enums import AvailabilityStatus
from euroleague_ai.news.injury_report import (
    ReportFormatError,
    ReportRow,
    match_rows,
    parse_injury_report,
)

Row = tuple[str, str, str, str]  # name, status label, round, comment


def page(blocks: dict[str, list[Row]], *, modified: str = "2025-10-27T09:30:00+0300") -> str:
    status_ids = {
        "Ready": 1,
        "Expected": 2,
        "Game-time": 4,
        "Doubtful": 5,
        "Out": 6,
        "Uncertain": 7,
    }
    body = []
    for team, rows in blocks.items():
        slug = team.lower().replace(" ", "-")
        body.append(
            f'<tr class="injury_reports__team-row bg-light-subtle" id="{slug}">'
            f'<td class="injury_reports__team-cell fs-sm fw-bold border-0" colspan="5">'
            f'<div><img src="/logo.png" alt="{team}" /><a href="/teams/1-{slug}.html">{team}</a></div>'
            f"</td></tr>"
        )
        for name, status, rounds, comment in rows:
            body.append(
                '<tr style="height: 50px;">'
                '<td class="p-3">\n  PG\n</td>'
                f'<td class="p-3"><a href="/players/1-x.html">{name}</a></td>'
                f'<td class="p-3"><span class="player-status player-status__id-{status_ids.get(status, 9)}">'
                f"{status}</span></td>"
                f'<td class="p-3">\n {rounds}\n</td>'
                f'<td class="p-3"><a href="https://example.test/n.html">{comment}</a></td>'
                "</tr>"
            )
    return (
        '<html><head><script type="application/ld+json">{"datePublished":"2024-01-01T00:00:00+0300",'
        f'"dateModified":"{modified}"}}</script></head><body>'
        '<table class="table_02 table-bordered w-100"><thead><tr><th>P</th><th>Player</th>'
        "<th>Status</th><th>Round</th><th>Comments</th></tr></thead><tbody>"
        + "".join(body)
        + '</tbody></table><table id="grid_layout_body"><tr><td>x</td></tr></table></body></html>'
    )


BLOCKS: dict[str, list[Row]] = {
    "Partizan Mozzart Bet Belgrade": [
        ("Luka Jovanovic", "Out", "Round 5", "Knee"),
        ("Marko Petrovic", "Out", "Round 5-6", "Will miss two weeks"),
    ],
    "Real Madrid": [
        ("Pablo Garcia", "Doubtful", "Indefinitely", "Ankle &amp; knee"),
        ("Diego Martinez", "Ready", "Round 5", "Back in training"),
    ],
}


class TestParse:
    def test_rows_teams_and_cells(self) -> None:
        parsed = parse_injury_report(page(BLOCKS), min_rows=1, min_teams=1)
        assert [r.name for r in parsed.rows] == [
            "Luka Jovanovic",
            "Marko Petrovic",
            "Pablo Garcia",
            "Diego Martinez",
        ]
        garcia = parsed.rows[2]
        assert garcia.team == "Real Madrid"
        assert garcia.comment == "Ankle & knee"
        assert (
            parsed.modified_at is not None
            and parsed.modified_at.utcoffset().total_seconds() == 3 * 3600
        )

    @pytest.mark.parametrize(
        ("label", "status"),
        [
            ("Ready", AvailabilityStatus.AVAILABLE),
            ("Expected", AvailabilityStatus.PROBABLE),
            ("Game-time", AvailabilityStatus.QUESTIONABLE),
            ("Uncertain", AvailabilityStatus.QUESTIONABLE),
            ("Doubtful", AvailabilityStatus.DOUBTFUL),
            ("Out", AvailabilityStatus.OUT),
            ("Suspended", AvailabilityStatus.SUSPENDED),
            ("Something new", None),
        ],
    )
    def test_status_labels(self, label: str, status: AvailabilityStatus | None) -> None:
        assert ReportRow("T", "PG", "A B", label, "Round 7", "").status is status

    @pytest.mark.parametrize(
        ("rounds", "span"),
        [
            ("Round 7", (7, 7)),
            ("Round 7-9", (7, 9)),
            ("Round 4\u201312", (4, 12)),
            ("Indefinitely", None),
            ("Long-term", None),
            ("Out for season", None),
        ],
    )
    def test_round_column(self, rounds: str, span: tuple[int, int] | None) -> None:
        assert ReportRow("T", "PG", "A B", "Out", rounds, "").round_range == span

    def test_a_thin_page_is_refused(self) -> None:
        with pytest.raises(ReportFormatError):
            parse_injury_report(page(BLOCKS))  # 4 rows, 2 teams: below the defaults
        with pytest.raises(ReportFormatError):
            parse_injury_report("<html><body>Checking your browser</body></html>")


def _player(name: str, team: uuid.UUID, code: str = "XXX") -> RosterPlayer:
    return RosterPlayer(uuid.uuid4(), name, team, code, None)


class TestMatching:
    def test_name_variants_match_within_the_club(self) -> None:
        pan, oly = uuid.uuid4(), uuid.uuid4()
        roster = [
            _player("ROGKAVOPOULOS, NIKOLAOS", pan),
            _player("SLOUKAS, KOSTAS", pan),
            _player("NUNN, KENDRICK", pan),
            _player("DOZIER, PJ", oly),
            _player("WALKER IV, LONNIE", oly),
            _player("TUBELIS, AZUOLAS", oly),
        ]
        rows = [
            ReportRow("Panathinaikos AKTOR Athens", "SF", n, "Out", "Round 7", "")
            for n in ("Nikos Rogkavopoulos", "Kostas Sloukas", "Kendrick Nunn")
        ] + [
            ReportRow("Some Club", "PG", n, "Out", "Round 7", "")
            for n in ("P.J. Dozier", "Lonnie Walker IV", "Ąžuolas Tubelis")
        ]
        matched = {m.row.name: m.player and m.player.full_name for m in match_rows(rows, roster)}
        assert matched == {
            "Nikos Rogkavopoulos": "ROGKAVOPOULOS, NIKOLAOS",
            "Kostas Sloukas": "SLOUKAS, KOSTAS",
            "Kendrick Nunn": "NUNN, KENDRICK",
            "P.J. Dozier": "DOZIER, PJ",
            "Lonnie Walker IV": "WALKER IV, LONNIE",
            "Ąžuolas Tubelis": "TUBELIS, AZUOLAS",
        }

    def test_surname_only_matches_inside_the_block_club(self) -> None:
        efes, madrid = uuid.uuid4(), uuid.uuid4()
        roster = [
            _player("JAMES, MIKE", efes),
            _player("CORDINIER, ISAIA", efes),
            _player("JAMES, JUSTIN", madrid),
        ]
        rows = [
            ReportRow("Anadolu Efes Istanbul", "PG", "Isaia Cordinier", "Out", "Indefinitely", ""),
            ReportRow("Anadolu Efes Istanbul", "PG", "M. James", "Out", "Out for season", ""),
        ]
        matched = {m.row.name: m.player for m in match_rows(rows, roster)}
        assert matched["M. James"] is not None and matched["M. James"].team_id == efes

    def test_unrostered_players_are_left_unmatched(self) -> None:
        team = uuid.uuid4()
        roster = [_player("WILLIS, DEREK", team)]
        rows = [
            ReportRow("Partizan", "C", "Derek Willis", "Out", "Round 7", ""),
            ReportRow(
                "Partizan", "SG", "Mitar Bosnjakovic", "Out", "Indefinitely", "Not in roster"
            ),
        ]
        matched = {m.row.name: m.player for m in match_rows(rows, roster)}
        assert matched["Derek Willis"] is not None
        assert matched["Mitar Bosnjakovic"] is None

"""Position and positional competition.

Five guards cannot all play twenty minutes. Predicting each player in isolation
had no way to know that, so two players with identical recent form got identical
projections even when one was behind three better players at his position.

The one-hot flags are the boring part. What these tests protect is the pecking
order: ``same_position_rank`` and ``same_position_minutes_share`` are facts about
the squad around a player, invisible in his own box scores.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from euroleague_ai.features.position import POSITIONS, summarise_position


@dataclass
class FakeAppearance:
    """Stands in for SquadAppearance; only these three fields are read."""

    player_id: uuid.UUID
    seconds_played: int
    did_not_play: bool = False

    @property
    def minutes(self) -> float:
        return self.seconds_played / 60.0


@pytest.fixture
def rotation() -> tuple[list[uuid.UUID], uuid.UUID, dict, list[FakeAppearance]]:
    """Five guards on a real rotation curve, plus one centre.

    Guard minutes per game: 30, 24, 16, 8, 2 -- which is what a EuroLeague guard
    rotation actually looks like, and emphatically not five players at twenty.
    """
    guards = [uuid.uuid4() for _ in range(5)]
    centre = uuid.uuid4()
    positions = {guard: "GUARD" for guard in guards}
    positions[centre] = "CENTER"

    appearances: list[FakeAppearance] = []
    for _ in range(10):
        for guard, minutes in zip(guards, (30, 24, 16, 8, 2), strict=True):
            appearances.append(FakeAppearance(guard, minutes * 60))
        appearances.append(FakeAppearance(centre, 28 * 60))
    return guards, centre, positions, appearances


class TestPeckingOrder:
    """The observation the whole module exists for."""

    def test_the_lead_guard_ranks_first(self, rotation) -> None:
        guards, _, positions, appearances = rotation
        result = summarise_position(
            player_id=guards[0], positions=positions, appearances=appearances
        )
        assert result["same_position_rank"] == 1.0

    def test_the_fourth_guard_ranks_fourth(self, rotation) -> None:
        """The fact that stops four guards all projecting twenty minutes."""
        guards, _, positions, appearances = rotation
        result = summarise_position(
            player_id=guards[3], positions=positions, appearances=appearances
        )
        assert result["same_position_rank"] == 4.0

    def test_rank_is_also_normalised_by_group_size(self, rotation) -> None:
        """Third of three is a different situation from third of six."""
        guards, _, positions, appearances = rotation
        result = summarise_position(
            player_id=guards[3], positions=positions, appearances=appearances
        )
        assert result["same_position_rank_pct"] == pytest.approx(0.8)

    def test_minutes_share_separates_the_group(self, rotation) -> None:
        guards, _, positions, appearances = rotation
        shares = [
            summarise_position(
                player_id=guard, positions=positions, appearances=appearances
            )["same_position_minutes_share"]
            for guard in guards
        ]
        assert shares == sorted(shares, reverse=True)
        assert shares[0] == pytest.approx(30 / 80)

    def test_shares_sum_to_one_across_the_position_group(self, rotation) -> None:
        """The budget property: a position group's minutes are divided, not created."""
        guards, _, positions, appearances = rotation
        total = sum(
            summarise_position(
                player_id=guard, positions=positions, appearances=appearances
            )["same_position_minutes_share"]
            for guard in guards
        )
        assert total == pytest.approx(1.0)


class TestGroupMembership:
    def test_the_group_is_counted(self, rotation) -> None:
        guards, _, positions, appearances = rotation
        result = summarise_position(
            player_id=guards[0], positions=positions, appearances=appearances
        )
        assert result["same_position_count"] == 5.0

    def test_a_lone_centre_competes_with_nobody(self, rotation) -> None:
        _, centre, positions, appearances = rotation
        result = summarise_position(
            player_id=centre, positions=positions, appearances=appearances
        )
        assert result["same_position_count"] == 1.0
        assert result["same_position_minutes_share"] == pytest.approx(1.0)

    def test_a_registered_but_unused_player_is_still_competition(self, rotation) -> None:
        """Signing a sixth guard changes the picture before he plays a minute."""
        guards, _, positions, appearances = rotation
        newcomer = uuid.uuid4()
        crowded = {**positions, newcomer: "GUARD"}

        before = summarise_position(
            player_id=guards[0], positions=positions, appearances=appearances
        )
        after = summarise_position(
            player_id=guards[0], positions=crowded, appearances=appearances
        )
        assert after["same_position_count"] == before["same_position_count"] + 1

    def test_a_player_who_has_never_featured_ranks_last(self, rotation) -> None:
        guards, _, positions, appearances = rotation
        newcomer = uuid.uuid4()
        crowded = {**positions, newcomer: "GUARD"}
        result = summarise_position(
            player_id=newcomer, positions=crowded, appearances=appearances
        )
        assert result["same_position_rank"] == 6.0
        assert result["same_position_minutes_share"] == pytest.approx(0.0)


class TestDidNotPlay:
    def test_a_dnp_counts_as_zero_rather_than_being_dropped(self, rotation) -> None:
        """A fit player who did not play is genuinely behind the others.

        Dropping those rows would raise his average and flatter him.
        """
        guards, _, positions, appearances = rotation
        padded = appearances + [
            FakeAppearance(guards[4], 0, did_not_play=True) for _ in range(5)
        ]
        result = summarise_position(
            player_id=guards[4], positions=positions, appearances=padded
        )
        assert result["same_position_rank"] == 5.0


class TestPositionFlags:
    def test_the_flags_are_one_hot(self, rotation) -> None:
        guards, centre, positions, appearances = rotation
        guard = summarise_position(
            player_id=guards[0], positions=positions, appearances=appearances
        )
        big = summarise_position(
            player_id=centre, positions=positions, appearances=appearances
        )
        assert guard["position_guard"] == 1.0
        assert guard["position_center"] == 0.0
        assert big["position_center"] == 1.0
        assert big["position_guard"] == 0.0

    def test_only_the_three_positions_the_feed_publishes(self) -> None:
        """The upstream API has no PG/SG split, and this must not pretend to."""
        assert POSITIONS == ("GUARD", "FORWARD", "CENTER")


class TestUnknownPosition:
    """Unknown is not the same as "not a guard"."""

    def test_unknown_is_flagged_explicitly(self, rotation) -> None:
        _, _, positions, appearances = rotation
        result = summarise_position(
            player_id=uuid.uuid4(), positions=positions, appearances=appearances
        )
        assert result["position_unknown"] == 1.0
        assert result["position_guard"] == 0.0

    def test_no_position_means_no_competition_signal(self, rotation) -> None:
        """NaN, not zero. Zero would read as "no competition at all"."""
        _, _, positions, appearances = rotation
        result = summarise_position(
            player_id=uuid.uuid4(), positions=positions, appearances=appearances
        )
        assert result["same_position_rank"] is None
        assert result["same_position_minutes_share"] is None

    def test_the_key_set_never_varies(self, rotation) -> None:
        """A missing feature must be NaN to the model, not an absent column."""
        guards, _, positions, appearances = rotation
        known = summarise_position(
            player_id=guards[0], positions=positions, appearances=appearances
        )
        unknown = summarise_position(
            player_id=uuid.uuid4(), positions=positions, appearances=appearances
        )
        assert set(known) == set(unknown)

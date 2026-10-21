"""Which injury report, if any, applies to a player in a given game.

Pure resolution over reports already filtered to ``observed_at <= cutoff`` by
:mod:`euroleague_ai.repositories.availability`, so it can be tested without a
database and cannot see anything the cutoff hides.

**Scope.** A report is one of three shapes, and each expires differently:

* **For one game** (``game_id`` set): applies to that game only.
* **Until a date** (``expected_return`` set, no game): applies to every game
  before that date.
* **Open-ended** (neither): applies until the player is reported back, or until
  he actually plays again. A box score showing him on the floor after the report
  retires it, so a forgotten "out" can never sideline a player indefinitely.

Among the reports that still apply, the newest wins (by when the status changed,
then by when we recorded it), so "out" on Monday and "available" on Thursday
resolve to available.

**What the prediction does with it.** Only certainties change a number:
:data:`RULES_OUT` sets P(plays) to zero. "Questionable", "doubtful" and
"probable" are shown, and kept as history, but do not touch the model's
probability. There is no record yet of how often a EuroLeague "game-time
decision" plays, and an invented discount would be a guess presented as a
measurement. Once a season of reports has accumulated, those labels can become
calibrated features.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from euroleague_ai.models.enums import AvailabilityStatus, StatusSourceType

if TYPE_CHECKING:  # pragma: no cover - typing only
    import uuid
    from collections.abc import Sequence
    from datetime import datetime

    from euroleague_ai.repositories.availability import StatusReport

#: Statuses that mean he will not take the floor. These set P(plays) to 0.
RULES_OUT: frozenset[AvailabilityStatus] = frozenset(
    {AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED}
)

#: A report read from news rules a player out only at or above this
#: confidence. Lower-confidence "out" readings are shown and kept, but the
#: model's own probability stands. Manual reports are taken as given.
NEWS_RULE_OUT_CONFIDENCE = 0.8

#: An open-ended news reading ("Fournier is out") lapses after this long unless
#: a newer article repeats it. Each round's injury report restates the long
#: absences, so a real long-term injury keeps being renewed; a one-off mention
#: in an old preview does not linger.
NEWS_OPEN_MAX_AGE = timedelta(days=8)

#: Statuses worth flagging to a reader, though they change no number.
FLAGGED: frozenset[AvailabilityStatus] = frozenset(
    {AvailabilityStatus.DOUBTFUL, AvailabilityStatus.QUESTIONABLE, AvailabilityStatus.PROBABLE}
)


@dataclass(frozen=True, slots=True)
class ResolvedStatus:
    """The report that applies, and how."""

    report: StatusReport
    scope: str  # "game" | "until" | "open"

    @property
    def status(self) -> AvailabilityStatus:
        return self.report.status

    @property
    def rules_out(self) -> bool:
        if self.report.status not in RULES_OUT:
            return False
        if self.report.source is StatusSourceType.NEWS:
            return (self.report.confidence or 0.0) >= NEWS_RULE_OUT_CONFIDENCE
        return True

    def as_dict(self) -> dict[str, object]:
        r = self.report
        return {
            "status": str(r.status),
            "reason": r.reason,
            "scope": self.scope,
            "until": r.expected_return.date().isoformat() if r.expected_return else None,
            "source": str(r.source),
            "confidence": r.confidence,
            "applied": self.rules_out,
            "reported_at": r.observed_at.isoformat(),
            "report_id": str(r.id),
        }


def resolve_status(
    reports: Sequence[StatusReport],
    *,
    game_id: uuid.UUID,
    tip_off: datetime,
    last_played: datetime | None,
) -> ResolvedStatus | None:
    """The newest report that still applies to this game, or ``None``.

    Args:
        reports: This player's reports known at the cutoff, any order.
        last_played: Tip-off of his latest game with minutes, as known at the
            cutoff. Retires open-ended and dated reports made before it.
    """
    applicable: list[ResolvedStatus] = []
    for report in reports:
        if report.game_id is not None:
            if report.game_id == game_id:
                applicable.append(ResolvedStatus(report, "game"))
            continue
        # Not scoped to a game: retired once he has played since the report.
        if last_played is not None and last_played > report.event_timestamp:
            continue
        if report.expected_return is not None:
            if report.expected_return > tip_off:
                applicable.append(ResolvedStatus(report, "until"))
            continue
        if (
            report.source is StatusSourceType.NEWS
            and tip_off - report.event_timestamp > NEWS_OPEN_MAX_AGE
        ):
            continue
        applicable.append(ResolvedStatus(report, "open"))

    if not applicable:
        return None
    newest = max(
        applicable,
        key=lambda item: (item.report.event_timestamp, item.report.observed_at),
    )
    # A report that the player is fine clears whatever came before it, and is
    # not itself worth showing.
    if newest.report.status in (AvailabilityStatus.AVAILABLE, AvailabilityStatus.UNKNOWN):
        return None
    return newest

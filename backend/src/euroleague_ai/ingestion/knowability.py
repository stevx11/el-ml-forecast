"""When a fact first became knowable.

**The problem this solves.**

``observed_at`` means "when this system learned the fact", and every feature
query filters on it. That is correct for data ingested live. It is catastrophic
for data backfilled from history.

Backfilling ten seasons today stamps every row with today's timestamp. A feature
built for a 2018 game with a 2018 cutoff then asks for rows with
``observed_at <= 2018``, finds none, and returns null for everything. The
leakage machinery works perfectly and the dataset is empty -- which is exactly
what happened here: 66,000 snapshots, every feature null, every baseline
correctly refusing to predict.

**The fix.**

For a completed historical game we did not observe live, the honest
reconstruction is *when the fact could first have been known by anyone*, not when
our ingest happened to run. A box score becomes public when the game ends. A
fixture is published weeks ahead. Using those derived times simulates having been
running at the time, which is precisely what a backtest needs.

    observed_at = min(ingest_time, derived_knowable_time)

The ``min`` matters in both directions:

* Backfilled history -- derived time is in the past, so it wins. We reconstruct
  what a running system would have recorded.
* Live ingestion -- for a fixture announced months ahead, the derived time
  (tip-off minus a week) is still in the future, so the actual ingest time wins.
  We never claim to have known something before we did.

**This is an approximation, and it is documented as one.** It assumes box scores
are public shortly after the final buzzer and schedules are published in advance,
both true in practice. It cannot reconstruct facts nobody recorded -- pre-game
injury reports remain unavailable for history, which is why availability uses an
appearance-based proxy instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta

#: How long after tip-off a box score is treated as public. A EuroLeague game
#: plus overtime and stat confirmation comfortably fits in three hours.
BOX_SCORE_AVAILABLE_AFTER = timedelta(hours=3)

#: How far ahead of tip-off a fixture is treated as known. Schedules are
#: published months in advance; a week is deliberately conservative, since
#: claiming a fixture was known earlier than it was would let schedule features
#: see too far ahead.
FIXTURE_KNOWN_BEFORE = timedelta(days=7)


def box_score_observed_at(tip_off_utc: datetime, ingested_at: datetime) -> datetime:
    """When a box score became knowable.

    >>> from datetime import UTC, datetime
    >>> tip = datetime(2018, 1, 10, 18, 0, tzinfo=UTC)
    >>> ingest = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
    >>> box_score_observed_at(tip, ingest).isoformat()
    '2018-01-10T21:00:00+00:00'

    A game ingested while still in progress keeps the real ingest time, because
    we genuinely did not know the result yet:

    >>> live = datetime(2018, 1, 10, 19, 0, tzinfo=UTC)
    >>> box_score_observed_at(tip, live) == live
    True
    """
    return min(ingested_at, tip_off_utc + BOX_SCORE_AVAILABLE_AFTER)


def fixture_observed_at(tip_off_utc: datetime, ingested_at: datetime) -> datetime:
    """When a fixture became knowable.

    >>> from datetime import UTC, datetime
    >>> tip = datetime(2018, 1, 10, 18, 0, tzinfo=UTC)
    >>> ingest = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
    >>> fixture_observed_at(tip, ingest).isoformat()
    '2018-01-03T18:00:00+00:00'

    A fixture ingested months before it is played keeps the ingest time -- we
    must never claim to have known it earlier than we did:

    >>> early = datetime(2017, 9, 1, tzinfo=UTC)
    >>> fixture_observed_at(tip, early) == early
    True
    """
    return min(ingested_at, tip_off_utc - FIXTURE_KNOWN_BEFORE)


def stint_observed_at(effective_from: datetime, ingested_at: datetime) -> datetime:
    """When a roster registration became knowable.

    The provider supplies the registration date directly, so that date is the
    knowable time -- a signing is public when it is announced.
    """
    return min(ingested_at, effective_from)

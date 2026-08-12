#!/usr/bin/env python3
"""Recapture test fixtures from the live EuroLeague API.

Run this when the upstream response shape changes. It writes the raw payloads
unmodified so the fixtures keep reflecting the API's real quirks -- the
misspelled ``foulsCommited``, float counting stats, padded person codes -- rather
than a cleaned-up version that the mapping layer would never actually meet.

    python scripts/capture_fixtures.py --season E2025 --games 1 2 3
    python scripts/capture_fixtures.py --season E2025 --limit 5 --output /tmp/fx

Captured files are large. Trim them by hand to a handful of entities before
committing, but never edit a *value* to make a test pass: if a test fails after
recapture, the mapping is wrong, not the fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api-live.euroleague.net/v2"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "backend/tests/fixtures/euroleague"
USER_AGENT = "euroleague-ai-fixture-capture/0.1"


async def _get(client: httpx.AsyncClient, path: str, **params: Any) -> Any:
    response = await client.get(path, params=params or None)
    response.raise_for_status()
    return response.json()


async def capture(
    *, season: str, competition: str, base_url: str, output: Path, games: list[int], limit: int
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "game_stats").mkdir(exist_ok=True)

    prefix = f"/competitions/{competition}/seasons/{season}"

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(30.0),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        clubs = await _get(client, f"{prefix}/clubs", limit=100)
        _write(output / "clubs.json", clubs)

        people = await _get(client, f"{prefix}/people", personType="J", limit=limit)
        _write(output / "people.json", people)

        schedule = await _get(client, f"{prefix}/games", limit=limit)
        _write(output / "games.json", schedule)

        for code in games:
            try:
                stats = await _get(client, f"{prefix}/games/{code}/stats")
            except httpx.HTTPStatusError as exc:
                print(f"  game {code}: skipped ({exc.response.status_code})", file=sys.stderr)
                continue
            _write(output / "game_stats" / f"{code}.json", stats)
            # Be polite to a free public API.
            await asyncio.sleep(0.25)


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="E2025", help="Season code, e.g. E2025.")
    parser.add_argument("--competition", default="E", help="Competition code.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=20, help="Rows per paginated endpoint.")
    args = parser.parse_args()

    print(f"Capturing {args.season} from {args.base_url} -> {args.output}")
    asyncio.run(
        capture(
            season=args.season,
            competition=args.competition,
            base_url=args.base_url,
            output=args.output,
            games=args.games,
            limit=args.limit,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

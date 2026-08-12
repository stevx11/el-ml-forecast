# Test fixtures

`euroleague/` mirrors the exact response shapes of the public EuroLeague v2 API,
captured from live endpoints:

| File | Endpoint |
|---|---|
| `clubs.json` | `/competitions/E/seasons/E2025/clubs` |
| `people.json` | `/competitions/E/seasons/E2025/people?personType=J` |
| `games.json` | `/competitions/E/seasons/E2025/games` |
| `game_stats/{code}.json` | `/competitions/E/seasons/E2025/games/{code}/stats` |

They are trimmed to a handful of entities so the suite stays fast and readable,
but every upstream quirk that the mapping layer has to handle is preserved
deliberately:

- `foulsCommited` — misspelled upstream, corrected at our boundary
- counting stats typed as floats (`12.0`) rather than integers
- `timePlayed` in seconds, not a `MM:SS` clock string
- space-padded person codes (`"P007200   "`) alongside bare ones (`"006590"`)
- naive ISO timestamps in fields documented as UTC
- `""` used where the API means "absent"

Regenerate from the live API with:

    python scripts/capture_fixtures.py --season E2025 --games 1 2 3

Do not hand-edit a captured file to make a test pass. If the upstream shape
changed, recapture and fix the mapping — a fixture edited into agreement with a
broken mapper is a test that proves nothing.

# EuroLeague AI

[![CI](https://github.com/stevx11/el-ml-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/stevx11/el-ml-predictor/actions/workflows/ci.yml)

A pre-game basketball intelligence and player-performance prediction platform for
EuroLeague.

The system ingests games, statistics, rosters and news continuously, stores **what
was known and when**, converts that history into leakage-free feature snapshots,
and predicts the statistical line each player is likely to produce in their next
game — with an explanation, and with predictions that can be honestly backtested.

```
Player: Marko Petrovic          Projected minutes: 31.4
Team: Partizan                  Points 18.7 · Rebounds 5.9 · Assists 6.2 · 3PM 2.1
Opponent: Real Madrid           P(15+) 72%  P(20+) 43%  P(25+) 17%

Why?  + Increased projected minutes    - Tough opponent defense
      + Usage up (teammate injured)    - Limited rest
```

> **Status: Phase 0 — foundation.** The schema, ingestion pipeline, live
> EuroLeague adapter, read API, and leakage-prevention machinery are built and
> tested. **No models are trained yet**, so the projection above is the target,
> not current output. See [Roadmap](#roadmap).

---

## Why this is not a stats CRUD app

Two requirements drive every design decision:

1. **A prediction is only meaningful relative to an instant.** A projection made
   48 hours before tip-off is a different prediction from one made after the
   starting lineup is announced. The system must be able to answer *"what did we
   know at 17:00?"*
2. **A prediction we cannot backtest is a prediction we cannot trust.** If today's
   model cannot be replayed honestly against the past, there is no way to know
   whether it works.

Both reduce to one thing: **the database records the history of knowledge, not
just the current state of the world.** Nothing is overwritten. Availability
changes, roster moves, and lineup announcements are stored as timestamped events
and intervals, and every feature query is filtered by *when we learned each
fact* — not when it happened.

That distinction is the whole game. A report published at 19:00 describing a
10:00 injury was not knowable at 17:00. Filtering on the event time looks correct
and leaks anyway.

---

## Quick start

Requires Docker and Docker Compose. Nothing else.

**Windows (PowerShell)** — `make` is not available, so use the bundled script:

```powershell
Copy-Item .env.example .env

.\make.ps1 up                        # build, start postgres + redis + api, migrate
.\make.ps1 ingest -Season E2025      # pull a real season from the EuroLeague API
```

If PowerShell blocks the script, allow local scripts for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**macOS / Linux:**

```bash
cp .env.example .env

make up                              # build, start postgres + redis + api, migrate
make ingest SEASON=E2025             # pull a real season from the EuroLeague API
```

Then, on either platform:

- API docs — <http://localhost:8000/docs>
- Health — <http://localhost:8000/health/ready>
- Games — <http://localhost:8000/api/v1/games?limit=10>

**Raw Docker commands**, if you would rather not use either wrapper:

```bash
docker compose up -d --build
docker compose exec backend alembic -c backend/alembic.ini upgrade head
docker compose exec backend euroleague-ai ingest all --season E2025
```

`.\make.ps1 help` or `make help` lists every target. The two wrappers expose the
same targets, so the rest of this README applies to both — read `make x` as
`.\make.ps1 x` on Windows.

---

## Architecture

A **modular monolith**: one deployable, clear internal boundaries, no
microservices until something actually forces the issue.

```
EuroLeague API ──▶ providers/ ──▶ ingestion/ ──▶ POSTGRES ──▶ features/ ──▶ ml/
  news feeds  ──▶   news/                      (bitemporal)      │          │
                                                   │             ▼          ▼
                                                   └────────▶ api/ ──▶ Angular (later)
```

| Layer | Responsibility |
|---|---|
| `providers/` | Adapters to external sources. A `BasketballDataProvider` protocol plus a live EuroLeague adapter and an offline fixture adapter. Domain code never sees provider JSON. |
| `ingestion/` | Identity resolution and temporal upserts. Decides whether a fact is an upsert, an interval close, or an append. |
| `models/` | SQLAlchemy ORM. The bitemporal schema is the source of truth. |
| `repositories/` | All SQL. Split into `catalog` (current state, for the API) and `history` (**`as_of` mandatory**, for features). |
| `features/` | Feature computation. The leakage boundary. |
| `ml/` | Baselines, training, evaluation, inference. |
| `api/` | FastAPI. Only endpoints backed by functionality that exists. |

Full detail: [`docs/architecture.md`](docs/architecture.md) ·
[`docs/data-model.md`](docs/data-model.md) ·
[`docs/ml-roadmap.md`](docs/ml-roadmap.md).

### Project structure

```
el-ml-predictor/
├── backend/
│   ├── src/euroleague_ai/
│   │   ├── api/          HTTP layer (FastAPI routers, dependencies)
│   │   ├── core/         config, logging, clock, errors
│   │   ├── db/           engine, session, declarative base
│   │   ├── models/       SQLAlchemy ORM — the bitemporal schema
│   │   ├── schemas/      Pydantic request/response models
│   │   ├── repositories/ all SQL; catalog (now) vs history (as_of)
│   │   ├── providers/    BasketballDataProvider protocol + adapters
│   │   ├── ingestion/    identity resolution, temporal upserts, jobs
│   │   ├── news/         collectors (+ planned event extraction)
│   │   ├── features/     as_of feature builders and snapshots
│   │   ├── ml/           baselines, training, evaluation  (phase 2+)
│   │   ├── main.py       app factory
│   │   └── cli.py        operational entrypoint
│   ├── migrations/       Alembic
│   └── tests/            unit · integration · fixtures
├── data/                 raw / processed / features / models  (gitignored)
├── docker/               Dockerfile + postgres init
├── docs/                 architecture · data-model · ml-roadmap
├── scripts/              fixture capture
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

**Why one package instead of top-level `ml/` and `news/` trees.** Feature
engineering, news extraction and prediction all need the same ORM models, the
same `as_of` clock, and the same identity resolution. Separate top-level trees
would mean duplicated code or `sys.path` manipulation, and three tool configs
instead of one. The boundaries are preserved by a strict layering rule
(`api → services → {ingestion, features, ml} → repositories → models → db → core`)
rather than by directory distance.

---

## How leakage is prevented

The failure mode that produces excellent offline metrics and a useless product.
Four defences, deliberately redundant:

**1. Schema.** There is no "current status" column anywhere for a feature to
accidentally read. Current state is always derived from timestamped history.

**2. API shape.** Every feature-relevant repository method takes `as_of` as a
mandatory keyword argument. No default, anywhere:

```python
async def build_player_form_features(
    session, *, player_id, team_id, as_of: datetime,  # mandatory
) -> dict[str, float | None]:
```

**3. One choke point.** Repositories apply the filter; feature builders never
write SQL. The leakage-critical predicate lives in a handful of reviewable
functions instead of being scattered across dozens of feature definitions:

```sql
WHERE observed_at <= :as_of      -- what we had actually recorded
  AND tip_off_utc  <  :as_of     -- and the game had actually happened
```

**4. Tests.** `backend/tests/integration/test_no_leakage.py` seeds a world where
post-cutoff rows carry deliberately extreme values, builds snapshots at a cutoff,
and asserts the extreme values never appear. It covers both future games *and*
the subtle case — a game that tipped off before the cutoff but was recorded
after it, which passes an event-time filter and leaks anyway.

Missing data is `None`, never `0.0`. A rookie's ten-game average is *unknown*;
substituting zero teaches the model that players without history score nothing.

---

## Data sources

**EuroLeague** — the public, unauthenticated JSON API at
`api-live.euroleague.net/v2`, the same endpoints that serve the official website.
The adapter identifies itself with a configurable User-Agent, rate-limits itself,
bounds concurrency, retries only transient failures, and caches in Redis.
**No HTML scraping happens anywhere in this codebase.** For commercial use,
obtain a licensed feed and write an adapter — the protocol makes that a drop-in
change.

Endpoints used:

| Data | Endpoint |
|---|---|
| Clubs | `/competitions/{c}/seasons/{s}/clubs` |
| Players + roster intervals | `/competitions/{c}/seasons/{s}/people?personType=J` |
| Schedule | `/competitions/{c}/seasons/{s}/games` |
| Box scores | `/competitions/{c}/seasons/{s}/games/{code}/stats` |

**News** — no public EuroLeague news feed was reachable when this was built, so
**no news source ships enabled**. What exists is a `NewsCollector` protocol and a
generic RSS/Atom collector that works against any feed you configure via
`NEWS_FEED_URLS`. Structured event extraction (turning *"his minutes will be
monitored tonight"* into `{"event_type": "MINUTES_LIMIT", "confidence": 0.91}`)
is designed and its tables exist, but is not implemented — see phase 5.

**Domestic leagues** (ABA, ACB, …) need no schema change: different `league_id`,
same tables. They matter because EuroLeague players play domestic games in
between, which changes fatigue, form and role. Blocked on a data source, not on
architecture.

---

## Development

### Local setup without Docker

Requires Python 3.12+ and a running PostgreSQL 16 and Redis.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # point DATABASE_URL at localhost
alembic -c backend/alembic.ini upgrade head
uvicorn euroleague_ai.main:app --reload
```

### Commands

```bash
euroleague-ai ingest all     --season E2025      # everything, in dependency order
euroleague-ai ingest teams   --season E2025
euroleague-ai ingest players --season E2025
euroleague-ai ingest games   --season E2025 --round 12
euroleague-ai ingest stats   --season E2025 --game-code 1
euroleague-ai features build --season E2025 --game-code 5
euroleague-ai news collect
```

Ingestion is **idempotent** — re-running never duplicates rows — so a nightly
cron and an ad-hoc backfill are the same command. Every run is recorded in
`ingestion_runs` with row counts and errors.

### Migrations

```bash
make migrate                              # alembic upgrade head
make revision M="add lineup confidence"   # autogenerate
```

Read every autogenerated revision before applying it. Alembic cannot infer
intent: a renamed column is indistinguishable from a drop plus an add.

### Tests

```bash
make test                                    # everything
docker compose exec backend pytest backend/tests/unit          # no database needed
docker compose exec backend pytest -m integration              # database-backed
docker compose exec backend pytest backend/tests/integration/test_no_leakage.py -v
```

| Suite | Proves | Needs a DB? |
|---|---|---|
| `tests/unit` | Parsing, mapping, config, clock, feature maths | No |
| `tests/integration/test_database.py` | Connectivity, constraints, temporal invariants | Yes |
| `tests/integration/test_ingestion.py` | End-to-end ingest, idempotency, identity resolution | Yes |
| `tests/integration/test_no_leakage.py` | **Features never see the future** | Yes |
| `tests/integration/test_api.py` | App starts, routes respond, DB dependency resolves | Yes |
| `tests/integration/test_migrations.py` | Migration and models have not drifted | Yes |

Database tests run against the real PostgreSQL created by Compose (never SQLite —
the schema uses Postgres-specific types), each inside a transaction that is rolled
back. They **skip** cleanly when `TEST_DATABASE_URL` is unset, so `pytest` works
on a laptop with no Docker running. The schema for the test suite is built by
running the actual migration, so a model change that never got a migration fails
the suite rather than production.

### Lint and types

```bash
make lint        # ruff
make typecheck   # mypy --strict
make check       # lint + typecheck + test — what CI runs
```

### CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

| Job | What it proves | Blocking |
|---|---|---|
| `tests` | Full suite against a real PostgreSQL 16 service | Yes |
| `migrations round-trip` | `upgrade head` → `downgrade base` → `upgrade head` | Yes |
| `lint and types` | ruff and mypy | **Not yet** |

Two details worth knowing.

The test job **fails if the integration tests skip**. They skip rather than fail
when `TEST_DATABASE_URL` is unset, so the suite still runs on a laptop with no
Docker — but in CI that safety valve would hide every leakage guard behind a
green tick. A step asks `conftest` the same question it asks itself, and fails if
the answer is "no database".

The quality job is `continue-on-error` because ruff and mypy have never been run
against this codebase and will report real findings on first contact. Merging
into a permanently red CI teaches people to ignore it. Once those findings are
cleared, drop the flag and make it blocking.

---

## Environment variables

Every variable is documented in [`.env.example`](.env.example) and read exactly
once, by `euroleague_ai.core.config.Settings`. There are no other reads of
`os.environ`, and no secrets in the repository.

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | — | **Must** use `postgresql+asyncpg://`. A sync DSN fails at startup, not at first query. |
| `TEST_DATABASE_URL` | — | Unset ⇒ database tests skip rather than fail. |
| `REDIS_URL` | `redis://redis:6379/0` | Cache now, task broker later. |
| `DATA_PROVIDER` | `euroleague` | `euroleague` (live) or `fixture` (offline, deterministic). |
| `PREDICTION_LEAD_MINUTES` | `60` | Cutoff before tip-off. Training and serving use the same value on purpose. |
| `EUROLEAGUE_REQUEST_RATE_PER_SECOND` | `5` | Politeness towards a free public API. |
| `NEWS_FEED_URLS` | *(empty)* | Comma-separated. No feeds ship enabled. |
| `LOG_FORMAT` | `console` | `json` in production. |

---

## Roadmap

| Phase | Deliverable | Gate before moving on |
|---|---|---|
| **0 ✅** | Schema, ingestion, provider adapters, read API, leakage tests, Docker | Migrations apply; a full season ingests; leakage tests pass |
| **1** | Full `as_of` feature catalogue — role, team, opponent, availability | Snapshots reproducible; leakage tests per feature family |
| **2** | Baselines + walk-forward backtest harness | Baseline MAE published for every target |
| **3** | LightGBM per target, minutes first | **Beats the best baseline** on walk-forward validation |
| **4** | Minutes → usage → opportunity → efficiency chain; Monte Carlo distributions | Calibration within tolerance |
| **5** | News collection, LLM event extraction, entity resolution | Measurable lift over phase 3 |
| **6** | Domestic leagues (ABA, ACB) for fatigue and role signals | Measurable lift |
| **7** | Prediction explanations, then the Angular frontend | — |

Two rules govern this list. **Minutes are predicted first**, because every other
box-score target is conditional on playing time. And **a model that does not beat
a meaningful baseline is not a model** — the bar is a minutes-scaled rate
projection, which is stronger than it sounds, and no phase advances without
clearing it.

Modelling detail: [`docs/ml-roadmap.md`](docs/ml-roadmap.md).

---

## License

See [LICENSE](LICENSE).

The EuroLeague API is not affiliated with this project. You are responsible for
complying with the terms of any data source you configure.

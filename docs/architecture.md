# Architecture

EuroLeague AI is a **pre-game basketball intelligence and player-performance prediction
system**. It ingests basketball data continuously, stores *what was known and when*,
converts that history into leakage-free feature snapshots, and predicts the statistical
line each player is likely to produce in their next game.

This document describes the target architecture and the state of the current
implementation. Anything marked **(planned)** is designed for but not yet built.

---

## 1. Product goal

For every upcoming game, answer:

> *Given everything that was known before tip-off, what is the most likely statistical
> performance of every relevant player?*

Two properties make this hard, and both drive the architecture:

1. **The answer changes over time.** A prediction made 48 hours before tip-off is a
   different (and worse) prediction than one made 30 minutes after the starting lineup
   is announced. The system must be able to make a prediction *as of any timestamp*.
2. **The answer must be historically verifiable.** Any prediction we make today must be
   reproducible against the past. If we cannot backtest, we cannot know whether the
   model is any good.

Both properties reduce to a single requirement: **the database must record the history of
knowledge, not just the current state of the world.**

---

## 2. Design principles

| Principle | Consequence |
|---|---|
| **Time-awareness is not a feature, it is the schema** | Mutable facts (availability, roster membership, lineups) are stored as append-only event or interval tables, never as overwritten columns. |
| **Leakage is a correctness bug, not a modelling detail** | Feature computation takes an explicit `as_of` timestamp and is physically incapable of reading rows stamped later. Enforced by tests. |
| **League-independent core** | EuroLeague is the first tenant, not the schema. Every competition-scoped row carries a `league_id` / `season_id`. |
| **Providers are replaceable** | All external data enters through a `BasketballDataProvider` protocol. Domain code never sees provider JSON. |
| **Modular monolith** | One deployable, clear internal module boundaries, no microservices until an actual scaling or ownership reason appears. |
| **Simplest thing that stays extensible** | LightGBM before neural networks. A CLI before Celery. Postgres before a feature store. |
| **The prediction engine matters more than the UI** | No frontend until the predictions are worth looking at. |

---

## 3. System context

```
     ┌──────────────────────────────────────────────────────────┐
     │                     EXTERNAL WORLD                        │
     │                                                           │
     │  EuroLeague API   Domestic leagues   News sources         │
     │  (games, stats,   (ABA/ACB/…)        (RSS/press/          │
     │   clubs, people)   (planned)          licensed feeds)     │
     └────────┬──────────────┬───────────────────┬───────────────┘
              │              │                   │
              ▼              ▼                   ▼
     ┌──────────────────────────────────────────────────────────┐
     │              PROVIDER LAYER (adapters)                    │
     │   BasketballDataProvider  │  NewsCollector                │
     │   ─ EuroLeagueProvider    │  ─ RssNewsCollector           │
     │   ─ FixtureProvider       │  ─ (licensed adapters)        │
     └────────────────────────┬─────────────────────────────────┘
                              │  provider-neutral DTOs
                              ▼
     ┌──────────────────────────────────────────────────────────┐
     │                   INGESTION LAYER                         │
     │   identity resolution → temporal upsert → raw snapshot    │
     └────────────────────────┬─────────────────────────────────┘
                              ▼
     ┌──────────────────────────────────────────────────────────┐
     │            POSTGRES — bitemporal domain store             │
     │   competition · people · games · stats · availability ·   │
     │   news · features · predictions · model registry          │
     └────────┬─────────────────────────────────┬───────────────┘
              │                                 │
              ▼                                 ▼
     ┌────────────────────┐          ┌──────────────────────────┐
     │  FEATURE LAYER     │          │      READ API            │
     │  as_of snapshots   │          │  FastAPI + Pydantic      │
     └─────────┬──────────┘          └──────────┬───────────────┘
               ▼                                 ▼
     ┌────────────────────┐          ┌──────────────────────────┐
     │  ML LAYER          │          │  Angular frontend        │
     │  baselines → GBDT  │─────────▶│  (planned)               │
     │  → distributions   │          └──────────────────────────┘
     └────────────────────┘
```

---

## 4. Module map

All importable code lives in a single package, `backend/src/euroleague_ai`. Modules are
layered; **imports may only point downward**.

```
euroleague_ai/
├── api/            HTTP layer. FastAPI routers, dependency wiring. Knows schemas + repositories.
├── schemas/        Pydantic response/request models. Pure data, no I/O.
├── services/       Use cases that coordinate repositories and providers.
├── ingestion/      Provider output → database. Identity resolution, temporal upserts.
├── providers/      Adapters to the outside world. Protocol + implementations.
├── news/           News collection and (planned) event extraction.
├── features/       as_of feature computation. The leakage boundary lives here.
├── ml/             Baselines, datasets, training, evaluation, inference.
├── repositories/   Data access. All SQL lives here. Returns ORM objects or DTOs.
├── models/         SQLAlchemy ORM. The schema is the source of truth.
├── db/             Engine, session, declarative base, custom types.
├── core/           Config, logging, clock, errors. Imported by everything, imports nothing.
└── cli.py          Operational entrypoint for ingestion and maintenance jobs.
```

**Layering rule:**
`api → services → {ingestion, features, ml} → repositories → models → db → core`

`providers/` and `news/` sit beside `ingestion` and may only import `core` and their own
DTOs. This keeps adapters testable without a database.

### Why one package instead of top-level `backend/`, `ml/`, `news/`

The original sketch had `ml/` and `news/` as sibling top-level directories. Feature
engineering, news extraction, and prediction all need the same ORM models, the same
`as_of` clock, and the same identity resolution. Splitting them into separate top-level
trees means either duplicated code or `sys.path` manipulation, and it makes a single
`mypy`/`ruff`/`pytest` configuration awkward.

They are therefore **subpackages of one installable distribution**. The boundaries the
sketch wanted are preserved by the layering rule above and enforced by review, not by
directory distance. Root-level `data/` is retained for on-disk artifacts (raw snapshots,
processed datasets, serialized models), which genuinely are not code.

---

## 5. Runtime topology

Development runs on Docker Compose with three services:

| Service | Purpose |
|---|---|
| `postgres` | Domain store. Postgres 16. The single source of truth. |
| `redis` | Cache and, later, the broker for scheduled jobs. Currently used for provider response caching and rate limiting. |
| `backend` | FastAPI app (`uvicorn`) plus the `euroleague-ai` CLI for ingestion jobs. |

There is deliberately **no worker service yet**. Ingestion runs as a CLI command
(`euroleague-ai ingest ...`), which is enough for a nightly cron or a manual backfill.
A Celery worker is added when — and only when — we need concurrent, retryable,
schedule-driven jobs. The job functions are already written as plain callables taking a
session, so promoting them to Celery tasks is a wrapper, not a rewrite.

---

## 6. Data flow

### 6.1 Ingestion

```
provider.fetch_games(season)
        │  returns provider-neutral DTOs (Pydantic)
        ▼
identity resolution
        │  external_ref (provider, entity_type, external_id) → internal UUID
        ▼
temporal upsert
        │  immutable facts  → INSERT … ON CONFLICT DO UPDATE
        │  mutable facts    → close current interval, open a new one
        │  observations     → append-only event row
        ▼
Postgres
```

Every write records `observed_at` — the moment *our system* learned the fact — separately
from `event_timestamp` / `effective_from` — the moment the fact became true in the world.
This is the bitemporal split, and it is what makes "what did we know at 17:00?"
answerable. See [`data-model.md`](./data-model.md) for the exact columns.

### 6.2 Feature computation

```
(player, game, as_of) ──▶ FeatureBuilder
                              │
                              ├─ repository queries, every one filtered by as_of
                              ├─ deterministic ordering
                              └─ named feature values + provenance
                              ▼
                     PlayerFeatureSnapshot (persisted, immutable)
```

A snapshot is written once and never updated. It is the exact input a model saw, which
means a prediction can be re-explained months later even if the underlying tables have
since been corrected.

### 6.3 Prediction

```
FeatureSnapshot + ModelVersion ──▶ inference ──▶ Prediction (+ PredictionOutcome later)
```

`Prediction` stores `model_version_id`, `prediction_timestamp`, and
`feature_snapshot_id`. Once the game finishes, an evaluation job joins actual results
onto predictions and writes `PredictionOutcome`, making model accuracy queryable without
recomputation.

---

## 7. Time-awareness

Three distinct timestamps appear throughout the schema. Confusing them is the single
most likely source of subtle bugs, so they are named consistently:

| Column | Meaning | Example |
|---|---|---|
| `event_timestamp` / `effective_from` | When the fact became true **in the world** | Player was ruled out at 10:00 |
| `observed_at` | When **we** recorded it | Our ingest ran at 10:42 and saw it |
| `created_at` / `updated_at` | Row bookkeeping | Audit only, never used in features |

**Features must filter on `observed_at`, not `event_timestamp`.** If a report published at
19:00 says a player was injured at 10:00, a model predicting at 17:00 could not have known
it. Filtering on `event_timestamp` would silently leak. This rule is encoded in the
repository layer and asserted by tests.

### Mutable state as intervals

Facts that change but are not events — roster membership, jersey number, coach — are
stored as **closed-open intervals** `[effective_from, effective_until)` with `NULL`
meaning "still open". Correcting history means closing the old row and inserting a new
one, never `UPDATE`ing the old values away.

### Observations as append-only events

Facts that are inherently point-in-time — availability changes, news items, lineup
announcements — are **append-only**. The `10:00 OUT → 14:00 QUESTIONABLE → 18:30 AVAILABLE`
sequence produces three rows. "Status as of T" is a query
(`ORDER BY event_timestamp DESC LIMIT 1 WHERE observed_at <= T`), not a column.

---

## 8. Data leakage prevention

Leakage is prevented at four levels, deliberately redundant.

**1. Schema.** There is no "current status" column anywhere that a feature could
accidentally read. Current state is always derived from timestamped history.

**2. API shape.** Every feature-relevant repository method requires `as_of: datetime` as a
mandatory keyword argument. You cannot call it without deciding what the model is allowed
to see. There is no default.

**3. A single filtering choke point.** Repositories apply the `as_of` predicate; feature
builders never write raw SQL. This means the leakage-critical logic lives in a small,
reviewable number of functions rather than being scattered across dozens of feature
definitions.

**4. Tests.** `backend/tests/integration/test_no_leakage.py` seeds a fixture world in
which post-cutoff rows carry deliberately extreme values, builds snapshots at a cutoff,
and asserts the extreme values never appear. A leak changes the numbers and fails the
test. This is the safety net that catches the case a reviewer misses.

**Prediction timestamp policy.** The default `prediction_timestamp` for a game is
`tip_off - PREDICTION_LEAD_MINUTES` (configurable, default 60 minutes). Training rows are
generated at the same offset so that training and serving see structurally identical
information. Generating training rows closer to tip-off than serving would inflate offline
metrics and disappoint in production.

---

## 9. Provider abstraction

```python
class BasketballDataProvider(Protocol):
    provider_key: str

    async def fetch_teams(self, season: SeasonRef) -> Sequence[TeamDTO]: ...
    async def fetch_players(self, season: SeasonRef) -> Sequence[PlayerStintDTO]: ...
    async def fetch_games(self, season: SeasonRef, *, round_number: int | None) -> Sequence[GameDTO]: ...
    async def fetch_game_stats(self, season: SeasonRef, game_code: int) -> GameStatsDTO | None: ...
```

Two implementations ship today:

- **`EuroLeagueProvider`** — talks to the public EuroLeague JSON API
  (`api-live.euroleague.net/v2`). Async `httpx`, bounded concurrency, retry with
  exponential backoff, optional Redis response cache.
- **`FixtureProvider`** — reads committed JSON fixtures from
  `backend/tests/fixtures/`. Deterministic, offline, used by the entire test suite.

Because both satisfy the same protocol, the ingestion pipeline is tested end-to-end
without network access, and adding Sportradar or an ACB adapter later requires no change
to ingestion, features, or models.

**Legal note.** The EuroLeague endpoints used are the same public, unauthenticated JSON
endpoints that serve the official website. The adapter identifies itself via a
configurable `User-Agent`, respects a configurable request rate, and caches aggressively.
No HTML scraping is performed anywhere in this codebase. If you intend to use this
commercially, obtain a licensed feed and write an adapter for it.

### News

No public EuroLeague news feed was reachable at build time, so **no news source is enabled
by default**. What ships is:

- a `NewsCollector` protocol,
- a generic `RssNewsCollector` that works against any RSS/Atom URL you configure via
  `NEWS_FEED_URLS`,
- `NewsArticle` storage with `published_at` / `observed_at`.

`NewsEvent` extraction (turning *"his minutes will be monitored tonight"* into
`{"event_type": "MINUTES_LIMIT", "player_id": …, "confidence": 0.91}`) is **planned**. The
table and enum exist so the schema does not need to change when extraction lands. The
extraction step will be an LLM call behind an interface, with entity resolution against
the `players` table and a confidence threshold below which events are stored but not fed
to features.

---

## 10. Background processing

Current: jobs are plain async functions in `ingestion/jobs.py`, invoked by the CLI.

```
euroleague-ai ingest teams    --season E2025
euroleague-ai ingest players  --season E2025
euroleague-ai ingest games    --season E2025
euroleague-ai ingest stats    --season E2025
euroleague-ai ingest all      --season E2025
euroleague-ai news collect
```

Planned progression:

1. **Now** — CLI + cron. Sufficient for daily ingestion and backfills.
2. **When jobs need retries and scheduling** — Celery beat + Redis. The job functions
   become task bodies unchanged.
3. **Never, unless proven necessary** — separate services per job type.

Each job run is recorded in `ingestion_runs` with status, row counts, and error detail, so
a failed overnight backfill is diagnosable without log archaeology.

---

## 11. Configuration

All configuration comes from environment variables via `pydantic-settings`
(`core/config.py`), loaded once and injected. There are no module-level reads of
`os.environ` outside that file, and no secrets in the repository. `.env.example` documents
every variable; `.env` is gitignored.

Notable knobs: `DATABASE_URL`, `REDIS_URL`, `EUROLEAGUE_API_BASE_URL`,
`EUROLEAGUE_REQUEST_RATE_PER_SECOND`, `PREDICTION_LEAD_MINUTES`, `LOG_LEVEL`,
`LOG_FORMAT`.

---

## 12. Observability

- **Structured logging** (`core/logging.py`). JSON in production, human-readable in
  development. A per-request correlation ID is bound via middleware and propagates into
  ingestion job logs.
- **Health endpoints.** `/health/live` is a process liveness check with no dependencies.
  `/health/ready` verifies Postgres and Redis connectivity and is what an orchestrator
  should poll.
- **Ingestion audit.** `ingestion_runs` gives per-job row counts and durations.
- **Planned:** Prometheus metrics, and model-drift monitoring driven by
  `PredictionOutcome`.

---

## 13. Testing strategy

| Layer | What it proves | Needs a DB? |
|---|---|---|
| `tests/unit` | Config, clock, time parsing, DTO mapping, baseline arithmetic | No |
| `tests/integration` | Migrations apply, models round-trip, temporal upserts behave, **no leakage** | Yes |
| `tests/integration/test_api.py` | App starts, routes respond, DB dependency resolves | Yes |

Database tests run against a real Postgres instance (never SQLite — the schema uses
Postgres-specific types) and each test runs in a transaction that is rolled back.
Tests requiring a database are marked `@pytest.mark.integration` and skip cleanly when
`TEST_DATABASE_URL` is unset, so `pytest` works on a laptop with no Docker running.

---

## 14. Architectural decisions

| # | Decision | Alternative | Why |
|---|---|---|---|
| 1 | Modular monolith | Microservices | One developer, one deployable, no scaling pressure. Boundaries are enforced by layering, and the cost of splitting later is low. |
| 2 | Async SQLAlchemy + asyncpg | Sync psycopg | Matches FastAPI, one driver, no sync/async duplication. Batch ML code that prefers sync can open its own engine. |
| 3 | Postgres for features | Dedicated feature store (Feast) | Volume is small (thousands of games, not billions of events). A feature store adds infrastructure without solving a problem we have. |
| 4 | Bitemporal columns, not `system_versioning` extensions | Temporal tables extension | Explicit columns are portable, obvious in queries, and reviewable. |
| 5 | Snapshots persisted, not recomputed | Recompute on demand | Guarantees a prediction can always be explained with the exact inputs it used, even after upstream corrections. |
| 6 | UUID primary keys | Bigint identity | Lets ingestion resolve identity and build graphs before flushing, and avoids ID collisions when merging data from multiple providers. |
| 7 | Separate `external_refs` table | `provider_id` columns on each table | One provider-agnostic identity mechanism; supports the same entity appearing in several providers with different IDs. |
| 8 | CLI before Celery | Celery from day one | No retry/schedule requirement yet. Adding Celery later is a wrapper around existing functions. |

---

## 15. Roadmap

**Phase 0 — Foundation (this commit).** Schema, migrations, provider abstraction, live
EuroLeague ingestion, read API, tests, Docker.

**Phase 1 — Feature layer.** `as_of` builders for form, role, schedule, team and opponent
context. Leakage test suite expanded per feature family.

**Phase 2 — Baselines.** Season average, last-N, weighted recency. Backtest harness and
MAE/RMSE/R² reporting. This is the bar every model must clear.

**Phase 3 — First model.** LightGBM per target, minutes first. Model registry, versioned
artifacts, `PredictionOutcome` evaluation loop.

**Phase 4 — Structured predictions.** Minutes → usage → opportunity → efficiency chain.
Monte Carlo simulation for `P(15+)`-style distributional output.

**Phase 5 — News intelligence.** Collector adapters, LLM event extraction, entity
resolution, availability features derived from news.

**Phase 6 — Domestic leagues.** ABA/ACB ingestion for fatigue, form, and role signals
between EuroLeague games.

**Phase 7 — Explanation and frontend.** Per-prediction attribution, then the Angular UI.

See [`ml-roadmap.md`](./ml-roadmap.md) for the modelling detail behind phases 1–5.

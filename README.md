# EuroLeague AI

[![CI](https://github.com/stevx11/el-ml-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/stevx11/el-ml-forecast/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![LightGBM](https://img.shields.io/badge/model-LightGBM-9ACD32)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine learning platform that predicts **EuroLeague player box-score lines before tip-off**: minutes, points, rebounds, assists, and three-pointers made.

The system continuously ingests games, rosters, and box scores. It records **what was known and when**, and turns that history into leakage-free feature snapshots. For each target it trains LightGBM models that must beat a strong baseline before they are allowed to serve. Every projection is published with a **distribution, threshold probabilities, and a plain-language explanation**, then scored against what actually happened.

```
Player: Marko Petrovic          Projected minutes: 31.4
Team: Partizan                  Points 18.7 · Rebounds 5.9 · Assists 6.2 · 3PM 2.1
Opponent: Real Madrid           P(15+) 72%  P(20+) 43%  P(25+) 17%

Why?  + Increased projected minutes    - Tough opponent defense
      + Usage up (teammate injured)    - Limited rest
```
<sub>Illustrative output format. The player and numbers are examples.</sub>

## Highlights

- **Bitemporal data model.** Nothing is overwritten. Every fact carries both when it happened and when the system learned it, so any past prediction can be replayed honestly.
- **Leakage prevention in four layers**: schema design, mandatory `as_of` arguments, a single SQL choke point, and dedicated adversarial tests.
- **Baseline-gated models.** A model is registered either way but only activated if it beats a minutes-scaled rate baseline on a chronological holdout. Negative results are kept, not discarded.
- **Who actually plays.** A separate classifier predicts the chance each roster candidate takes the floor at all. It is trained on exactly the population inference projects, including players who were left out, and gated on log loss against appearance-rate baselines.
- **Expected values that add up.** The L1 models predict medians, and medians of skewed box scores sum to squads that score roughly eight points less than real ones. Each target also gets a mean-targeting head (Tweedie booster or shifted median, chosen on validation), which is published only if it beats the median on test RMSE and on squad-total bias. Weighted by P(plays), it gives realistic team totals.
- **Distributions, not just point estimates.** Calibrated intervals and P(X+) thresholds come from empirical held-out residuals, with coverage measured directly.
- **Explainable projections.** Exact TreeSHAP contributions are grouped into themes (minutes, form, role, opponent, schedule, ...) and phrased as readable reasons.
- **Closed feedback loop.** Published predictions are scored against real box scores for point accuracy and interval calibration.
- **Unattended operation.** A scheduled pipeline runs ingest → score → predict twice a day, with each step recovering independently.
- **Production engineering**: async FastAPI, SQLAlchemy 2 with Alembic migrations, Docker Compose, a CI migration round-trip, and an extensive unit and integration test suite.

## Tech Stack

| Area | Tools |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Data | PostgreSQL 16 (asyncpg), SQLAlchemy 2, Alembic, Redis |
| ML | LightGBM, scikit-learn, pandas, polars, NumPy |
| CLI | Typer |
| Frontend | Vanilla JS, HTML, CSS (served by FastAPI, no build step) |
| Quality | pytest, ruff, mypy (strict), GitHub Actions |
| Ops | Docker, Docker Compose |

## Quick Start

Requires Docker and Docker Compose. Nothing else.

**macOS / Linux**

```bash
cp .env.example .env

make up                              # build, start postgres + redis + api, migrate
make ingest SEASON=E2025             # pull a season from the public EuroLeague API
```

**Windows (PowerShell).** `make` isn't available, so use the bundled script:

```powershell
Copy-Item .env.example .env

.\make.ps1 up
.\make.ps1 ingest -Season E2025
```

If PowerShell blocks the script, allow local scripts for this session with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Then open:

- **Web app**: <http://localhost:8000/app>
- **API docs**: <http://localhost:8000/docs>
- **Health check**: <http://localhost:8000/health/ready>

`make help` (or `.\make.ps1 help`) lists every target. Both wrappers expose the same targets.

### From data to predictions

Run these inside the backend container with `docker compose exec backend <command>`:

```bash
euroleague-ai ingest seasons --from 2016 --to 2025 --wait   # backfill history, newest first
euroleague-ai features backfill --from 2016 --to 2025       # build as_of snapshots (incl. roster candidates)
euroleague-ai ml baselines                                  # publish the bar to beat
euroleague-ai ml train                                      # train all five targets, then P(plays)
euroleague-ai news collect                                  # fetch feeds, read injuries out of articles
euroleague-ai news injuries                                 # what was read: published date, scope, sentence
euroleague-ai news extract --replace --days 30              # re-read stored articles after a rules change
euroleague-ai injuries list                                 # reports that apply to next games
euroleague-ai injuries set Petrovic out --reason knee      # manual override (before T-60 min)
euroleague-ai ml reconcile                                  # fit squad-minutes scaling (after each train)
euroleague-ai ml predict --days 7                           # project upcoming fixtures
euroleague-ai ml why --game-code 120 --player petrovic      # explain one projection
euroleague-ai ml score                                      # score finished games
euroleague-ai ml accuracy                                   # realised accuracy + calibration
```

To run `ingest → score → predict` automatically at the UTC times in `PIPELINE_RUN_AT`, start the opt-in scheduler service:

```bash
docker compose --profile scheduler up -d scheduler
```

---

## Architecture

A **modular monolith**: one deployable unit with clear internal boundaries, and no microservices until something actually requires them.

```
EuroLeague API ──▶ providers/ ──▶ ingestion/ ──▶ POSTGRES ──▶ features/ ──▶ ml/ ──▶ predictions
  RSS feeds   ──▶   news/                      (bitemporal)                         │
                                                    │                               ▼
                                                    └──────────▶ api/ ◀──── scored outcomes
                                                                  │
                                                                  ▼
                                                             frontend/ (/app)
```

| Layer | Responsibility |
|---|---|
| `providers/` | Adapters to external sources: a `BasketballDataProvider` protocol, a live EuroLeague adapter, and an offline fixture adapter. Domain code never sees provider JSON. |
| `ingestion/` | Identity resolution and temporal upserts. Decides whether a fact is an upsert, an interval close, or an append. |
| `models/` | SQLAlchemy ORM. The bitemporal schema is the source of truth. |
| `repositories/` | All SQL, split into `catalog` (current state, for the API) and `history` (**`as_of` mandatory**, for features). |
| `features/` | As-of feature builders: player form, role, team context, opponent, availability, position. This is the leakage boundary. |
| `ml/` | Baselines, training, tuning, distributions, explanations, inference, outcome scoring, model registry. |
| `scheduling/` | The unattended daily pipeline. |
| `api/` | FastAPI routers for games, players, teams, predictions, and the fixtures/accuracy board. |

Full detail: [`docs/architecture.md`](docs/architecture.md) · [`docs/data-model.md`](docs/data-model.md) · [`docs/ml-roadmap.md`](docs/ml-roadmap.md)

### Project structure

```
el-ml-forecast/
├── backend/
│   ├── src/euroleague_ai/
│   │   ├── api/          HTTP layer (FastAPI routers, dependencies)
│   │   ├── core/         config, logging, clock, errors
│   │   ├── db/           engine, session, declarative base
│   │   ├── models/       SQLAlchemy ORM: the bitemporal schema
│   │   ├── schemas/      Pydantic request/response models
│   │   ├── repositories/ all SQL; catalog (now) vs history (as_of)
│   │   ├── providers/    BasketballDataProvider protocol + adapters
│   │   ├── ingestion/    identity resolution, temporal upserts, jobs
│   │   ├── news/         RSS/Atom collectors
│   │   ├── features/     as_of feature builders and snapshots
│   │   ├── ml/           baselines, training, distributions, explanations, scoring
│   │   ├── quality/      data quality checks
│   │   ├── scheduling/   unattended ingest → score → predict loop
│   │   ├── main.py       app factory
│   │   └── cli.py        operational entry point (Typer)
│   ├── migrations/       Alembic
│   └── tests/            unit · integration · fixtures
├── frontend/             static web client (fixtures, projections, accuracy)
├── data/                 raw / processed / features / models (gitignored)
├── docker/               Dockerfile + postgres init
├── docs/                 architecture · data-model · ml-roadmap
├── scripts/              fixture capture
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

**Why one package instead of separate `ml/` and `news/` trees.** Feature engineering, news extraction, and prediction all need the same ORM models, the same `as_of` clock, and the same identity resolution. Separate top-level trees would mean duplicated code or `sys.path` manipulation. Boundaries are enforced by a strict layering rule instead (`api → {ingestion, features, ml} → repositories → models → db → core`).

---

## Why Time Matters

Two requirements drive every design decision:

1. **A prediction is only meaningful relative to an instant.** A projection made 48 hours before tip-off is a different prediction from one made after the lineup is announced. The system must be able to answer *"what did we know at 17:00?"*
2. **A prediction that can't be backtested can't be trusted.** If today's model can't be replayed honestly against the past, there is no way to know whether it works.

Both reduce to one principle: **the database records the history of knowledge, not just the current state of the world.** Every feature query is filtered by *when each fact was learned*, not when it happened. A report published at 19:00 about a 10:00 injury was not knowable at 17:00. Filtering on event time alone looks correct but still leaks.

## How Leakage Is Prevented

Leakage produces excellent offline metrics and a useless product. There are four defences, deliberately redundant:

**1. Schema.** No table has a "current status" column for a feature to read by accident. Current state is always derived from timestamped history.

**2. API shape.** Every feature-relevant repository method takes `as_of` as a mandatory keyword argument, with no default anywhere:

```python
async def build_player_form_features(
    session, *, player_id, team_id, as_of: datetime,  # mandatory
) -> dict[str, float | None]:
```

**3. One choke point.** Repositories apply the filter; feature builders never write SQL. The leakage-critical predicate lives in a handful of reviewable functions:

```sql
WHERE observed_at <= :as_of      -- what we had actually recorded
  AND tip_off_utc  <  :as_of     -- and the game had actually happened
```

**4. Tests.** `test_no_leakage.py` seeds post-cutoff rows with deliberately extreme values, builds snapshots at the cutoff, and asserts the extreme values never appear. It also covers the subtle case: a game that tipped off before the cutoff but was *recorded* after it. That game passes an event-time filter and leaks anyway.

Missing data is `None`, never `0.0`. A rookie's ten-game average is *unknown*; substituting zero would teach the model that players without history score nothing.

---

## Modelling Approach

| Decision | Rationale |
|---|---|
| **Minutes are predicted first** | Every other box-score target depends on playing time, and minutes are by far the most predictable target (baseline R² 0.48 vs 0.31 for points). |
| **L1 objective** | Box-score distributions are right-skewed. MAE is the reported metric, so MAE is what gets optimised, rather than chasing the occasional 40-point night. |
| **Chronological holdout only** | A random split puts a player's adjacent games on both sides, so the model would be tuned for interpolation, a task it never faces in production. |
| **Strong baseline gate** | The model to beat is a per-minute rate scaled by recent minutes, which is far stronger than a season average. |
| **Empirical residual distributions** | Intervals come from how wrong the model actually was on held-out data. This needs no extra models and can't produce crossing quantiles. |
| **Select on validation, report on test** | Hyperparameter search never reports the score it selected on, so selection noise can't be passed off as accuracy. |
| **Median for the gate, mean for the headline** | MAE rewards the median, so the models are trained for it. A reader adds lines up, which needs the mean. Both are stored: the median for scoring and intervals, the expected value for display and team totals. |
| **Availability is its own model** | Stat models predict a line *if he plays*. P(plays) is a classifier over the full candidate list, because folding absences into the regressions as zeros would teach them that injured stars score nothing. |
| **Experiments are measured, not assumed** | A structured minutes × rate chain was built and tested; the direct model won on all four targets, so the chain was rejected. |

Every trained model is recorded in a registry with its training data, scores, and artifact path. Together with `prediction_timestamp` and `feature_snapshot_id`, that makes any prediction reproducible months later.

---

## Data Sources

**EuroLeague**: the public JSON API at `api-live.euroleague.net/v2`, the same endpoints that serve the official website. The adapter identifies itself with a configurable User-Agent, rate-limits itself, bounds concurrency, retries only transient failures, and caches responses in Redis. **No HTML scraping happens anywhere in this codebase.** For commercial use, obtain a licensed feed and write an adapter; the provider protocol makes that a drop-in change.

| Data | Endpoint |
|---|---|
| Clubs | `/competitions/{c}/seasons/{s}/clubs` |
| Players + roster intervals | `/competitions/{c}/seasons/{s}/people?personType=J` |
| Schedule | `/competitions/{c}/seasons/{s}/games` |
| Box scores | `/competitions/{c}/seasons/{s}/games/{code}/stats` |

**News and injuries**: `NEWS_FEED_URLS` points at RSS feeds with full article text (the Eurohoops EuroLeague feed by default). `news collect`, which the scheduler runs before predictions, stores new articles with the time they were fetched, then reads injury statuses out of them by rule. It matches rostered players by name, takes the status from the wording ("ruled out", "game-time decision", "returns from"), and skips negations and past tense. Each reading keeps its sentence and a confidence. A confident "out" (0.8 or higher) sets P(plays) to 0 and squad minutes go to teammates; weaker readings are flagged only. `news injuries` lists what was read. Reports can also be entered by hand with `injuries set` or `injuries import`, and the newest report wins. The official EuroLeague site blocks automated access, so it is not read directly.

**Domestic leagues** (ABA, ACB, ...) need no schema change: a different `league_id`, same tables. This is currently blocked on a data source, not on architecture.

---

## Development

### Local setup without Docker

Requires Python 3.12+, PostgreSQL 16, and Redis.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"
cp .env.example .env                                   # point DATABASE_URL at localhost
alembic -c backend/alembic.ini upgrade head
uvicorn euroleague_ai.main:app --reload
```

### Ingestion

```bash
euroleague-ai ingest all     --season E2025      # everything, in dependency order
euroleague-ai ingest games   --season E2025 --round 12
euroleague-ai ingest stats   --season E2025 --game-code 1
```

Ingestion is **idempotent**: re-running never duplicates rows, so a nightly job and an ad-hoc backfill are the same command. Every run is recorded in `ingestion_runs` with row counts and errors.

### Migrations

```bash
make migrate                              # alembic upgrade head
make revision M="add lineup confidence"   # autogenerate
```

Read every autogenerated revision before applying it. Alembic can't infer intent: a renamed column looks the same as a drop plus an add.

### Tests

```bash
make test                                                      # everything
docker compose exec backend pytest backend/tests/unit          # no database needed
docker compose exec backend pytest -m integration              # database-backed
```

| Suite | Proves | Needs a DB? |
|---|---|---|
| `tests/unit` | Parsing, mapping, feature maths, baselines, distributions, explanations, tuning, scheduling | No |
| `test_database.py` | Connectivity, constraints, temporal invariants | Yes |
| `test_ingestion.py` | End-to-end ingest, idempotency, identity resolution | Yes |
| `test_features.py` | Feature snapshots built from seeded history | Yes |
| `test_no_leakage.py` | **Features never see the future** | Yes |
| `test_api.py` | The app starts, routes respond, and the DB dependency resolves | Yes |
| `test_migrations.py` | Migrations and models have not drifted | Yes |

Database tests run against a real PostgreSQL instance (never SQLite, because the schema uses Postgres-specific types), each inside a rolled-back transaction. They **skip** cleanly when `TEST_DATABASE_URL` is unset, so `pytest` still works on a laptop without Docker.

### CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

| Job | What it proves | Blocking |
|---|---|---|
| `tests` | Full suite against a real PostgreSQL 16 service | Yes |
| `migrations round-trip` | `upgrade head` → `downgrade base` → `upgrade head` | Yes |
| `lint and types` | ruff + mypy `--strict` | Not yet |

In CI the test job **fails if the integration tests skip**. Locally, skipping is a convenience; in CI it would hide every leakage guard behind a green tick.

---

## Environment Variables

Every variable is documented in [`.env.example`](.env.example) and read exactly once, by `euroleague_ai.core.config.Settings`. No other code reads `os.environ`, and the repository contains no secrets.

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | — | **Must** use `postgresql+asyncpg://`. A sync DSN fails at startup, not at the first query. |
| `TEST_DATABASE_URL` | — | If unset, database tests skip instead of failing. |
| `REDIS_URL` | `redis://redis:6379/0` | Response cache. |
| `DATA_PROVIDER` | `euroleague` | `euroleague` (live) or `fixture` (offline, deterministic). |
| `PREDICTION_LEAD_MINUTES` | `60` | Cutoff before tip-off. Training and serving use the same value on purpose. |
| `EUROLEAGUE_REQUEST_RATE_PER_SECOND` | `2` | Politeness towards a free public API. |
| `PIPELINE_RUN_AT` | `06:00,16:00` | UTC times for the scheduled pipeline. |
| `NEWS_FEED_URLS` | *(empty)* | Comma-separated. No feeds ship enabled. |
| `LOG_FORMAT` | `console` | Use `json` in production. |

---

## Roadmap

- News event extraction (injury, minutes-limit, and lineup signals) as model features
- Domestic-league data for fatigue and form signals
- Make the lint and type-check CI job blocking
- Additional targets (steals, blocks, turnovers)

## License

[MIT](LICENSE). This project is not affiliated with EuroLeague. You are responsible for complying with the terms of any data source you configure.

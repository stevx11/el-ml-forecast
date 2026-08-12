# ML roadmap

How the prediction system gets built, in the order it gets built, and what has to be true
before each step is allowed to proceed.

The governing rule: **a model that does not beat a meaningful baseline is not a model, it
is a liability.** Every phase below ends with a measured comparison, and the project does
not advance past a phase that fails it.

---

## Phase ordering

| Phase | Deliverable | Gate to pass before moving on |
|---|---|---|
| 0 | Foundation: schema, ingestion, storage | Migrations apply; a full season ingests; leakage tests pass |
| 1 | `as_of` feature builders | Snapshots reproducible; leakage tests pass per feature family |
| 2 | Baselines + backtest harness | Baseline MAE published for every target |
| 3 | LightGBM per target | Beats best baseline on walk-forward validation |
| 4 | Structured chain + distributions | Calibration error within tolerance |
| 5 | News-derived features | Measurable lift over Phase 3 features |
| 6 | Domestic-league signals | Measurable lift on fatigue/form features |

---

## Phase 1 — Features

### Contract

```python
def build_player_features(
    session: AsyncSession,
    *,
    player_id: UUID,
    game_id: UUID,
    as_of: datetime,          # mandatory, no default
) -> dict[str, float | None]:
```

`as_of` is a required keyword argument with no default anywhere in the feature layer. You
cannot compute a feature without stating what the model is allowed to see. Every
repository call underneath filters `observed_at <= as_of`.

Missing values are `None`, never `0.0`. A rookie's "last 10 games average" is unknown, and
LightGBM handles genuine NaN natively — silently substituting zero teaches the model that
rookies score nothing.

### Feature families

**Player form** — for each of `minutes, points, rebounds, assists, fg3m, fga, fta, usage`:
season mean, last-3 / 5 / 10 / 20 means, exponentially weighted mean (half-life ≈ 5 games),
median, standard deviation, and OLS slope over the last 10 games.

DNPs are excluded from form windows rather than counted as zeros; a separate
`games_missed_last_10` feature carries that information explicitly, because "rested" and
"scored zero in 22 minutes" are different states.

**Player role** — `is_starter_last_game`, `starts_in_last_5`, `starter_rate_season`,
`minutes_volatility` (σ of last 10), `usage_rate`, `fga_per_36`, `fg3a_rate`, `fta_rate`,
`assist_rate`, `rebound_rate`, `turnover_rate`.

**Schedule** — `days_rest`, `games_last_7d`, `games_last_14d`, `minutes_last_7d`,
`previous_game_minutes`, `is_back_to_back`, `is_short_rest` (< 2 days), `is_home`,
`travel_km` (great-circle between venue coordinates), `timezone_delta_hours`,
`is_euroleague_double_week`.

The last one matters disproportionately in Europe. Double-round weeks compress two
EuroLeague games plus a domestic fixture into six days, and rotations visibly shorten.

**Team context** — `offensive_rating`, `defensive_rating`, `net_rating`, `pace`,
`oreb_pct`, `dreb_pct`, `turnover_pct`, `fg3a_rate`, `fta_rate`, plus last-5 form
variants.

**Opponent** — `opp_defensive_rating`, `opp_pace`, `opp_dreb_pct`, `opp_fg3_pct_allowed`,
`opp_points_in_paint_allowed`, `opp_turnovers_forced`, `opp_fta_allowed`, and
`opp_def_rating_vs_position`.

All opponent aggregates are computed strictly from games with `tip_off_utc < as_of`. This
is the classic leakage trap: computing season-long opponent ratings once and reusing them
for every game in the season means March's defensive rating informs an October prediction.

**Availability context** — `teammates_out_count`, `teammates_questionable_count`,
`teammate_minutes_out` (sum of season minutes of unavailable teammates — a far better
usage-shift proxy than a raw count), `rotation_size_last_5`,
`is_lineup_confirmed`, `starter_probability`.

**News-derived (Phase 5)** — `has_minutes_limit_flag`, `days_since_return_from_injury`,
`role_change_recent`, each weighted by extraction confidence.

### Leakage tests

Every feature family gets a test in the same shape: seed a fixture world where all
post-cutoff rows carry deliberately extreme values (a 60-point game, a season-ending
injury), build a snapshot at the cutoff, assert the extreme values do not move the
features. A leak changes the numbers and the test fails.

---

## Phase 2 — Baselines

Implemented before any model, in `euroleague_ai/ml/baselines.py`:

1. **Season mean** — player's season average for the target.
2. **Last-5 mean.**
3. **Last-10 mean.**
4. **Weighted blend** — `0.4 × season + 0.6 × last-5`.
5. **Minutes-scaled rate** — season per-minute rate × last-5 mean minutes. Usually the
   strongest simple baseline, because most of the variance in a box-score line is variance
   in playing time.

Baseline #5 is the one to beat. Anecdotally it is competitive with a poorly-featured GBDT,
and reporting it honestly prevents the classic self-deception of comparing a tuned model
against a deliberately weak straw man.

### Backtest protocol

**Walk-forward by round, never a random split.** Train on rounds `1..k`, predict round
`k+1`, advance. A random `train_test_split` over player-games leaks: the same player's
adjacent games land on both sides, and the model learns to interpolate rather than
forecast. Reported accuracy under random splitting is roughly meaningless for this task.

The first four rounds of a season are excluded from evaluation — form windows are too
sparse to be informative, and including them mostly measures how the model handles
missingness.

---

## Phase 3 — First model

One LightGBM regressor per target, trained independently:

```
features → LightGBM → point prediction
```

Order of construction: `minutes` first, then `points`, `rebounds`, `assists`, `fg3m`.

**Minutes comes first because everything else is conditional on it.** A points model
trained without a minutes signal spends most of its capacity re-learning rotation
patterns. Once minutes are predicted well, downstream targets can consume
`predicted_minutes` as an input feature and concentrate on efficiency and role.

Objective: L1 (`regression_l1`) rather than L2. Box-score distributions are
right-skewed and MSE chases the 40-point outliers at the expense of the 90% of predictions
that matter for typical use.

Sample restriction: rows where the player was available and the team played. DNPs are
handled by a separate availability model rather than being folded into the regression as
zeros.

**Gate:** beats baseline #5 on walk-forward MAE, on a held-out season, for every target.
If it does not, the fix is features, not hyperparameters.

---

## Phase 4 — Structured chain and distributions

Only after Phase 3 clears its gate.

```
        GAME
          │
          ▼
   MINUTES MODEL ─────────────────┐
          │                       │
          ▼                       │
    ROLE / USAGE                  │
          │                       │
          ▼                       │
  OPPORTUNITY MODEL               │  predicted minutes feeds
  (FGA, FG3A, FTA, touches)       │  every downstream stage
          │                       │
          ▼                       │
  EFFICIENCY MODELS ◀─────────────┘
  (FG%, 3P%, FT%, AST rate, REB rate)
          │
          ▼
   MONTE CARLO SIMULATION
          │
          ▼
     DISTRIBUTIONS
```

Decomposition rather than one big model because the components are individually testable
and individually explainable. When a projection looks wrong, the chain localises the
error — bad minutes, bad usage, or bad efficiency — where a monolithic model offers only
a SHAP plot and a shrug.

### Distributional output

Simulation, per player, per game:

1. Sample minutes from the minutes model's predictive distribution.
2. Sample possessions/usage conditional on minutes.
3. Sample shot attempts by type conditional on usage.
4. Sample makes from efficiency models (beta-binomial, so small samples do not produce
   overconfident percentages).
5. Aggregate into a stat line. Repeat ~10,000 times.

Yields the target product output:

```
Expected: 18.7    Median: 18
P(15+): 72%   P(20+): 43%   P(25+): 17%
```

**Do not build the simulator before the point predictions are reliable.** A simulation
wrapped around a poorly calibrated mean produces confident, precise, wrong probabilities —
strictly worse than an honest point estimate, because it invites trust it has not earned.

---

## Phase 5 — News

```
source → collector → parser → entity resolution → LLM extraction → NewsEvent → features
```

Entity resolution against the `players` table handles the hard cases: transliteration
(Šarić / Saric), name order (EuroLeague returns `SURNAME, FORENAME`), and shared surnames
disambiguated by team context.

Extraction is an LLM call behind an interface, returning typed events with confidence.
Events below a configurable threshold are stored but excluded from features. Generic
sentiment analysis is explicitly not the goal — the signal is the basketball event.

**Gate:** measurable MAE improvement over Phase 3 features. News features are the most
likely source of spurious lift, since article timestamps are easy to get subtly wrong, so
this gate is tested with particular care against `observed_at` rather than `published_at`.

---

## Phase 6 — Domestic leagues

EuroLeague players also play ABA, ACB, Lega A, and BSL games between EuroLeague fixtures.
Ignoring those games misstates fatigue, form, and role.

Ingesting them requires no schema change — different `league_id`, same tables. Features
gain league-scoped variants: `minutes_last_7d_all_competitions` alongside
`minutes_last_7d_euroleague`, and `games_last_7d_all_competitions`, which is often
materially higher than the EuroLeague-only count during a double week.

Blocked on a data source, not on architecture.

---

## Evaluation

**Point predictions:** MAE (primary), RMSE, R², and MAE stratified by projected-minutes
bucket. The stratified view matters — a model can look strong overall while being useless
for the 10–20 minute rotation players, which is exactly where the interesting decisions
are.

**Probabilistic predictions:** Brier score, log loss, reliability diagrams, and prediction
interval coverage (an 80% interval should contain the truth ~80% of the time).

**Continuous monitoring:** `prediction_outcomes` is written by a post-game job, so rolling
MAE by model version is a SQL query. Drift shows up as a trend, not as a surprise.

---

## Model registry

Every trained model writes a `model_versions` row: name, semver, target, algorithm,
`feature_set_version`, hyperparameters, `training_data_hash`, metrics, artifact path.

Every prediction references `model_version_id`, `prediction_timestamp`, and
`feature_snapshot_id` — enough to reproduce it exactly, months later, even after upstream
data corrections.

---

## Explanation

Per-prediction attribution from SHAP values on the GBDT, mapped to human-readable drivers
and stored in `predictions.explanation`:

```
Why?
+ Increased projected minutes
+ Increased usage due to teammate injury
+ Recent scoring trend
- Tough opponent defense
- Limited rest
```

The mapping from feature name to phrase is a curated table, not generated text. Feature
attributions are numerically faithful; the phrasing is a presentation layer over them, and
keeping the two separate prevents an explanation that sounds plausible while contradicting
the model.

---

## Known risks

| Risk | Mitigation |
|---|---|
| **Small data.** ~340 EuroLeague games/season, ~25 relevant players/game. | Pool across seasons and competitions. Prefer GBDT over deep learning until data volume justifies otherwise. Regularise aggressively. |
| **Rotation volatility.** European coaches change rotations sharply and without warning. | Recency-weighted features; explicit coach-change signals; wide predictive intervals on low-minutes players. |
| **Roster churn.** Mid-season transfers and short-term injury replacements are routine. | Interval-based stints; features degrade gracefully to team/positional priors when player history is thin. |
| **Leakage.** The failure mode that produces excellent offline metrics and a useless product. | Mandatory `as_of`, single filtering choke point, dedicated test suite, walk-forward validation only. |
| **Overfitting to a strong season.** | Walk-forward validation, held-out season, baseline comparison at every phase gate. |
| **News timestamp error.** Publishers backdate; syndication reorders. | Filter on `observed_at`, never `published_at`. Treat any lift from news features as suspect until reproduced on a fresh season. |

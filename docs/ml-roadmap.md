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
handled by a separate availability model (below) rather than being folded into the
regression as zeros.

### Availability: P(plays)

Every stat model predicts a line *given that the player plays*, but inference projects
every roster candidate, and that list always includes a few who will not appear.
`ml/availability.py` predicts the probability that each candidate takes the floor.

- **Population:** exactly the inference candidate list, rebuilt per historical game with
  `repositories.rosters.candidate_players` at the same cutoff. The snapshot backfill
  writes rows for these candidates too, so injured players finally have a row.
- **Label:** 1 if the player has playing time in this team's box score, else 0. "Dressed,
  never used" and "not in the squad" are both 0, and both are reported separately.
- **Features:** the v5 snapshot, unchanged. Recent-appearance counts are baseline inputs
  only. If the model cannot beat them, that is the case for adding them in v6.
- **Baselines:** league base rate; P(plays | played last game); P(plays | played k of the
  team's last n games, debut flag), shrunk toward the base rate.
- **Gate:** lower test log loss than the best baseline. Also reported: Brier, AUC,
  calibration error, a reliability table, and the error in *expected players used per
  team-game* (the sum of P(plays) against how many actually got on the floor).

Served as target `PLAYS`, scored after each game like every other target.

### Expected values

The L1 models predict the conditional median. For right-skewed box scores the
median sits below the mean, and summed medians miss real squad totals badly. Model
0.1.8's held-out residuals put the gap at roughly 8 points, 4 rebounds, 3 assists and 3
made threes per team. `ml/expectation.py` adds a mean-targeting head to every target:

- **Candidates:** a second booster (Tweedie, power 1.3, for counts; squared error for
  minutes), and the median plus its error cell's *mean* residual.
- **Selection:** lower validation RMSE. **Report:** test.
- **Gate:** beats the median it replaces on test RMSE *and* on squad-total bias. RMSE
  against the best baseline is reported as a warning, not a gate.
- **Serving:** `predictions.expected_value` holds E[X | plays] beside the median.
  Multiplied by P(plays), it gives the realistic line, and the board sums it into squad
  totals. Models trained before this change serve the shifted median from their stored
  distribution.
- **Monitoring:** `ml accuracy` reports expected-value bias and RMSE, plus squad-total
  error for medians vs expected values against the real box score.

### Squad minutes reconciliation

Minutes are predicted player by player, so nothing makes a squad add up to the
minutes a real team plays. `ml/reconcile.py` scales each side of a fixture by one
factor so the squad's P(plays)-weighted minutes move toward the measured average for
real squads (just over 200, because of overtime):
`s = (T / sum P(plays) x minutes) ** alpha`, capped at 40 minutes per player, with any
capped excess spread over the rest of the side.

- **Fit:** `euroleague-ai ml reconcile`. It scores the active MINUTES and PLAYS models
  on periods neither has seen, picks the strongest `alpha` from 0 to 1 whose validation
  player-minutes MAE is no worse than unscaled, and reports the result on test.
- **Gate:** on only if test squad error is smaller and player MAE is not significantly
  worse: a paired scaled-vs-unscaled comparison, with the standard error clustered by
  squad, failing above two standard errors.
- **Serving:** applied to the minutes expected value only (median, intervals and
  thresholds untouched), and only for the model versions it was fitted on. Rerun it
  after every `ml train`.

### Squad stat reconciliation

The same idea for points, assists, rebounds and threes, except the squad target is
not a constant. `ml/squad_stats.py` fits, per stat, a linear anchor for each side's
total on the validation period: the squad's P(plays)-weighted expected sum plus
team context known at the cutoff (expected team points and pace for points; the
squad's expected points as well for assists; both squads' expected rebounds, both
teams' rebounds per game, pace and the expected total for rebounds; threes attempted
per game and pace for threes). Each side is scaled by `(A / S) ** alpha`, bounded to
0.6-1.6.

- **Fit:** the same `ml reconcile` run, written to `reconciliation/stats.json`.
- **Gate, per stat:** test squad error must fall and player squared error must not be
  significantly worse (paired, clustered by squad). A stat that fails stays off.
- **Serving:** expected values only, after injury reports (a player ruled out drops
  out of `S`, so teammates take his share) and minutes reconciliation. Each adjusted
  line records its `squad_factor` in the explanation.

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

### Per-player spreads

The served distribution is the held-out residual distribution, conditioned on projected
level and history depth. On its own that gives a streaky shooter and a steady big man
projected for the same 12 points identical ranges. Each player's residuals are now scaled by
his own recent swing:

```
k = clip((volatility / median volatility at his projected level) ** power, 0.5, 2.0)
```

The cells store residuals divided by `k`, and a prediction multiplies its cell's quantiles
(and its mean residual, for the shifted-median expected value) by its own `k`. Volatility is
the target's own `std10_<stat>`, or `std10_minutes`. Players with no recent games get
`k = 1`.

The source and the power (0.25 to 1) are chosen **on validation only**. Every candidate is
fitted on the earlier 60% of validation and scored by quantile (pinball) loss on the later
40%. A candidate is kept only if it beats the unscaled spread by 0.2%; otherwise the spread
stays unscaled, exactly as before. `ml train` prints the choice, the 80% coverage per
volatility third next to the unscaled coverage, and the Brier score of the published P(X+)
chances next to the unscaled one. A model trained before this loads unscaled.

**Early in a season** the error is wider than the history-depth cells know: form windows
still reach into last season and roles are new. In the walk-forward folds the season being
played covered 77% where full seasons covered 79-80%, and live coverage was 76-77%. A second
factor, chosen on validation the same way and on top of the volatility scale, widens the
spread while little of the player's last 20 games is from this season:
`k_season = 1 + w * (1 - share)`, with `w` from 0.05 to 0.5.

**What is served** is the chosen shape refitted on validation + test, the most recent
errors there are (the booster never trained on either). The coverage and Brier numbers in
the training report still come from the validation fit, so they stay out of sample.

### Injury labels and P(plays)

"Probable", "questionable", "doubtful", "ready" and unapplied news "out" readings move
P(plays) by one log-odds shift per label, fitted on scored rounds with a normal prior on
the shift (sd 1), and switched on only if each round, scored by the other rounds' fit, has
a lower log loss than the model alone (`injuries calibrate`, after each scored round).
Round 8 showed why: players listed probable got a mean P(plays) of 0.33 and 69% played.
The model's own P(plays) is kept in each prediction (`availability.model_p_plays`), so
every refit starts from it.

---

## Feature set v6

`features/context_v6.py` adds four families, every one additive (no v5 feature
changes name or meaning), which is why v6 vectors can serve v5 models
(`features.registry.SERVES`) until the backfill and retrain are done:

- **appearances**: share of the team's last five games this season he played,
  team games in that window, not yet debuted for the club;
- **opponent_position**: what the opponent allowed to his position group over its
  last ten games (points, assists, threes, rebounds per game; share of points);
- **coach**: the current coach's games in charge, changed within five games;
- **schedule_venue**: days to the team's next fixture (fixtures are knowable seven
  days ahead, live and in the backfill alike), a within-three-days flag, a
  postseason flag, and his home/away points and minutes split.

Leak tests in `tests/integration/test_features_v6.py` cover late box scores,
late fixtures and coach changes. Rollout: `features backfill` (v6 snapshots), then
`ml ablate` (each family removed in turn, against twice the retraining noise),
`EXCLUDED_FEATURE_FAMILIES` for any family that helps nothing, `ml train` (gated
season by season), `ml reconcile`, `ml predict`.

## Phase 5a — Injury reports (started)

Collected automatically from news feeds with full text (`news collect`, run by the
scheduler before predictions), and readable from the official round report as
written up there. `news/injury_extract.py` reads statuses by rule: rostered names,
clause-level status phrases, negation and past-tense guards, a confidence per reading.
Readings land in `news_events` (audit) and `player_status_events` (`source = NEWS`,
`observed_at` = fetch time). Each reading gets a scope from its wording (rules-v2):
"ruled out for X's game on Friday" covers the team's next fixture only; "sidelined
for 6 to 8 weeks" lasts until the lower bound; anything else is open but lapses after
8 days unless a newer article repeats it. Recaps ("last played in Round 2", "it was
communicated that…") are discounted below the rule-out threshold. After a rules
change, `news extract --replace --days N` re-reads stored articles and drops the
old readings. The daily BasketNews table (`injuries report`, `source = REPORT`) is the
primary source: one row per player for all 20 clubs, statuses set by an editor,
the Round column as scope, only changes written, and a player who drops off the
table recorded as back. Every fetch is logged (`ingestion_runs`,
`injury_report.fetch`) with the clubs it listed, and a player's newest table row
counts as confirmed at the last fetch that repeated it: a news reading outranks
the table only if it is newer than that fetch. News extraction (rules-v3) never
carries a status across a contrast word ("X, Y and Z are all available while W
is ruled out"). Manual entry remains for overrides:
`euroleague-ai injuries set|import|list|history`. Reports are scoped to one game,
to every game before a return date, or open-ended until the player plays again.

At prediction time the report that applies at the cutoff (newest first) is attached to
every line. **Out** and **suspended** set P(plays) to 0, before minutes reconciliation,
so teammates absorb the minutes. News readings must reach 0.8 confidence to do so. Other labels are shown but change nothing yet: there
is no history of how often a "game-time decision" plays. After a season of reports
they can become calibrated features.

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

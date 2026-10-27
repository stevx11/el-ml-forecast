-- Accuracy for one round, active models only.
-- Run with:  psql ... -v round=7
-- The season is the one of the most recently scored game.
-- Read-only: temporary views vanish when psql exits.

CREATE TEMP VIEW g AS
SELECT ga.id, ga.game_code
FROM games ga
WHERE ga.round_number = :round
  AND ga.season_id = (
      SELECT ga2.season_id FROM games ga2
      WHERE EXISTS (SELECT 1 FROM predictions p JOIN prediction_outcomes o ON o.prediction_id = p.id
                    WHERE p.game_id = ga2.id)
      ORDER BY ga2.tip_off_utc DESC LIMIT 1)
  AND EXISTS (SELECT 1 FROM predictions p JOIN prediction_outcomes o ON o.prediction_id = p.id
              WHERE p.game_id = ga.id);

CREATE TEMP VIEW plays AS
SELECT p.game_id, p.player_id, p.predicted_value AS p_plays, o.did_not_appear
FROM predictions p
JOIN prediction_outcomes o ON o.prediction_id = p.id
JOIN model_versions m ON m.id = p.model_version_id AND m.is_active
WHERE p.game_id IN (SELECT id FROM g) AND p.target = 'PLAYS';

CREATE TEMP VIEW lines AS
SELECT p.game_id, p.player_id, t.code AS team, p.target::text AS target,
       p.predicted_value AS median,
       COALESCE(p.expected_value, p.predicted_value) AS expected,
       pl.p_plays, o.actual_value AS actual, o.did_not_appear,
       (p.quantiles ->> '0.1')::float AS lo, (p.quantiles ->> '0.9')::float AS hi
FROM predictions p
JOIN prediction_outcomes o ON o.prediction_id = p.id
JOIN model_versions m ON m.id = p.model_version_id AND m.is_active
JOIN player_feature_snapshots s ON s.id = p.player_feature_snapshot_id
JOIN teams t ON t.id = s.team_id
LEFT JOIN plays pl ON pl.game_id = p.game_id AND pl.player_id = p.player_id
WHERE p.game_id IN (SELECT id FROM g) AND p.target <> 'PLAYS';

CREATE TEMP VIEW squads AS
SELECT game_id, team, target,
       sum(median)                          AS median_sum,
       sum(expected * COALESCE(p_plays, 1)) AS expected_sum,
       sum(actual)                          AS actual_sum
FROM lines GROUP BY game_id, team, target;

SELECT 'round ' || :round AS round, count(*) AS games_scored FROM g;

-- 1. Players who played: error per stat, and how often the 80% range held.
SELECT target,
       count(*) AS players,
       round(avg(abs(median - actual))::numeric, 3)   AS mae_median,
       round(avg(median - actual)::numeric, 3)        AS bias_median,
       round(avg(abs(expected - actual))::numeric, 3) AS mae_expected,
       round(avg(expected - actual)::numeric, 3)      AS bias_expected,
       round(100.0 * avg(CASE WHEN actual BETWEEN lo AND hi THEN 1 ELSE 0 END), 0) AS "in 80% range %",
       sum(CASE WHEN actual < lo THEN 1 ELSE 0 END) AS below,
       sum(CASE WHEN actual > hi THEN 1 ELSE 0 END) AS above
FROM lines WHERE NOT did_not_appear
GROUP BY target ORDER BY target;

-- 2. Squad totals per team-game: expected x P(plays) against the box score.
SELECT target,
       count(*) AS team_games,
       round(avg(expected_sum - actual_sum)::numeric, 2)      AS expected_bias,
       round(avg(abs(expected_sum - actual_sum))::numeric, 2) AS expected_error,
       round(avg(median_sum - actual_sum)::numeric, 2)        AS median_bias,
       round(avg(abs(median_sum - actual_sum))::numeric, 2)   AS median_error
FROM squads GROUP BY target ORDER BY target;

-- 3. Who plays: Brier score, and players used per team-game.
SELECT count(*) AS candidates,
       round(avg((p_plays - CASE WHEN did_not_appear THEN 0 ELSE 1 END) ^ 2)::numeric, 4) AS brier,
       round(100.0 * avg(CASE WHEN did_not_appear THEN 0 ELSE 1 END), 1) AS "played %",
       round((100.0 * avg(p_plays))::numeric, 1) AS "predicted %"
FROM plays;

SELECT count(*) AS team_games,
       round(avg(expected_used - used)::numeric, 2)      AS players_used_bias,
       round(avg(abs(expected_used - used))::numeric, 2) AS players_used_error
FROM (
    SELECT pl.game_id, s.team_id,
           sum(pl.p_plays) AS expected_used,
           sum(CASE WHEN pl.did_not_appear THEN 0 ELSE 1 END) AS used
    FROM plays pl
    JOIN predictions p ON p.game_id = pl.game_id AND p.player_id = pl.player_id AND p.target = 'PLAYS'
    JOIN model_versions m ON m.id = p.model_version_id AND m.is_active
    JOIN player_feature_snapshots s ON s.id = p.player_feature_snapshot_id
    GROUP BY pl.game_id, s.team_id
) per_team;

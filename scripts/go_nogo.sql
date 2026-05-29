-- GO / NO-GO gatekeeper (SQLite-friendly, NULL-guarded)
-- Usage:
--   export SINCE_ID=$(sqlite3 db/fie_prod.sqlite "SELECT MAX(id)-2000 FROM trades;")
--   sqlite3 db/fie_prod.sqlite <(sed "s/\\$SINCE_ID/$SINCE_ID/g" scripts/go_nogo.sql)
--
-- Notes:
-- - Works on EDGE zone: vol=mid, scen=on, p_model in [0.25, 0.40)
-- - Requires context fields to be present: is_prior, hour_weight, hour_utc, p_bucket

.headers on
.mode column

WITH base AS (
  SELECT *
  FROM trades
  WHERE id > $SINCE_ID
    AND pnl IS NOT NULL
    AND p_model IS NOT NULL
    AND is_prior IS NOT NULL
    AND regime_key IS NOT NULL
),
parsed AS (
  SELECT
    *,
    -- Use rk_* aliases to avoid name collisions with table columns (hour_utc exists in trades).
    CAST(substr(regime_key, 1, instr(regime_key, '|')-1) AS INTEGER) AS rk_hour,
    substr(
      regime_key,
      instr(regime_key,'|')+1,
      instr(substr(regime_key, instr(regime_key,'|')+1), '|')-1
    ) AS rk_vol_bucket,
    -- scen_bucket = everything after the 2nd '|'
    substr(
      substr(regime_key, instr(regime_key,'|')+1),
      instr(substr(regime_key, instr(regime_key,'|')+1), '|') + 1
    ) AS rk_scen_bucket
  FROM base
),
edge_zone AS (
  SELECT *
  FROM parsed
  WHERE rk_vol_bucket = 'mid'
    AND rk_scen_bucket = 'on'
    AND rk_hour BETWEEN 0 AND 23
    AND p_model >= 0.25 AND p_model < 0.40
),
labeled AS (
  SELECT *,
    CASE
      WHEN ROW_NUMBER() OVER (ORDER BY id) <= (COUNT(*) OVER())/2 THEN 'first'
      ELSE 'last'
    END AS part
  FROM edge_zone
),
metrics AS (
  SELECT
    COUNT(*) AS edge_n,
    SUM(CASE WHEN is_prior=0 THEN 1 ELSE 0 END) AS known_n,
    COUNT(DISTINCT p_bucket) AS n_buckets,

    AVG(CASE WHEN is_prior=0 THEN pnl END) AS edge_known,
    AVG(CASE WHEN is_prior=1 THEN pnl END) AS edge_prior,

    AVG(CASE WHEN hour_weight < 0.3 THEN pnl END) AS low_w,
    AVG(CASE WHEN hour_weight >= 0.6 THEN pnl END) AS high_w,

    AVG(CASE WHEN part='first' THEN pnl END) AS first_half,
    AVG(CASE WHEN part='last'  THEN pnl END) AS last_half,

    AVG(CASE WHEN part='first' THEN rk_hour END) AS first_hour,
    AVG(CASE WHEN part='last'  THEN rk_hour END) AS last_hour,

    AVG(pnl / NULLIF(hour_weight, 0)) AS pnl_per_unit
  FROM labeled
)
SELECT
  edge_n,
  known_n,
  n_buckets,
  edge_known,
  edge_prior,
  low_w,
  high_w,
  first_half,
  last_half,
  first_hour,
  last_hour,
  pnl_per_unit,
  CASE
    -- 1) Data sufficiency
    WHEN edge_n < 300 THEN 'NO-GO: not enough EDGE samples'
    WHEN known_n < 150 THEN 'NO-GO: not enough KNOWN samples'

    -- 2) Fake EDGE (single point)
    WHEN n_buckets IS NULL OR n_buckets <= 1 THEN 'NO-GO: single p_bucket (quantized edge)'

    -- 3) No advantage vs prior
    WHEN edge_known IS NULL OR edge_prior IS NULL THEN 'NO-GO: missing known/prior means (need both)'
    WHEN edge_known <= edge_prior THEN 'NO-GO: no edge vs prior'

    -- 4) Weight monotonicity (only if both buckets exist)
    WHEN high_w IS NULL OR low_w IS NULL THEN 'NO-GO: insufficient low/high weight buckets'
    WHEN high_w <= low_w THEN 'NO-GO: no weight monotonicity'

    -- 5) Drift / non-stationarity
    WHEN first_half IS NULL OR last_half IS NULL OR edge_known IS NULL THEN 'NO-GO: insufficient stability split'
    WHEN ABS(first_half - last_half) > ABS(edge_known) THEN 'NO-GO: unstable (drift)'

    -- 6) Distribution shift: comparing different hours
    WHEN first_hour IS NULL OR last_hour IS NULL THEN 'NO-GO: insufficient distribution split'
    WHEN ABS(first_hour - last_hour) > 2 THEN 'NO-GO: distribution shift (hours changed)'

    -- 7) Normalized edge must be positive
    WHEN pnl_per_unit IS NULL THEN 'NO-GO: pnl_per_unit is NULL'
    WHEN pnl_per_unit <= 0 THEN 'NO-GO: no normalized edge'

    ELSE 'GO'
  END AS verdict
FROM metrics;


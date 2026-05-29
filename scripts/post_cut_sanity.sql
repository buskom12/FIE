-- Post-cut sanity checks for regime_key integrity
-- Usage:
--   export CUT_ID=<id>   # id at restart moment
--   sqlite3 db/fie_prod.sqlite <(sed "s/\\$CUT_ID/$CUT_ID/g" scripts/post_cut_sanity.sql)

.headers on
.mode column

WITH w AS (
  SELECT *
  FROM trades
  WHERE id > $CUT_ID
)
SELECT
  'post_cut_regime_key' AS section,
  COUNT(*) AS total,
  SUM(CASE WHEN regime_key IS NULL THEN 1 ELSE 0 END) AS nulls,
  SUM(CASE WHEN regime_key='UNKNOWN|unknown|unknown' THEN 1 ELSE 0 END) AS unknowns,
  ROUND(
    (SUM(CASE WHEN regime_key='UNKNOWN|unknown|unknown' THEN 1 ELSE 0 END) * 1.0)
    / NULLIF(COUNT(*), 0),
    4
  ) AS unknown_share
FROM w;

-- Optional: show top keys in the post-cut window
SELECT
  'post_cut_top_regime_key' AS section,
  regime_key,
  COUNT(*) AS n
FROM trades
WHERE id > $CUT_ID
GROUP BY regime_key
ORDER BY n DESC
LIMIT 10;


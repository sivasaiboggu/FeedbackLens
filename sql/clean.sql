CREATE TABLE reviews AS
WITH normalized AS (
  SELECT review_id, trim(text) AS text, sentiment, lower(trim(text)) AS fingerprint
  FROM raw WHERE length(trim(text)) > 0 AND sentiment IN (0,1)
), ranked AS (
  SELECT *, row_number() OVER (PARTITION BY fingerprint ORDER BY review_id) AS duplicate_rank
  FROM normalized
)
SELECT review_id,text,sentiment FROM ranked WHERE duplicate_rank=1;

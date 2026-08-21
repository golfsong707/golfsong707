-- Audit log of every row the cleaning step had to repair or flag.
DROP TABLE IF EXISTS price_anomalies;

CREATE TABLE price_anomalies AS
SELECT
    d.date,
    c.ticker,
    c.company_name,
    s.sector_name,
    f.open,
    f.high,
    f.low,
    f.close,
    f.volume,
    f.return_1d,
    f.quality_flag
FROM fact_daily_price f
JOIN dim_date d    ON f.date_id = d.date_id
JOIN dim_company c ON f.company_id = c.company_id
JOIN dim_sector s  ON c.sector_id = s.sector_id
WHERE f.quality_flag IS NOT NULL
  AND f.quality_flag != '';

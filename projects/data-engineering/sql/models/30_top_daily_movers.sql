-- The 25 largest single-day price moves (by absolute return) in the dataset.
DROP TABLE IF EXISTS top_daily_movers;

CREATE TABLE top_daily_movers AS
SELECT
    d.date,
    c.ticker,
    c.company_name,
    s.sector_name,
    ROUND(f.return_1d, 6) AS return_1d
FROM fact_daily_price f
JOIN dim_date d    ON f.date_id = d.date_id
JOIN dim_company c ON f.company_id = c.company_id
JOIN dim_sector s  ON c.sector_id = s.sector_id
WHERE f.return_1d IS NOT NULL
ORDER BY ABS(f.return_1d) DESC
LIMIT 25;

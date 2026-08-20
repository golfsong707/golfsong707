-- Quarterly performance by sector: average daily return, volume and
-- cumulative (summed) daily return.
DROP TABLE IF EXISTS sector_performance;

CREATE TABLE sector_performance AS
SELECT
    d.year,
    d.quarter,
    s.sector_name,
    COUNT(*)                            AS obs,
    ROUND(AVG(f.return_1d), 6)          AS avg_daily_return,
    ROUND(SUM(f.volume), 0)             AS total_volume,
    ROUND(SUM(f.return_1d), 6)          AS cumulative_return
FROM fact_daily_price f
JOIN dim_date d    ON f.date_id = d.date_id
JOIN dim_company c ON f.company_id = c.company_id
JOIN dim_sector s  ON c.sector_id = s.sector_id
WHERE f.return_1d IS NOT NULL
GROUP BY d.year, d.quarter, s.sector_name;

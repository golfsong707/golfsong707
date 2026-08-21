-- Wide, analytics-ready daily view joining the price fact to its dimensions.
DROP TABLE IF EXISTS daily_returns_enriched;

CREATE TABLE daily_returns_enriched AS
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
    f.sma_20,
    f.sma_50,
    f.vol_20,
    f.quality_flag
FROM fact_daily_price f
JOIN dim_date d    ON f.date_id = d.date_id
JOIN dim_company c ON f.company_id = c.company_id
JOIN dim_sector s  ON c.sector_id = s.sector_id;

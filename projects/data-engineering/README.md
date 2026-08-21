# 🏗️ MarketScope — Equity Data Pipeline & Analytics Warehouse

**Category:** Data Engineering · **Stack:** Python (pandas, NumPy) · SQLite · SQL

An end-to-end ETL pipeline that turns messy, semi-realistic market data into a
clean, queryable star-schema warehouse, then layers dbt-style SQL models on top
for analytics-ready tables.

## What it does

1. **Generate** raw CSVs (`companies`, `daily_prices`, `fundamentals`) with
   realistic data-quality problems baked in — missing closes, duplicate rows,
   impossible OHLC (`low > high`), negative volumes and price spikes.
2. **Extract** the raw files and surface their shape.
3. **Transform** — clean, validate and enrich:
   - de-duplicate and normalise company records
   - repair negative volumes and impossible OHLC, forward-fill missing closes
   - recompute missing EPS from net income ÷ shares outstanding
   - derive `return_1d`, `sma_10/20/50`, `vol_20` and valuation metrics
   - tag every repaired/flagged row with an auditable `quality_flag`
4. **Load** into a SQLite **star-schema warehouse** (dimensions + facts).
5. **Model** — run versioned SQL models to produce analytics tables
   (`sector_performance`, `daily_returns_enriched`, `top_daily_movers`,
   `price_anomalies`).

## Quickstart

```bash
cd projects/data-engineering
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python run_pipeline.py      # generate -> extract -> transform -> load -> model
pytest -q                   # unit tests for the cleaning rules
```

Inspect the warehouse afterwards:

```bash
sqlite3 warehouse/marketscope.db "SELECT * FROM sector_performance LIMIT 5;"
```

## Architecture

```
raw CSV files ──▶ extract ──▶ transform (clean/validate/enrich) ──▶ load ──▶ SQL models
                                                                      │            │
                                                                      ▼            ▼
                                                            star schema        analytics tables
```

## Schema

| Table              | Grain                          | Key columns |
| ------------------ | ------------------------------ | ----------- |
| `dim_sector`       | one row per sector             | `sector_id`, `sector_name` |
| `dim_date`         | one row per trading day        | `date_id` (YYYYMMDD), `year`, `quarter` |
| `dim_company`      | one row per company            | `company_id`, `ticker`, `sector_id` |
| `fact_daily_price` | one row per company per day    | `date_id`, `company_id`, OHLCV, `return_1d`, SMAs, `vol_20`, `quality_flag` |
| `fact_fundamentals`| one row per company per quarter| `revenue`, `net_income`, `eps`, `market_cap`, `pe_ratio` |

## Data quality report (example run)

| Metric | Value |
| ------ | ----- |
| duplicate rows removed | 6 (prices), 1 (companies), 2 (fundamentals) |
| negative volumes repaired | 5 |
| impossible OHLC repaired | 8 |
| missing closes forward-filled | 281 |
| missing EPS recomputed | 3 |
| outlier returns flagged | 8 |

## File layout

```
src/
  generate_raw.py   # synthetic data with injected quality issues
  extract.py        # read raw CSVs
  transform.py      # cleaning, validation & enrichment rules
  load.py           # star-schema load into SQLite
  build_models.py   # runs sql/models/*.sql over the warehouse
  pipeline.py       # orchestrator
sql/models/         # dbt-style SQL models (analytics layer)
tests/              # pytest unit tests
```

## Key engineering ideas

- **Auditable fixes** — every repair is recorded in `quality_flag`, so the
  `price_anomalies` table doubles as a data-quality audit log.
- **Star schema** — facts and dimensions separated with surrogate integer keys
  and foreign keys enforced.
- **Versioned SQL models** — analytics logic lives in plain SQL files executed
  in order, mirroring a dbt project without the dependency.
- **Reproducible** — a fixed seed means the raw data (and every downstream
  table) can be regenerated deterministically.

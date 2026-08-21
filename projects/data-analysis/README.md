# 📈 MarketScope — Market Performance & Portfolio Analytics

**Category:** Data Analysis / BI · **Stack:** Python · pandas · matplotlib · seaborn · Streamlit

Business-intelligence analytics over the MarketScope equity universe, with an
interactive **Streamlit dashboard**. It reads clean data from the
**Data Engineering** warehouse (and falls back to synthetic data standalone).

## Deliverables

| Deliverable | Where |
| ----------- | ----- |
| Interactive dashboard | `streamlit run app/dashboard.py` |
| Static report (CSVs + console summary) | `python run_analysis.py` |
| Exploratory notebook | `notebooks/01_exploratory_analysis.ipynb` |

## What's measured

- **Headline market stats** — total & annualized return, volatility, Sharpe
  ratio, max drawdown.
- **Per-ticker KPIs** — total return, annualized return/vol, Sharpe, max
  drawdown, grouped by sector.
- **Sector indices** — equal-weight cumulative performance (rebased to 1.0).
- **Correlations** — daily-return correlation at sector or ticker level.
- **Top / bottom movers** — best and worst performers over a rolling window.

## Quickstart

```bash
cd projects/data-analysis
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

streamlit run app/dashboard.py   # interactive dashboard
python run_analysis.py           # static report to reports/
```

The dashboard sidebar lets you filter by **sector** and **date range**; every
chart and KPI recomputes on the filtered universe.

## Dashboard views

1. **KPI cards** — market total return, annualized return, volatility, Sharpe,
   max drawdown.
2. **Cumulative sector performance** — line chart of equal-weight sector
   indices.
3. **Performance table + correlation heatmap** — per-ticker KPIs (styled,
   colour-graded) alongside the return-correlation matrix.
4. **Top/bottom movers** — bar chart of the best and worst performers over a
   configurable window.

## Methodology notes

- **Equal-weight indices** are built from prices rebased to 1.0 at the window
  start (the standard construction), not from averaging arithmetic returns,
  which would inflate results.
- **Annualization** uses 252 trading days per year; Sharpe assumes a 0% risk-
  free rate.
- Returns are **illustrative** — the universe is synthetic and simulated for
  demonstration.

## File layout

```
src/
  make_dataset.py   # load from DE warehouse or generate synthetic data
  analysis.py       # KPIs, indices, correlations, movers
app/
  dashboard.py      # Streamlit dashboard
notebooks/          # exploratory walkthrough
run_analysis.py     # static CSV report generator
```

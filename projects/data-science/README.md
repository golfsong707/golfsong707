# 🔮 Next-Day Stock Price Direction — ML Modelling

**Category:** Data Science · **Stack:** Python · scikit-learn · pandas · matplotlib

A supervised-learning project that predicts whether a stock's next-day close
will be higher than today's close, using features engineered strictly from past
data. It reads clean data from the **Data Engineering** warehouse (and falls
back to synthetic data when run standalone).

## Pipeline

```
data ──▶ feature engineering ──▶ time-ordered split ──▶ train 4 models ──▶ evaluate
```

## Features (all computed at time `t`, no look-ahead)

| Group | Features |
| ----- | -------- |
| Returns / momentum | `ret_1`, `ret_5`, `ret_10`, `ret_20` |
| Price vs moving averages | `sma_ratio_10/20/50` |
| Momentum oscillator | `rsi_14` |
| Risk / volume | `vol_20`, `volume_z`, `range_pct`, `gap` |
| Calendar | `day_of_week` |

## Models

- **Majority-class baseline** (sanity check — any model must beat this)
- **Logistic regression** (scaled)
- **Random forest**
- **Gradient boosting**

## Methodology highlights

- **Leakage-free split** — the test set is the last 20% of calendar dates, so
  the model is always evaluated on data strictly in the future.
- **Honest baseline** — every model is benchmarked against "always predict up".
- **Multiple metrics** — accuracy, precision, recall, F1 and ROC-AUC, plus
  ROC curves, a confusion matrix and feature importances.

## Quickstart

```bash
cd projects/data-science
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python run_all.py            # data -> features -> train -> evaluate
# or explore interactively
jupyter notebook notebooks/01_modeling_walkthrough.ipynb
```

Outputs are written to `outputs/` (metrics, predictions, plots) and trained
models are pickled to `models/`.

## Results & interpretation

| Model | Accuracy | ROC-AUC |
| ----- | -------- | ------- |
| majority baseline | ~0.50 | 0.50 |
| logistic regression | ~0.50 | ~0.49 |
| random forest | ~0.49 | ~0.49 |
| gradient boosting | ~0.50 | ~0.49 |

Direction on this simulated market is close to a coin flip, so accuracy sits at
chance level for every model — **the expected and honest result** for a
(near-)efficient market. The project deliberately showcases the *discipline*:
a correct time-series split, a proper baseline, and transparent evaluation are
what separate a real modelling workflow from a misleading backtest.

> ⚠️ **Disclaimer** — this is a portfolio demonstration using synthetic data.
> Nothing here is investment advice, and the methodology is not a trading
> signal.

## File layout

```
src/
  make_dataset.py   # load from DE warehouse or generate synthetic data
  features.py       # feature engineering + time split
  train.py          # model training & metrics
  evaluate.py       # ROC / confusion matrix / feature importance plots
notebooks/          # end-to-end walkthrough notebook
run_all.py          # one-command pipeline
```

"""Interactive MarketScope analytics dashboard (Streamlit).

Run from the project root:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    compute_kpis,
    correlation_matrix,
    headline,
    sector_index,
    top_movers,
)
from src.make_dataset import load_market_data  # noqa: E402

st.set_page_config(page_title="MarketScope Analytics", layout="wide")


@st.cache_data
def get_prices() -> pd.DataFrame:
    return load_market_data()


prices = get_prices()

st.title("📊 MarketScope — Market Performance & Portfolio Analytics")
st.caption("Equal-weight performance analytics over the MarketScope equity universe.")

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
all_sectors = sorted(prices["sector"].unique())
sel_sectors = st.sidebar.multiselect("Sectors", all_sectors, default=all_sectors)

min_date = prices["date"].min().date()
max_date = prices["date"].max().date()
date_range = st.sidebar.slider(
    "Date range", min_value=min_date, max_value=max_date, value=(min_date, max_date)
)

window = st.sidebar.slider("Movers window (trading days)", 5, 120, 30, step=5)

filtered = prices[
    prices["sector"].isin(sel_sectors)
    & prices["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
stats = headline(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total return", f"{stats['total_return']:.1%}")
c2.metric("Annualized return", f"{stats['annualized_return']:.1%}")
c3.metric("Annualized volatility", f"{stats['annualized_vol']:.1%}")
c4.metric("Sharpe ratio", f"{stats['sharpe']:.2f}")
c5.metric("Max drawdown", f"{stats['max_drawdown']:.1%}")

st.divider()

# --------------------------------------------------------------------------- #
# Cumulative sector performance
# --------------------------------------------------------------------------- #
st.subheader("Cumulative sector performance (equal-weight, base = 1.0)")
curve = sector_index(filtered)[sorted(set(filtered["sector"]))]
st.line_chart(curve)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Performance by ticker")
    kpis = compute_kpis(filtered)
    styled = kpis.style.format(
        {
            "total_return": "{:.1%}",
            "annualized_return": "{:.1%}",
            "annualized_vol": "{:.1%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.1%}",
        }
    ).background_gradient(subset=["total_return"], cmap="RdYlGn")
    st.dataframe(styled, height=430)

with col_right:
    st.subheader("Correlation of daily returns")
    level = "ticker" if len(sel_sectors) == 1 else "sector"
    corr = correlation_matrix(filtered, level=level)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    annot = corr.shape[0] <= 12
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Daily-return correlation ({level} level)")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader(f"Top / bottom movers (last {window} trading days)")
movers = top_movers(filtered, window=window)
movers["return_pct"] = movers["return_pct"]
chart = movers.set_index("ticker")["return_pct"]
st.bar_chart(chart)

st.caption("Source: MarketScope synthetic universe · cleaned via the data-engineering pipeline.")

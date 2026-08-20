"""Core analytics: KPIs, indices, correlations and movers."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_kpis(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker performance KPIs over the supplied window."""
    p = prices.sort_values(["ticker", "date"])
    g = p.groupby("ticker")["close"]

    first = g.first()
    last = g.last()
    n = g.count()
    ret = g.pct_change()

    total_return = last / first - 1
    ann_return = (1 + total_return) ** (TRADING_DAYS / n) - 1
    ann_vol = ret.groupby(p["ticker"]).std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol

    drawdown = p["close"] / g.cummax() - 1
    max_dd = drawdown.groupby(p["ticker"]).min()

    sector = p.drop_duplicates("ticker").set_index("ticker")["sector"]

    df = pd.DataFrame(
        {
            "ticker": first.index,
            "sector": [sector.get(t) for t in first.index],
            "total_return": total_return.values,
            "annualized_return": ann_return.values,
            "annualized_vol": ann_vol.values,
            "sharpe": sharpe.values,
            "max_drawdown": max_dd.values,
        }
    )
    return df.sort_values("total_return", ascending=False).reset_index(drop=True)


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    return p


def _rebased(prices: pd.DataFrame) -> pd.DataFrame:
    """Rebase each ticker's close to 1.0 at the start of the supplied window."""
    p = prices.sort_values(["ticker", "date"]).copy()
    p["norm"] = p["close"] / p.groupby("ticker")["close"].transform("first")
    return p


def sector_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight index per sector from rebased prices (base = 1.0)."""
    p = _rebased(prices)
    return p.groupby(["date", "sector"])["norm"].mean().unstack("sector")


def market_index(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight market-wide index from rebased prices (base = 1.0)."""
    p = _rebased(prices)
    return p.groupby("date")["norm"].mean()


def correlation_matrix(prices: pd.DataFrame, level: str = "sector") -> pd.DataFrame:
    """Return correlation matrix of daily returns at sector or ticker level."""
    p = _daily_returns(prices)
    if level == "sector":
        wide = p.groupby(["date", "sector"])["ret"].mean().unstack("sector")
    else:
        wide = p.pivot_table(index="date", columns="ticker", values="ret")
    return wide.corr()


def top_movers(prices: pd.DataFrame, n: int = 10, window: int = 30) -> pd.DataFrame:
    """Best and worst performers over the last ``window`` trading days."""
    p = prices.sort_values("date")
    last = p.groupby("ticker").tail(window)
    first = last.groupby("ticker")["close"].first()
    lastclose = last.groupby("ticker")["close"].last()
    perf = (lastclose / first - 1) * 100
    perf = perf.sort_values()
    movers = pd.DataFrame({"ticker": perf.index, "return_pct": perf.values})
    worst, best = movers.head(n), movers.tail(n).iloc[::-1]
    return pd.concat(
        [worst.assign(group="worst"), best.assign(group="best")], ignore_index=True
    )


def headline(prices: pd.DataFrame) -> dict[str, float]:
    """Headline market stats for the supplied window."""
    mkt = market_index(prices)
    total = mkt.iloc[-1] - 1
    ann = (1 + total) ** (TRADING_DAYS / len(mkt)) - 1
    vol = mkt.pct_change().std() * np.sqrt(TRADING_DAYS)
    dd = (mkt / mkt.cummax() - 1).min()
    return {
        "total_return": total,
        "annualized_return": ann,
        "annualized_vol": vol,
        "sharpe": ann / vol,
        "max_drawdown": dd,
    }

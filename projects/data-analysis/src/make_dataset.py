"""Prepare the analytics dataset.

Sources clean, enriched data from the Data Engineering warehouse
(``projects/data-engineering/warehouse/marketscope.db``). If that database is
not present, an equivalent deterministic synthetic universe is generated so the
project is fully self-contained.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 707
RNG = np.random.default_rng(SEED)
START = "2021-01-04"
END = "2023-12-29"

SECTORS: dict[str, list[str]] = {
    "Technology": ["APEX", "NOVA", "QUANT", "VECTOR"],
    "Finance": ["MERIDIAN", "CITADEL", "HARBOR", "SUMMIT"],
    "Healthcare": ["HELIX", "GENOME", "PULSE", "VITAE"],
    "Energy": ["TITAN", "VOLT", "HELIOS", "STRATA"],
    "Consumer": ["LUMINA", "ORBIT", "NESTA", "PURE"],
    "Industrials": ["FORGE", "AXON", "GRID", "TURBINE"],
}

DE_WAREHOUSE = (
    Path(__file__).resolve().parents[3]
    / "projects"
    / "data-engineering"
    / "warehouse"
    / "marketscope.db"
)


def from_warehouse() -> pd.DataFrame | None:
    if not DE_WAREHOUSE.exists():
        return None
    conn = sqlite3.connect(DE_WAREHOUSE)
    try:
        df = pd.read_sql_query(
            "SELECT date, ticker, company_name, sector_name AS sector, "
            "       open, high, low, close, volume "
            "FROM daily_returns_enriched "
            "ORDER BY ticker, date",
            conn,
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _generate() -> pd.DataFrame:
    days = pd.bdate_range(START, END)
    n_days = len(days)

    tickers, sectors = [], []
    for sector, tks in SECTORS.items():
        for t in tks:
            tickers.append(t)
            sectors.append(sector)

    n = len(tickers)
    mu = RNG.uniform(0.04, 0.16, size=n)
    beta = RNG.uniform(0.7, 1.3, size=n)
    idio_vol = RNG.uniform(0.008, 0.020, size=n)
    start_price = RNG.uniform(10.0, 300.0, size=n)

    mkt = RNG.normal(0.0003, 0.010, size=n_days)
    idio = RNG.normal(0.0, 1.0, size=(n_days, n))
    ret = (mu / 252.0)[None, :] + beta[None, :] * mkt[:, None] + idio_vol[None, :] * idio
    close = start_price[None, :] * np.exp(np.cumsum(ret, axis=0))

    frames = []
    for j, ticker in enumerate(tickers):
        c = close[:, j]
        open_ = np.empty(n_days)
        open_[0] = c[0]
        open_[1:] = c[:-1] * (1 + RNG.normal(0.0, 0.003, size=n_days - 1))
        hi = np.maximum(open_, c) * (1 + np.abs(RNG.normal(0.0, 0.004, size=n_days)))
        lo = np.minimum(open_, c) * (1 - np.abs(RNG.normal(0.0, 0.004, size=n_days)))
        base = RNG.uniform(500_000, 20_000_000, size=1)[0]
        ret_abs = np.abs(np.diff(np.log(c), prepend=np.log(c[0])))
        volume = RNG.lognormal(mean=np.log(base), sigma=0.5, size=n_days) * (1 + 20 * ret_abs)
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "sector": sectors[j],
                    "company_name": ticker,
                    "open": np.round(open_, 2),
                    "high": np.round(hi, 2),
                    "low": np.round(lo, 2),
                    "close": np.round(c, 2),
                    "volume": np.round(volume).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_market_data() -> pd.DataFrame:
    df = from_warehouse()
    if df is None:
        print("[make_dataset] DE warehouse not found; generating synthetic clean data.")
        df = _generate()
    else:
        print(f"[make_dataset] loaded {len(df)} rows from the DE warehouse.")
    return df

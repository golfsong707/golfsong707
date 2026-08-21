"""Load stage: write cleaned data into a SQLite star-schema warehouse.

Schema
------
dim_sector        (sector_id, sector_name)
dim_date          (date_id, date, year, quarter, month, day_of_week)
dim_company       (company_id, ticker, company_name, sector_id, exchange, country, listed_date)
fact_daily_price  (date_id, company_id, open, high, low, close, volume,
                   return_1d, sma_10, sma_20, sma_50, vol_20, quality_flag)
fact_fundamentals (company_id, period_end, revenue, net_income, eps,
                   shares_outstanding, market_cap, pe_ratio)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

WAREHOUSE = Path(__file__).resolve().parents[1] / "warehouse" / "marketscope.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_sector (
    sector_id   INTEGER PRIMARY KEY,
    sector_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id     INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    year        INTEGER,
    quarter     INTEGER,
    month       INTEGER,
    day_of_week INTEGER
);

CREATE TABLE IF NOT EXISTS dim_company (
    company_id   INTEGER PRIMARY KEY,
    ticker       TEXT UNIQUE NOT NULL,
    company_name TEXT,
    sector_id    INTEGER,
    exchange     TEXT,
    country      TEXT,
    listed_date  TEXT,
    FOREIGN KEY (sector_id) REFERENCES dim_sector (sector_id)
);

CREATE TABLE IF NOT EXISTS fact_daily_price (
    date_id      INTEGER,
    company_id   INTEGER,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       INTEGER,
    return_1d    REAL,
    sma_10       REAL,
    sma_20       REAL,
    sma_50       REAL,
    vol_20       REAL,
    quality_flag TEXT,
    PRIMARY KEY (date_id, company_id),
    FOREIGN KEY (date_id)    REFERENCES dim_date (date_id),
    FOREIGN KEY (company_id) REFERENCES dim_company (company_id)
);

CREATE TABLE IF NOT EXISTS fact_fundamentals (
    company_id          INTEGER,
    period_end          TEXT,
    revenue             REAL,
    net_income          REAL,
    eps                 REAL,
    shares_outstanding  INTEGER,
    market_cap          REAL,
    pe_ratio            REAL,
    PRIMARY KEY (company_id, period_end),
    FOREIGN KEY (company_id) REFERENCES dim_company (company_id)
);
"""


def _connect(db_path: str | Path = WAREHOUSE) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _native(v) -> object:
    """Convert a value to a native Python type sqlite3 can bind safely."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    return v


def _rows(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Build bind-parameter tuples with native Python values."""
    return [
        tuple(_native(v) for v in row)
        for row in df[cols].itertuples(index=False, name=None)
    ]


def load(
    companies: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    db_path: str | Path = WAREHOUSE,
) -> Path:
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA)

        # dim_sector
        sectors = companies[["sector"]].drop_duplicates().reset_index(drop=True)
        sectors["sector_id"] = range(1, len(sectors) + 1)
        conn.executemany(
            "INSERT OR REPLACE INTO dim_sector (sector_id, sector_name) VALUES (?, ?)",
            _rows(sectors, ["sector_id", "sector"]),
        )

        # dim_date (from the union of price dates)
        dates = pd.DataFrame({"date": pd.to_datetime(prices["date"])}).drop_duplicates()
        dates["date_id"] = dates["date"].dt.strftime("%Y%m%d").astype(int)
        dates["year"] = dates["date"].dt.year
        dates["quarter"] = dates["date"].dt.quarter
        dates["month"] = dates["date"].dt.month
        dates["day_of_week"] = dates["date"].dt.dayofweek
        conn.executemany(
            "INSERT OR REPLACE INTO dim_date (date_id, date, year, quarter, month, day_of_week) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _rows(dates, ["date_id", "date", "year", "quarter", "month", "day_of_week"]),
        )

        # dim_company
        comp = companies.merge(sectors, on="sector", how="left")
        comp["listed_date"] = pd.to_datetime(comp["listed_date"]).dt.strftime("%Y-%m-%d")
        conn.executemany(
            "INSERT OR REPLACE INTO dim_company "
            "(company_id, ticker, company_name, sector_id, exchange, country, listed_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            _rows(
                comp,
                ["company_id", "ticker", "company_name", "sector_id", "exchange", "country", "listed_date"],
            ),
        )

        # fact_daily_price
        prices = prices.copy()
        prices["date_id"] = pd.to_datetime(prices["date"]).dt.strftime("%Y%m%d").astype(int)
        prices["company_id"] = prices["ticker"].map(
            dict(zip(companies["ticker"], companies["company_id"]))
        )
        price_cols = [
            "date_id", "company_id", "open", "high", "low", "close", "volume",
            "return_1d", "sma_10", "sma_20", "sma_50", "vol_20", "quality_flag",
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO fact_daily_price "
            "(date_id, company_id, open, high, low, close, volume, "
            " return_1d, sma_10, sma_20, sma_50, vol_20, quality_flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _rows(prices, price_cols),
        )

        # fact_fundamentals
        fundamentals = fundamentals.copy()
        fundamentals["company_id"] = fundamentals["ticker"].map(
            dict(zip(companies["ticker"], companies["company_id"]))
        )
        fundamentals["period_end"] = pd.to_datetime(
            fundamentals["period_end"]
        ).dt.strftime("%Y-%m-%d")
        fund_cols = [
            "company_id", "period_end", "revenue", "net_income", "eps",
            "shares_outstanding", "market_cap", "pe_ratio",
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO fact_fundamentals "
            "(company_id, period_end, revenue, net_income, eps, shares_outstanding, market_cap, pe_ratio) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _rows(fundamentals, fund_cols),
        )

        conn.commit()
    finally:
        conn.close()
    return Path(db_path)


def table_counts(db_path: str | Path = WAREHOUSE) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        tables = [
            "dim_sector", "dim_date", "dim_company",
            "fact_daily_price", "fact_fundamentals",
        ]
        counts = {}
        for t in tables:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return counts
    finally:
        conn.close()

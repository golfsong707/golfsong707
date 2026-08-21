"""Generate raw market data with embedded data-quality issues.

Simulates a small equity universe (24 tickers across 6 sectors) over roughly
three years of trading days and writes three raw CSV files to ``data/raw/``:

* ``companies.csv``     - company master data
* ``daily_prices.csv``  - daily OHLCV prices
* ``fundamentals.csv``  - quarterly fundamentals

The raw files intentionally contain realistic data-quality problems (missing
values, duplicate rows, impossible OHLC, negative volumes and price spikes) so
the downstream transform stage has genuine cleaning and validation work to do.
"""

from __future__ import annotations

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

# ticker -> (company name, exchange, country)
COMPANY_META: dict[str, tuple[str, str, str]] = {
    "APEX": ("Apex Computing Corp.", "NASDAQ", "USA"),
    "NOVA": ("NovaSoft Solutions", "NASDAQ", "USA"),
    "QUANT": ("Quantum Logic Systems", "NYSE", "USA"),
    "VECTOR": ("Vector Dynamics", "NYSE", "USA"),
    "MERIDIAN": ("Meridian Capital Group", "NYSE", "USA"),
    "CITADEL": ("Citadel Trust Bank", "NYSE", "USA"),
    "HARBOR": ("Harborline Financial", "LSE", "UK"),
    "SUMMIT": ("Summit Asset Management", "NASDAQ", "USA"),
    "HELIX": ("Helix Therapeutics", "NASDAQ", "USA"),
    "GENOME": ("Genome Analytics", "NASDAQ", "USA"),
    "PULSE": ("Pulse Medical Devices", "NYSE", "USA"),
    "VITAE": ("Vitae Pharmaceuticals", "LSE", "UK"),
    "TITAN": ("Titan Energy Partners", "NYSE", "USA"),
    "VOLT": ("Volt Grid Utilities", "NYSE", "USA"),
    "HELIOS": ("Helios Solar Group", "NASDAQ", "USA"),
    "STRATA": ("Strata Resources", "TSX", "Canada"),
    "LUMINA": ("Lumina Retail Group", "NYSE", "USA"),
    "ORBIT": ("Orbit Consumer Goods", "NASDAQ", "USA"),
    "NESTA": ("Nesta Foods", "LSE", "UK"),
    "PURE": ("Pureline Beverages", "NASDAQ", "USA"),
    "FORGE": ("Forge Manufacturing", "NYSE", "USA"),
    "AXON": ("Axon Logistics", "NYSE", "USA"),
    "GRID": ("Grid Infrastructure", "TSX", "Canada"),
    "TURBINE": ("Turbine Engineering", "NASDAQ", "USA"),
}


def _build_universe() -> pd.DataFrame:
    rows = []
    company_id = 0
    for sector, tickers in SECTORS.items():
        for ticker in tickers:
            name, exchange, country = COMPANY_META[ticker]
            company_id += 1
            rows.append(
                {
                    "company_id": company_id,
                    "ticker": ticker,
                    "company_name": name,
                    "sector": sector,
                    "exchange": exchange,
                    "country": country,
                    "listed_date": "2018-01-02",
                }
            )
    return pd.DataFrame(rows)


def _simulate_prices(companies: pd.DataFrame) -> pd.DataFrame:
    """Simulate correlated daily OHLCV using a single market factor (GBM)."""
    days = pd.bdate_range(START, END)
    n_days = len(days)
    tickers = companies["ticker"].tolist()
    n = len(tickers)

    mu = RNG.uniform(0.04, 0.16, size=n)          # annualised drift
    beta = RNG.uniform(0.7, 1.3, size=n)          # market exposure
    idio_vol = RNG.uniform(0.008, 0.020, size=n)  # daily idiosyncratic vol
    start_price = RNG.uniform(10.0, 300.0, size=n)

    mkt = RNG.normal(0.0003, 0.010, size=n_days)  # shared market factor
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

        base_volume = RNG.uniform(500_000, 20_000_000, size=1)[0]
        ret_abs = np.abs(np.diff(np.log(c), prepend=np.log(c[0])))
        volume = RNG.lognormal(mean=np.log(base_volume), sigma=0.5, size=n_days)
        volume = volume * (1 + 20 * ret_abs)

        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": np.round(open_, 2),
                    "high": np.round(hi, 2),
                    "low": np.round(lo, 2),
                    "close": np.round(c, 2),
                    "volume": np.round(volume).astype(int),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _simulate_fundamentals(companies: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    quarters = pd.period_range("2021Q1", "2023Q4", freq="Q")

    base_rev = rng.uniform(500e6, 80e9, size=len(companies))
    margin = rng.uniform(0.03, 0.25, size=len(companies))
    growth = rng.uniform(0.01, 0.05, size=len(companies))
    shares = rng.uniform(100e6, 5e9, size=len(companies))

    rows = []
    for i, (_, comp) in enumerate(companies.iterrows()):
        for q, period in enumerate(quarters):
            rev = base_rev[i] * (1 + growth[i]) ** q * (1 + rng.normal(0, 0.05))
            net_income = rev * margin[i] * (1 + rng.normal(0, 0.10))
            eps = net_income / shares[i]
            rows.append(
                {
                    "ticker": comp["ticker"],
                    "period_end": period.end_time.date(),
                    "revenue": round(rev, 2),
                    "net_income": round(net_income, 2),
                    "eps": round(eps, 4),
                    "shares_outstanding": int(shares[i]),
                }
            )
    return pd.DataFrame(rows)


def _inject_issues(
    prices: pd.DataFrame, companies: pd.DataFrame, fundamentals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    n = len(prices)

    # 1) missing closes
    prices.loc[rng.choice(n, size=int(n * 0.015), replace=False), "close"] = np.nan

    # 2) low > high (impossible OHLC)
    bad = rng.choice(n, size=8, replace=False)
    swap = prices.loc[bad, ["low", "high"]].to_numpy()
    prices.loc[bad, "low"] = swap[:, 1]
    prices.loc[bad, "high"] = swap[:, 0]

    # 3) negative volume
    neg = rng.choice(n, size=5, replace=False)
    prices.loc[neg, "volume"] = -np.abs(prices.loc[neg, "volume"])

    # 4) price spikes (outliers)
    spike = rng.choice(n, size=4, replace=False)
    prices.loc[spike, "close"] = prices.loc[spike, "close"] * 3.0

    # 5) duplicate rows
    prices = pd.concat([prices, prices.sample(n=6, random_state=99)], ignore_index=True)

    # fundamentals issues
    f = fundamentals.copy()
    f.loc[f.sample(n=3, random_state=7).index, "eps"] = np.nan
    f.loc[f.sample(n=2, random_state=9).index, "revenue"] = -1.0
    f = pd.concat([f, f.sample(n=2, random_state=8)], ignore_index=True)

    # company issues
    comp = companies.copy()
    comp = pd.concat([comp, comp.sample(n=1, random_state=5)], ignore_index=True)
    comp.loc[comp["ticker"] == "HELIOS", "country"] = np.nan

    return prices, comp, f


def main(out_dir: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    companies = _build_universe()
    prices = _simulate_prices(companies)
    fundamentals = _simulate_fundamentals(companies)
    prices, companies, fundamentals = _inject_issues(prices, companies, fundamentals)

    paths = {
        "companies": out_dir / "companies.csv",
        "prices": out_dir / "daily_prices.csv",
        "fundamentals": out_dir / "fundamentals.csv",
    }
    companies.to_csv(paths["companies"], index=False)
    prices.to_csv(paths["prices"], index=False)
    fundamentals.to_csv(paths["fundamentals"], index=False)

    print(f"[generate] wrote raw files to {out_dir}/")
    for name, p in paths.items():
        print(f"[generate]   {p.name}: {len(companies if name == 'companies' else prices if name == 'prices' else fundamentals)} rows")
    return paths


if __name__ == "__main__":
    main()

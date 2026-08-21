"""Transform stage: clean, validate and enrich the raw data.

Each function takes the raw frame, applies deterministic cleaning rules,
records what was repaired (so the fixes are auditable) and returns the cleaned
frame plus a small quality report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Companies (dimension)
# --------------------------------------------------------------------------- #
def clean_companies(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    n_before = len(df)

    df = df.drop_duplicates(subset=["company_id", "ticker"])
    df["sector"] = df["sector"].str.strip().str.title()
    df["country"] = df["country"].fillna("USA")
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")

    report = {
        "rows_before": n_before,
        "rows_after": len(df),
        "duplicates_removed": n_before - len(df),
        "missing_country_filled": int(df["country"].isna().sum() == 0),
    }
    return df, report


# --------------------------------------------------------------------------- #
# Daily prices (fact)
# --------------------------------------------------------------------------- #
def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    n_before = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Capture issues before repairing (kept as columns so they survive row drops).
    neg_vol = (df["volume"] < 0).to_numpy()
    broken = (df["low"] > df["high"]).to_numpy()

    # Repair: negative volume -> 0
    df.loc[neg_vol, "volume"] = 0
    df["volume"] = df["volume"].fillna(0).astype("int64")

    # Repair: impossible OHLC -> swap low/high
    vals = df.loc[broken, ["low", "high"]].to_numpy()
    df.loc[broken, "low"] = vals[:, 1]
    df.loc[broken, "high"] = vals[:, 0]

    # Forward-fill missing closes within each ticker, then drop leading NaNs.
    df["close"] = df.groupby("ticker")["close"].ffill()
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Enrichment
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()
    df["sma_10"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(10).mean())
    df["sma_20"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(20).mean())
    df["sma_50"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(50).mean())
    df["vol_20"] = df.groupby("ticker")["return_1d"].transform(lambda s: s.rolling(20).std())

    outlier = df["return_1d"].abs() > 0.3

    df["quality_flag"] = ""
    df.loc[neg_vol, "quality_flag"] += "repaired_negative_volume;"
    df.loc[broken, "quality_flag"] += "repaired_low_high;"
    df.loc[outlier, "quality_flag"] += "outlier_return;"

    report = {
        "rows_before": n_before,
        "rows_after": len(df),
        "duplicates_removed": n_before - len(df) - int(pd.isna(raw["date"]).sum()),
        "negative_volumes_repaired": int(neg_vol.sum()),
        "bad_ohlc_repaired": int(broken.sum()),
        "missing_closes_filled": int(raw["close"].isna().sum()),
        "outlier_returns_flagged": int(outlier.sum()),
    }
    return df, report


# --------------------------------------------------------------------------- #
# Fundamentals (fact)
# --------------------------------------------------------------------------- #
def clean_fundamentals(
    raw: pd.DataFrame, cleaned_prices: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    n_before = len(df)

    df = df.drop_duplicates()
    n_dedup = len(df)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end"])

    neg_rev = df["revenue"] < 0
    df.loc[neg_rev, "revenue"] = np.nan
    df["revenue"] = df.groupby("ticker")["revenue"].ffill().bfill()
    df["net_income"] = df.groupby("ticker")["net_income"].ffill().bfill()

    # Recompute EPS where missing: EPS = net income / shares outstanding.
    df["eps"] = df["eps"].fillna(df["net_income"] / df["shares_outstanding"])
    df = df.dropna(subset=["eps"]).reset_index(drop=True)

    # Enrich with valuation metrics using the latest cleaned close per ticker.
    latest = cleaned_prices.sort_values("date").groupby("ticker")["close"].last()
    df["market_cap"] = df["ticker"].map(latest) * df["shares_outstanding"]
    df["pe_ratio"] = df["ticker"].map(latest) / df["eps"]
    df.loc[df["eps"] <= 0, "pe_ratio"] = np.nan

    report = {
        "rows_before": n_before,
        "rows_after": len(df),
        "duplicates_removed": n_before - n_dedup,
        "negative_revenues_repaired": int(neg_rev.sum()),
        "missing_eps_recomputed": int(raw["eps"].isna().sum()),
    }
    return df, report

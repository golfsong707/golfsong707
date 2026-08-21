"""Feature engineering for next-day price direction.

All features use only information available at (or before) time ``t`` so there
is no look-ahead leakage. The target is defined per ticker as
``close[t+1] > close[t]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "rsi_14", "vol_20", "volume_z", "range_pct", "gap", "day_of_week",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), 50.0)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"]).copy()

    g = df.groupby("ticker")
    close = df["close"]

    df["ret_1"] = g["close"].pct_change(1)
    df["ret_5"] = g["close"].pct_change(5)
    df["ret_10"] = g["close"].pct_change(10)
    df["ret_20"] = g["close"].pct_change(20)

    df["sma_ratio_10"] = close / g["close"].transform(lambda s: s.rolling(10).mean())
    df["sma_ratio_20"] = close / g["close"].transform(lambda s: s.rolling(20).mean())
    df["sma_ratio_50"] = close / g["close"].transform(lambda s: s.rolling(50).mean())

    df["rsi_14"] = df.groupby("ticker")["close"].transform(_rsi)

    df["vol_20"] = g["ret_1"].transform(lambda s: s.rolling(20).std())

    df["volume_z"] = (
        df["volume"] - g["volume"].transform(lambda s: s.rolling(20).mean())
    ) / g["volume"].transform(lambda s: s.rolling(20).std())

    df["range_pct"] = (df["high"] - df["low"]) / close
    df["gap"] = g["open"].pct_change(1)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Target: next-day direction (no leakage — uses close[t+1] only as the label).
    df["target"] = (g["close"].shift(-1) > close).astype(int)

    return df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)


def split_time(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split so the test set is strictly in the future."""
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    train = df[df["date"] <= cut]
    test = df[df["date"] > cut]
    return train, test

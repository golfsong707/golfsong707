"""Unit tests for the transform stage's cleaning rules."""

import pandas as pd
import pytest

from src.transform import clean_companies, clean_prices


def _raw_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05"],
            "ticker": ["APEX", "NOVA", "APEX", "NOVA"],
            "open": [100.0, 50.0, 101.0, 51.0],
            "high": [101.0, 51.0, 99.0, 52.0],  # third row is broken (low > high)
            "low": [99.0, 49.0, 102.0, 50.0],
            "close": [100.5, 50.5, 100.0, 51.5],
            "volume": [1000, 2000, 500, -3000],
        }
    )


def test_clean_prices_repairs_bad_ohlc():
    out, _ = clean_prices(_raw_prices())
    assert (out["low"] <= out["high"]).all()


def test_clean_prices_repairs_negative_volume():
    out, _ = clean_prices(_raw_prices())
    assert (out["volume"] >= 0).all()


def test_clean_prices_drops_duplicates():
    raw = pd.concat([_raw_prices(), _raw_prices()], ignore_index=True)
    out, report = clean_prices(raw)
    assert len(out) == 4
    assert report["duplicates_removed"] == 4


def test_clean_prices_flags_repaired_rows():
    out, _ = clean_prices(_raw_prices())
    flagged = out[out["quality_flag"].str.contains("repaired")]
    assert len(flagged) == 2  # one broken OHLC + one negative volume


def test_clean_companies_fills_missing_country():
    raw = pd.DataFrame(
        {
            "company_id": [1, 2],
            "ticker": ["APEX", "HELIOS"],
            "company_name": ["Apex Computing Corp.", "Helios Solar Group"],
            "sector": ["technology", "Energy"],
            "exchange": ["NASDAQ", "NASDAQ"],
            "country": ["USA", None],
            "listed_date": ["2018-01-02", "2018-01-02"],
        }
    )
    out, _ = clean_companies(raw)
    assert out["country"].isna().sum() == 0
    assert set(out["sector"]) == {"Technology", "Energy"}

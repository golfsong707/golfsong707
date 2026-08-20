"""Extract stage: read the raw CSV files and surface their shape/content."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "companies": "companies.csv",
    "prices": "daily_prices.csv",
    "fundamentals": "fundamentals.csv",
}


def extract(raw_dir: str | Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    data: dict[str, pd.DataFrame] = {}
    for key, filename in FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file {path}. Run `python -m src.generate_raw` first."
            )
        data[key] = pd.read_csv(path)
    return data


def summarize(data: dict[str, pd.DataFrame]) -> str:
    lines = []
    for key, df in data.items():
        lines.append(f"  {key:<12} rows={len(df):<8} cols={list(df.columns)}")
    return "\n".join(lines)


if __name__ == "__main__":
    frames = extract()
    print(summarize(frames))

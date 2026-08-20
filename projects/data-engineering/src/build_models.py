"""Analytics-modelling layer: run versioned SQL models over the warehouse.

Model files live in ``sql/models/`` and are executed in filename order
(dbt-style). Each file is responsible for a single analytics table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .load import WAREHOUSE, _connect

MODELS_DIR = Path(__file__).resolve().parents[1] / "sql" / "models"


def build_models(db_path: str | Path = WAREHOUSE, models_dir: str | Path = MODELS_DIR) -> list[str]:
    models_dir = Path(models_dir)
    conn = _connect(db_path)
    built: list[str] = []
    try:
        for path in sorted(models_dir.glob("*.sql")):
            sql = path.read_text()
            conn.executescript(sql)
            conn.commit()
            built.append(path.name)
    finally:
        conn.close()
    return built


def model_table_counts(db_path: str | Path = WAREHOUSE) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        tables = ["sector_performance", "daily_returns_enriched", "top_daily_movers", "price_anomalies"]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        conn.close()

"""End-to-end orchestrator: generate -> extract -> transform -> load -> model."""

from __future__ import annotations

from pathlib import Path

from . import build_models, extract, generate_raw, load, transform


def run() -> None:
    print("=" * 70)
    print("MARKETSCOPE ETL PIPELINE")
    print("=" * 70)

    # 0) Generate raw data (idempotent).
    print("\n[1/5] Generating raw data...")
    generate_raw.main()

    # 1) Extract.
    print("\n[2/5] Extracting raw files...")
    raw = extract.extract()
    print(extract.summarize(raw))

    # 2) Transform.
    print("\n[3/5] Cleaning & transforming...")
    companies, comp_report = transform.clean_companies(raw["companies"])
    prices, price_report = transform.clean_prices(raw["prices"])
    fundamentals, fund_report = transform.clean_fundamentals(raw["fundamentals"], prices)

    for label, rep in [
        ("companies", comp_report),
        ("prices", price_report),
        ("fundamentals", fund_report),
    ]:
        print(f"  {label}: {rep}")

    # 3) Load into the warehouse.
    print("\n[4/5] Loading into star-schema warehouse...")
    db = load.load(companies, prices, fundamentals)
    print(f"  warehouse written to {db}")
    for table, count in load.table_counts(db).items():
        print(f"    {table:<18} {count} rows")

    # 4) Build analytics models.
    print("\n[5/5] Building analytics models...")
    built = build_models.build_models(db)
    for name in built:
        print(f"  applied {name}")
    for table, count in build_models.model_table_counts(db).items():
        print(f"    {table:<24} {count} rows")

    print("\nPipeline complete. ✅")


if __name__ == "__main__":
    run()

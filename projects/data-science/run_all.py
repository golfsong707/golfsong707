"""End-to-end run: data -> features -> train -> evaluate."""

from src.evaluate import make_plots
from src.features import build_features
from src.make_dataset import load_market_data
from src.train import train_and_evaluate


def main() -> None:
    print("=" * 70)
    print("NEXT-DAY PRICE DIRECTION — MODEL PIPELINE")
    print("=" * 70)

    data = load_market_data()
    features = build_features(data)
    print(f"[features] {len(features)} rows, {len(features['ticker'].unique())} tickers")

    metrics, _, importance = train_and_evaluate(features)

    print("\nModel comparison (test set):")
    print(metrics.round(4).to_string(index=False))

    print("\nTop features (random forest):")
    print(importance.head(8).round(4).to_string(index=False))

    print("\nPlots:")
    for p in make_plots():
        print(f"  {p}")


if __name__ == "__main__":
    main()

"""Generate a static analysis report (CSVs) without the dashboard UI."""

from __future__ import annotations

from pathlib import Path

from src.analysis import compute_kpis, correlation_matrix, headline, sector_index, top_movers
from src.make_dataset import load_market_data

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_market_data()

    kpis = compute_kpis(prices)
    sectors = sector_index(prices)
    corr = correlation_matrix(prices, level="sector")
    movers = top_movers(prices)
    stats = headline(prices)

    kpis.to_csv(REPORT_DIR / "ticker_kpis.csv", index=False)
    sectors.to_csv(REPORT_DIR / "sector_index.csv")
    corr.to_csv(REPORT_DIR / "sector_correlation.csv")
    movers.to_csv(REPORT_DIR / "top_movers.csv", index=False)

    print("=" * 60)
    print("MARKETSCOPE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Market total return      : {stats['total_return']:.2%}")
    print(f"Market annualized return : {stats['annualized_return']:.2%}")
    print(f"Market annualized vol    : {stats['annualized_vol']:.2%}")
    print(f"Market Sharpe ratio      : {stats['sharpe']:.2f}")
    print(f"Market max drawdown      : {stats['max_drawdown']:.2%}")
    print("\nTop 5 tickers by total return:")
    print(kpis.head(5)[["ticker", "sector", "total_return", "sharpe"]].to_string(index=False))
    print("\nBottom 5 tickers by total return:")
    print(kpis.tail(5)[["ticker", "sector", "total_return", "sharpe"]].to_string(index=False))
    print(f"\nReports written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()

"""Fetch and cache the default ETF universe.

Run with:

    uv --cache-dir .uv-cache run python scripts/fetch_etf_prices.py
"""

from __future__ import annotations

from sparse_portfolio import DEFAULT_ETF_UNIVERSE, get_prices


def main() -> None:
    prices = get_prices(
        DEFAULT_ETF_UNIVERSE,
        start="2015-01-01",
        end="2026-05-01",
        cache_path="data/raw/etf_prices.csv",
        refresh=True,
    )
    print(f"Downloaded {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print("Tickers:", ", ".join(prices.columns))
    print("Wrote data/raw/etf_prices.csv")


if __name__ == "__main__":
    main()

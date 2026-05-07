"""Run a monthly walk-forward historical-mean sparse portfolio backtest."""

from __future__ import annotations

from pathlib import Path

from sparse_portfolio import load_prices_csv, run_historical_mean_backtest


def main() -> None:
    prices = load_prices_csv("data/raw/etf_prices.csv")
    result = run_historical_mean_backtest(
        prices,
        lookback=252,
        cardinality=5,
        risk_aversion=10.0,
        covariance="ledoit_wolf",
        penalty_multiplier=0.1,
        start="2018-01-01",
        include_neal=True,
        neal_reads=32,
        neal_sweeps=500,
        seed=2026,
    )

    output_dir = Path("results/phase3")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.daily_returns.to_csv(output_dir / "historical_daily_returns.csv")
    result.summary.to_csv(output_dir / "historical_summary.csv", index=False)
    result.selections.to_csv(output_dir / "historical_selections.csv", index=False)

    print("Historical-mean monthly backtest summary")
    print(result.summary.to_string(index=False, float_format="{:.6f}".format))
    print()
    print("Wrote results/phase3/historical_daily_returns.csv")
    print("Wrote results/phase3/historical_summary.csv")
    print("Wrote results/phase3/historical_selections.csv")


if __name__ == "__main__":
    main()

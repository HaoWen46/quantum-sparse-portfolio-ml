"""Run a monthly walk-forward gradient boosting sparse portfolio backtest."""

from __future__ import annotations

from pathlib import Path

from sparse_portfolio import load_prices_csv
from sparse_portfolio.backtest import run_ml_forecast_backtest


def main() -> None:
    prices = load_prices_csv("data/raw/etf_prices.csv")
    result = run_ml_forecast_backtest(
        prices,
        model="gradient_boosting",
        lookback=252,
        train_lookback_rows=2_000,
        target_horizon=5,
        cardinality=5,
        risk_aversion=10.0,
        covariance="ledoit_wolf",
        penalty_multiplier=0.1,
        start="2020-01-01",
        include_exact=True,
        include_neal=False,
        seed=2026,
    )

    output_dir = Path("results/phase3")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.daily_returns.to_csv(output_dir / "gradient_boosting_daily_returns.csv")
    result.summary.to_csv(output_dir / "gradient_boosting_summary.csv", index=False)
    result.selections.to_csv(output_dir / "gradient_boosting_selections.csv", index=False)

    print("Gradient boosting monthly backtest summary")
    print(result.summary.to_string(index=False, float_format="{:.6f}".format))
    print()
    print("Wrote results/phase3/gradient_boosting_daily_returns.csv")
    print("Wrote results/phase3/gradient_boosting_summary.csv")
    print("Wrote results/phase3/gradient_boosting_selections.csv")


if __name__ == "__main__":
    main()

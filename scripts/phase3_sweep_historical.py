"""Sweep q and K for historical-mean sparse portfolio backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sparse_portfolio import load_prices_csv, run_historical_mean_backtest


def main() -> None:
    prices = load_prices_csv("data/raw/etf_prices.csv")
    rows: list[pd.DataFrame] = []
    q_values = [1.0, 3.0, 10.0, 30.0]
    k_values = [3, 5, 8]

    for cardinality in k_values:
        for risk_aversion in q_values:
            result = run_historical_mean_backtest(
                prices,
                lookback=252,
                cardinality=cardinality,
                risk_aversion=risk_aversion,
                covariance="ledoit_wolf",
                penalty_multiplier=0.1,
                start="2018-01-01",
                include_neal=False,
                include_exact=False,
            )
            summary = result.summary.copy()
            summary.insert(0, "risk_aversion", risk_aversion)
            summary.insert(0, "cardinality", cardinality)
            rows.append(summary)
            print(f"Finished K={cardinality}, q={risk_aversion}")

    sweep = pd.concat(rows, ignore_index=True)
    output_dir = Path("results/phase3")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "historical_q_k_sweep.csv"
    sweep.to_csv(output_path, index=False)

    compact = sweep[
        [
            "cardinality",
            "risk_aversion",
            "strategy",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "mean_turnover",
            "mean_objective_gap_vs_exact",
        ]
    ].sort_values(["cardinality", "risk_aversion", "sharpe"], ascending=[True, True, False])

    print()
    print("Historical q/K sweep")
    print(compact.to_string(index=False, float_format="{:.6f}".format))
    print()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

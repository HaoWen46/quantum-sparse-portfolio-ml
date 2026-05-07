"""Run a first real-ETF sparse selection snapshot.

Run after fetching prices:

    uv --cache-dir .uv-cache run python scripts/phase2_baseline_selection.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sparse_portfolio import (
    build_cardinality_qubo,
    exact_cardinality_optimum,
    greedy_forward_selection,
    latest_forecast_inputs,
    load_prices_csv,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    penalty_scale,
    portfolio_objective,
    top_k_by_mu,
)


def _solution_row(name: str, solution, tickers, mu, sigma, risk_aversion, exact_value):
    selected = [ticker for ticker, flag in zip(tickers, solution.x) if flag]
    base_value = portfolio_objective(solution.x, mu, sigma, risk_aversion)
    return {
        "solver": name,
        "selected": ",".join(selected),
        "cardinality": int(solution.x.sum()),
        "base_objective": base_value,
        "gap_vs_exact": base_value - exact_value,
        "mean_mu_daily": float(pd.Series(mu, index=tickers).loc[selected].mean()),
    }


def main() -> None:
    prices = load_prices_csv("data/raw/etf_prices.csv")
    cardinality = 5
    risk_aversion = 10.0
    selection_risk_aversion = risk_aversion / cardinality
    lookback = 252

    inputs = latest_forecast_inputs(
        prices,
        lookback=lookback,
        covariance="ledoit_wolf",
    )
    penalty = 0.1 * penalty_scale(inputs.mu, inputs.sigma, selection_risk_aversion)
    qubo, offset = build_cardinality_qubo(
        inputs.mu,
        inputs.sigma,
        cardinality,
        selection_risk_aversion,
        penalty,
    )

    exact = exact_cardinality_optimum(
        inputs.mu,
        inputs.sigma,
        cardinality,
        selection_risk_aversion,
    )

    rows = [
        _solution_row(
            "exact_constrained",
            exact,
            inputs.tickers,
            inputs.mu,
            inputs.sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _solution_row(
            "top_k_mu",
            top_k_by_mu(
                inputs.mu, inputs.sigma, cardinality, selection_risk_aversion
            ),
            inputs.tickers,
            inputs.mu,
            inputs.sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _solution_row(
            "greedy",
            greedy_forward_selection(
                inputs.mu, inputs.sigma, cardinality, selection_risk_aversion
            ),
            inputs.tickers,
            inputs.mu,
            inputs.sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _solution_row(
            "local_search",
            local_search_swaps(
                inputs.mu, inputs.sigma, cardinality, selection_risk_aversion
            ),
            inputs.tickers,
            inputs.mu,
            inputs.sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _solution_row(
            "neal_qubo",
            neal_simulated_annealing_qubo(
                qubo,
                offset,
                seed=2026,
                num_reads=128,
                num_sweeps=2_000,
                beta_range=(0.1, 100.0),
            ),
            inputs.tickers,
            inputs.mu,
            inputs.sigma,
            selection_risk_aversion,
            exact.value,
        ),
    ]

    result = pd.DataFrame(rows)
    output_dir = Path("results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_selection.csv"
    result.to_csv(output_path, index=False)

    print(f"Latest date: {inputs.returns_window.index.max().date()}")
    print(f"Lookback: {lookback} trading days")
    print(f"Portfolio risk aversion: {risk_aversion}")
    print(f"Binary-selection risk aversion: {selection_risk_aversion}")
    print(f"Penalty: {penalty:.8f}")
    print()
    print(result.to_string(index=False, float_format="{:.8f}".format))
    print()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

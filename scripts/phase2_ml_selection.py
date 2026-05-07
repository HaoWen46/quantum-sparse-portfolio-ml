"""Run latest ETF selection with course-covered ML return forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sparse_portfolio import (
    build_cardinality_qubo,
    exact_cardinality_optimum,
    latest_ml_return_forecast,
    ledoit_wolf_covariance,
    load_prices_csv,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    penalty_scale,
    portfolio_objective,
    returns_from_prices,
    top_k_by_mu,
)


def _row(name: str, solution, tickers, mu, sigma, risk_aversion, exact_value):
    selected = [ticker for ticker, flag in zip(tickers, solution.x) if flag]
    base_value = portfolio_objective(solution.x, mu, sigma, risk_aversion)
    return {
        "solver": name,
        "selected": ",".join(selected),
        "cardinality": int(solution.x.sum()),
        "base_objective": base_value,
        "gap_vs_exact": base_value - exact_value,
    }


def main() -> None:
    prices = load_prices_csv("data/raw/etf_prices.csv")
    returns = returns_from_prices(prices)
    lookback = 252
    cardinality = 5
    risk_aversion = 10.0
    selection_risk_aversion = risk_aversion / cardinality
    model = "lasso"

    mu_series = latest_ml_return_forecast(
        prices,
        model=model,
        target_horizon=5,
        train_lookback_rows=2_000,
        random_state=2026,
    )
    tickers = list(mu_series.index)
    returns_window = returns[tickers].tail(lookback).dropna(axis=0, how="any")
    mu = mu_series.to_numpy()
    sigma = ledoit_wolf_covariance(returns_window)
    penalty = 0.1 * penalty_scale(mu, sigma, selection_risk_aversion)
    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality, selection_risk_aversion, penalty
    )

    exact = exact_cardinality_optimum(mu, sigma, cardinality, selection_risk_aversion)
    rows = [
        _row(
            "exact_constrained",
            exact,
            tickers,
            mu,
            sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _row(
            "top_k_mu",
            top_k_by_mu(mu, sigma, cardinality, selection_risk_aversion),
            tickers,
            mu,
            sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _row(
            "local_search",
            local_search_swaps(mu, sigma, cardinality, selection_risk_aversion),
            tickers,
            mu,
            sigma,
            selection_risk_aversion,
            exact.value,
        ),
        _row(
            "neal_qubo",
            neal_simulated_annealing_qubo(
                qubo,
                offset,
                seed=2026,
                num_reads=128,
                num_sweeps=2_000,
                beta_range=(0.1, 100.0),
            ),
            tickers,
            mu,
            sigma,
            selection_risk_aversion,
            exact.value,
        ),
    ]
    result = pd.DataFrame(rows)

    output_dir = Path("results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ml_selection_{model}.csv"
    result.to_csv(output_path, index=False)

    print(f"Model: {model}")
    print(f"Latest feature date: {prices.index.max().date()}")
    print(f"Portfolio risk aversion: {risk_aversion}")
    print(f"Binary-selection risk aversion: {selection_risk_aversion}")
    print()
    print(result.to_string(index=False, float_format="{:.8f}".format))
    print()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

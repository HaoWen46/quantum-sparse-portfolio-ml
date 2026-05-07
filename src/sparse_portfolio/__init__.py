"""Utilities for sparse QUBO portfolio experiments."""

from sparse_portfolio.qubo import (
    BinarySolution,
    build_cardinality_qubo,
    exact_cardinality_optimum,
    exact_qubo_optimum,
    penalized_objective,
    portfolio_objective,
    qubo_energy,
)
from sparse_portfolio.backtest import (
    BacktestResult,
    monthly_rebalance_dates,
    run_historical_mean_backtest,
    run_ml_forecast_backtest,
    summarize_backtest,
)
from sparse_portfolio.data import (
    DEFAULT_ETF_UNIVERSE,
    download_prices_yfinance,
    get_prices,
    load_prices_csv,
    returns_from_prices,
    save_prices_csv,
)
from sparse_portfolio.experiments import (
    SyntheticPortfolioInstance,
    benchmark_solvers,
    generate_synthetic_instance,
    penalty_scale,
    run_penalty_sweep,
    summarize_penalty_sweep,
)
from sparse_portfolio.features import (
    make_forward_returns,
    make_return_feature_panel,
    make_return_features,
)
from sparse_portfolio.forecasts import (
    ForecastInputs,
    historical_mean_forecast,
    latest_forecast_inputs,
    latest_ml_return_forecast,
)
from sparse_portfolio.risk import ledoit_wolf_covariance, sample_covariance
from sparse_portfolio.solvers import (
    dimod_exact_qubo,
    greedy_forward_selection,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    simulated_annealing_qubo,
    top_k_by_mu,
)

__all__ = [
    "BinarySolution",
    "BacktestResult",
    "DEFAULT_ETF_UNIVERSE",
    "ForecastInputs",
    "SyntheticPortfolioInstance",
    "benchmark_solvers",
    "build_cardinality_qubo",
    "download_prices_yfinance",
    "exact_cardinality_optimum",
    "exact_qubo_optimum",
    "dimod_exact_qubo",
    "generate_synthetic_instance",
    "get_prices",
    "historical_mean_forecast",
    "latest_forecast_inputs",
    "ledoit_wolf_covariance",
    "load_prices_csv",
    "latest_ml_return_forecast",
    "make_forward_returns",
    "make_return_feature_panel",
    "make_return_features",
    "monthly_rebalance_dates",
    "penalty_scale",
    "penalized_objective",
    "portfolio_objective",
    "qubo_energy",
    "returns_from_prices",
    "run_penalty_sweep",
    "run_historical_mean_backtest",
    "run_ml_forecast_backtest",
    "sample_covariance",
    "save_prices_csv",
    "summarize_backtest",
    "summarize_penalty_sweep",
    "greedy_forward_selection",
    "local_search_swaps",
    "neal_simulated_annealing_qubo",
    "simulated_annealing_qubo",
    "top_k_by_mu",
]

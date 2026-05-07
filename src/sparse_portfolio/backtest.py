"""Walk-forward backtesting for sparse portfolio strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from sparse_portfolio.data import returns_from_prices
from sparse_portfolio.forecasts import historical_mean_forecast
from sparse_portfolio.forecasts import latest_ml_return_forecast
from sparse_portfolio.qubo import build_cardinality_qubo, exact_cardinality_optimum
from sparse_portfolio.risk import ledoit_wolf_covariance, sample_covariance
from sparse_portfolio.solvers import (
    local_search_swaps,
    neal_simulated_annealing_qubo,
    top_k_by_mu,
)
from sparse_portfolio.experiments import penalty_scale


@dataclass(frozen=True)
class BacktestResult:
    """Daily returns, summary metrics, and selections from a backtest."""

    daily_returns: pd.DataFrame
    summary: pd.DataFrame
    selections: pd.DataFrame


def monthly_rebalance_dates(
    returns: pd.DataFrame,
    *,
    lookback: int,
    start: str | None = None,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Return month-end trading dates with enough prior data."""

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    dates = returns.dropna(how="all").index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        raise ValueError("no dates available after applying start/end filters")

    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
    eligible = [date for date in month_ends if returns.loc[:date].shape[0] >= lookback]
    return pd.DatetimeIndex(eligible)


def run_historical_mean_backtest(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    cardinality: int = 5,
    risk_aversion: float = 10.0,
    covariance: str = "ledoit_wolf",
    penalty_multiplier: float = 0.1,
    start: str | None = "2018-01-01",
    end: str | None = None,
    include_neal: bool = True,
    include_exact: bool = True,
    neal_reads: int = 32,
    neal_sweeps: int = 500,
    seed: int = 0,
) -> BacktestResult:
    """Backtest historical-mean forecasts with sparse portfolio solvers."""

    if not 0 < cardinality <= prices.shape[1]:
        raise ValueError("cardinality must be between 1 and number of assets")

    returns = returns_from_prices(prices).dropna(axis=1, how="any")
    rebalance_dates = monthly_rebalance_dates(
        returns,
        lookback=lookback,
        start=start,
        end=end,
    )
    if len(rebalance_dates) < 2:
        raise ValueError("need at least two rebalance dates")

    strategy_returns: dict[str, pd.Series] = {}
    selection_rows: list[dict[str, object]] = []
    previous_weights: dict[str, pd.Series] = {}

    for rebalance_index, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[rebalance_index + 1]
        train_window = returns.loc[:rebalance_date].tail(lookback).dropna(how="any")
        tickers = list(train_window.columns)
        mu = historical_mean_forecast(train_window).to_numpy()
        sigma = _covariance(train_window, covariance)
        selection_risk_aversion = _binary_selection_risk_aversion(
            risk_aversion, cardinality
        )
        penalty = penalty_multiplier * penalty_scale(
            mu, sigma, selection_risk_aversion
        )
        qubo, offset = build_cardinality_qubo(
            mu,
            sigma,
            cardinality,
            selection_risk_aversion,
            penalty,
        )

        exact = (
            exact_cardinality_optimum(mu, sigma, cardinality, selection_risk_aversion)
            if include_exact
            else None
        )
        solutions = {
            "top_k_mu": top_k_by_mu(
                mu, sigma, cardinality, selection_risk_aversion
            ),
            "local_search": local_search_swaps(
                mu, sigma, cardinality, selection_risk_aversion
            ),
        }
        if exact is not None:
            solutions = {"exact": exact, **solutions}
        if include_neal:
            solutions["neal"] = neal_simulated_annealing_qubo(
                qubo,
                offset,
                seed=seed + rebalance_index,
                num_reads=neal_reads,
                num_sweeps=neal_sweeps,
                beta_range=(0.1, 100.0),
            )

        hold_returns = returns.loc[
            (returns.index > rebalance_date) & (returns.index <= next_rebalance_date),
            tickers,
        ]
        if hold_returns.empty:
            continue

        for strategy, solution in solutions.items():
            weights = _equal_weights(tickers, solution.x)
            period_returns = hold_returns.mul(weights, axis=1).sum(axis=1)
            if strategy in strategy_returns:
                strategy_returns[strategy] = pd.concat(
                    [strategy_returns[strategy], period_returns]
                )
            else:
                strategy_returns[strategy] = period_returns

            previous = previous_weights.get(strategy)
            turnover = np.nan if previous is None else _turnover(previous, weights)
            previous_weights[strategy] = weights
            selected = list(weights[weights > 0].index)
            selection_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "next_rebalance_date": next_rebalance_date,
                    "strategy": strategy,
                    "selected": ",".join(selected),
                    "cardinality": int(solution.x.sum()),
                    "feasible": int(solution.x.sum()) == cardinality,
                    "turnover": turnover,
                    "objective_gap_vs_exact": (
                        float(
                            _base_value(
                                solution.x, mu, sigma, selection_risk_aversion
                            )
                            - exact.value
                        )
                        if exact is not None
                        else np.nan
                    ),
                }
            )

    daily_returns = pd.DataFrame(strategy_returns).sort_index()
    summary = summarize_backtest(
        daily_returns,
        pd.DataFrame(selection_rows),
    )
    return BacktestResult(
        daily_returns=daily_returns,
        summary=summary,
        selections=pd.DataFrame(selection_rows),
    )


def run_ml_forecast_backtest(
    prices: pd.DataFrame,
    *,
    model: str = "lasso",
    lookback: int = 252,
    train_lookback_rows: int = 2_000,
    target_horizon: int = 5,
    cardinality: int = 5,
    risk_aversion: float = 10.0,
    covariance: str = "ledoit_wolf",
    penalty_multiplier: float = 0.1,
    start: str | None = "2020-01-01",
    end: str | None = None,
    include_exact: bool = True,
    include_neal: bool = False,
    neal_reads: int = 32,
    neal_sweeps: int = 500,
    seed: int = 0,
) -> BacktestResult:
    """Backtest course-covered ML forecasts with sparse portfolio solvers."""

    if not 0 < cardinality <= prices.shape[1]:
        raise ValueError("cardinality must be between 1 and number of assets")

    returns = returns_from_prices(prices).dropna(axis=1, how="any")
    rebalance_dates = monthly_rebalance_dates(
        returns,
        lookback=lookback,
        start=start,
        end=end,
    )
    if len(rebalance_dates) < 2:
        raise ValueError("need at least two rebalance dates")

    strategy_returns: dict[str, pd.Series] = {}
    selection_rows: list[dict[str, object]] = []
    previous_weights: dict[str, pd.Series] = {}

    for rebalance_index, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[rebalance_index + 1]
        price_history = prices.loc[:rebalance_date]
        mu_series = latest_ml_return_forecast(
            price_history,
            model=model,
            target_horizon=target_horizon,
            train_lookback_rows=train_lookback_rows,
            random_state=seed + rebalance_index,
        )
        tickers = [ticker for ticker in mu_series.index if ticker in returns.columns]
        train_window = (
            returns.loc[:rebalance_date, tickers]
            .tail(lookback)
            .dropna(axis=0, how="any")
        )
        if train_window.shape[0] < 2:
            continue
        tickers = list(train_window.columns)
        mu = mu_series.reindex(tickers).to_numpy()
        sigma = _covariance(train_window, covariance)
        selection_risk_aversion = _binary_selection_risk_aversion(
            risk_aversion, cardinality
        )
        penalty = penalty_multiplier * penalty_scale(
            mu, sigma, selection_risk_aversion
        )
        qubo, offset = build_cardinality_qubo(
            mu,
            sigma,
            cardinality,
            selection_risk_aversion,
            penalty,
        )

        exact = (
            exact_cardinality_optimum(mu, sigma, cardinality, selection_risk_aversion)
            if include_exact
            else None
        )
        solutions = {
            "top_k_mu": top_k_by_mu(
                mu, sigma, cardinality, selection_risk_aversion
            ),
            "local_search": local_search_swaps(
                mu, sigma, cardinality, selection_risk_aversion
            ),
        }
        if exact is not None:
            solutions = {"exact": exact, **solutions}
        if include_neal:
            solutions["neal"] = neal_simulated_annealing_qubo(
                qubo,
                offset,
                seed=seed + rebalance_index,
                num_reads=neal_reads,
                num_sweeps=neal_sweeps,
                beta_range=(0.1, 100.0),
            )

        hold_returns = returns.loc[
            (returns.index > rebalance_date) & (returns.index <= next_rebalance_date),
            tickers,
        ]
        if hold_returns.empty:
            continue

        for strategy, solution in solutions.items():
            weights = _equal_weights(tickers, solution.x)
            period_returns = hold_returns.mul(weights, axis=1).sum(axis=1)
            if strategy in strategy_returns:
                strategy_returns[strategy] = pd.concat(
                    [strategy_returns[strategy], period_returns]
                )
            else:
                strategy_returns[strategy] = period_returns

            previous = previous_weights.get(strategy)
            turnover = np.nan if previous is None else _turnover(previous, weights)
            previous_weights[strategy] = weights
            selected = list(weights[weights > 0].index)
            selection_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "next_rebalance_date": next_rebalance_date,
                    "strategy": strategy,
                    "selected": ",".join(selected),
                    "cardinality": int(solution.x.sum()),
                    "feasible": int(solution.x.sum()) == cardinality,
                    "turnover": turnover,
                    "objective_gap_vs_exact": (
                        float(
                            _base_value(
                                solution.x, mu, sigma, selection_risk_aversion
                            )
                            - exact.value
                        )
                        if exact is not None
                        else np.nan
                    ),
                }
            )

    daily_returns = pd.DataFrame(strategy_returns).sort_index()
    selections = pd.DataFrame(selection_rows)
    return BacktestResult(
        daily_returns=daily_returns,
        summary=summarize_backtest(daily_returns, selections),
        selections=selections,
    )


def summarize_backtest(
    daily_returns: pd.DataFrame,
    selections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute standard performance metrics from daily strategy returns."""

    rows: list[dict[str, float | str]] = []
    for strategy in daily_returns.columns:
        returns = daily_returns[strategy].dropna()
        if returns.empty:
            continue
        equity = (1.0 + returns).cumprod()
        annual_return = equity.iloc[-1] ** (252 / len(returns)) - 1.0
        annual_volatility = returns.std(ddof=1) * np.sqrt(252)
        annualized_mean_return = returns.mean() * 252
        sharpe = (
            annualized_mean_return / annual_volatility
            if annual_volatility > 0
            else np.nan
        )
        drawdown = equity / equity.cummax() - 1.0
        row = {
            "strategy": strategy,
            "n_days": float(len(returns)),
            "annual_return": float(annual_return),
            "annualized_mean_return": float(annualized_mean_return),
            "annual_volatility": float(annual_volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown.min()),
            "cumulative_return": float(equity.iloc[-1] - 1.0),
        }
        if selections is not None and not selections.empty:
            subset = selections[selections["strategy"] == strategy]
            row["mean_turnover"] = float(subset["turnover"].mean())
            row["mean_objective_gap_vs_exact"] = float(
                subset["objective_gap_vs_exact"].mean()
            )
            if "feasible" in subset:
                row["feasibility_rate"] = float(subset["feasible"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def _covariance(returns: pd.DataFrame, covariance: str) -> np.ndarray:
    if covariance == "ledoit_wolf":
        return ledoit_wolf_covariance(returns)
    if covariance == "sample":
        return sample_covariance(returns)
    raise ValueError("covariance must be 'ledoit_wolf' or 'sample'")


def _binary_selection_risk_aversion(
    portfolio_risk_aversion: float,
    cardinality: int,
) -> float:
    """Convert equal-weight portfolio risk aversion to binary subset scale."""

    if cardinality <= 0:
        raise ValueError("cardinality must be positive")
    return float(portfolio_risk_aversion / cardinality)


def _equal_weights(tickers: Iterable[str], x: np.ndarray) -> pd.Series:
    tickers = list(tickers)
    selected = np.asarray(x, dtype=int)
    if selected.sum() <= 0:
        raise ValueError("cannot build equal weights for empty selection")
    weights = pd.Series(0.0, index=tickers)
    weights.iloc[np.flatnonzero(selected == 1)] = 1.0 / selected.sum()
    return weights


def _turnover(previous: pd.Series, current: pd.Series) -> float:
    aligned = pd.concat([previous, current], axis=1).fillna(0.0)
    return float(0.5 * np.abs(aligned.iloc[:, 1] - aligned.iloc[:, 0]).sum())


def _base_value(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
) -> float:
    x_float = np.asarray(x, dtype=float)
    return float(risk_aversion * x_float @ sigma @ x_float - mu @ x_float)

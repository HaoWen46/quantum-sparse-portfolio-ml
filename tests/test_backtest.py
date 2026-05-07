import numpy as np
import pandas as pd
import pytest

from sparse_portfolio import monthly_rebalance_dates, returns_from_prices
from sparse_portfolio.backtest import (
    _binary_selection_risk_aversion,
    run_historical_mean_backtest,
    run_ml_forecast_backtest,
    summarize_backtest,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=320)
    trend = np.linspace(0, 1, len(dates))
    return pd.DataFrame(
        {
            "A": 100 + 12 * trend + np.sin(trend * 8),
            "B": 90 + 8 * trend + np.cos(trend * 5),
            "C": 75 + 4 * trend + np.sin(trend * 7) * 0.5,
            "D": 60 + 2 * trend + np.cos(trend * 6) * 0.5,
        },
        index=dates,
    )


def test_monthly_rebalance_dates_are_month_ends_after_lookback():
    returns = returns_from_prices(_prices())
    dates = monthly_rebalance_dates(returns, lookback=60, start="2020-06-01")

    assert len(dates) > 0
    assert dates.min() >= pd.Timestamp("2020-06-01")


def test_run_historical_mean_backtest_produces_outputs():
    result = run_historical_mean_backtest(
        _prices(),
        lookback=60,
        cardinality=2,
        risk_aversion=1.0,
        start="2020-06-01",
        include_neal=False,
    )

    assert not result.daily_returns.empty
    assert {"exact", "top_k_mu", "local_search"}.issubset(result.daily_returns.columns)
    assert not result.summary.empty
    assert not result.selections.empty


def test_binary_selection_risk_aversion_matches_equal_weight_scale():
    assert _binary_selection_risk_aversion(10.0, cardinality=5) == 2.0


def test_summarize_backtest_uses_standard_annualized_sharpe():
    returns = pd.DataFrame({"strategy": [0.01, -0.005, 0.015, 0.0]})

    summary = summarize_backtest(returns)

    expected = returns["strategy"].mean() / returns["strategy"].std(ddof=1) * np.sqrt(252)
    assert summary["sharpe"].iloc[0] == pytest.approx(expected)


def test_ml_backtest_can_skip_exact_enumeration():
    result = run_ml_forecast_backtest(
        _prices(),
        lookback=60,
        train_lookback_rows=120,
        target_horizon=3,
        cardinality=2,
        risk_aversion=1.0,
        start="2020-10-01",
        include_exact=False,
        include_neal=False,
        seed=7,
    )

    assert not result.daily_returns.empty
    assert "exact" not in result.daily_returns.columns
    assert {"top_k_mu", "local_search"}.issubset(result.daily_returns.columns)
    assert result.selections["objective_gap_vs_exact"].isna().all()

import numpy as np
import pandas as pd

from sparse_portfolio import historical_mean_forecast, latest_forecast_inputs


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "SPY": np.linspace(100, 120, len(dates)),
            "TLT": np.linspace(80, 82, len(dates)),
            "GLD": np.linspace(150, 165, len(dates)),
        },
        index=dates,
    )


def test_historical_mean_forecast_returns_series():
    returns = _prices().pct_change(fill_method=None).dropna()

    forecast = historical_mean_forecast(returns, lookback=10)

    assert forecast.index.tolist() == ["SPY", "TLT", "GLD"]
    assert np.isfinite(forecast.to_numpy()).all()


def test_latest_forecast_inputs_aligns_mu_sigma_tickers():
    inputs = latest_forecast_inputs(_prices(), lookback=20, covariance="ledoit_wolf")

    assert inputs.tickers == ["SPY", "TLT", "GLD"]
    assert inputs.mu.shape == (3,)
    assert inputs.sigma.shape == (3, 3)
    assert inputs.returns_window.shape == (20, 3)

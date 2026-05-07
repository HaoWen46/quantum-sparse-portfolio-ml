import numpy as np
import pandas as pd

from sparse_portfolio import latest_ml_return_forecast
from sparse_portfolio.forecasts import _date_ordered_cv_splits


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    trend = np.linspace(0, 1, len(dates))
    return pd.DataFrame(
        {
            "A": 100 + 10 * trend + np.sin(trend * 10),
            "B": 90 + 5 * trend + np.cos(trend * 8),
            "C": 70 + 2 * trend + np.sin(trend * 12) * 0.5,
        },
        index=dates,
    )


def test_latest_ml_return_forecast_lasso_returns_latest_ticker_scores():
    forecast = latest_ml_return_forecast(
        _prices(),
        model="lasso",
        target_horizon=3,
        train_lookback_rows=120,
        random_state=1,
    )

    assert forecast.index.tolist() == ["A", "B", "C"]
    assert np.isfinite(forecast.to_numpy()).all()


def test_date_ordered_cv_splits_validate_only_future_dates():
    index = (
        pd.DataFrame(index=_prices().index, columns=["A", "B", "C"])
        .stack(future_stack=True)
        .index
    )
    index.names = ["date", "ticker"]

    splits = _date_ordered_cv_splits(index, max_splits=4)
    row_dates = pd.Index(index.get_level_values("date"))

    assert len(splits) == 4
    for train_indices, validation_indices in splits:
        assert row_dates[train_indices].max() < row_dates[validation_indices].min()
